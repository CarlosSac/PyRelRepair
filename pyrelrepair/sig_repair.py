"""SigRepair: Stage 2 - Function Signature-based Repair.

Retrieves relevant function signatures from the project codebase using
nomic-embed-text (via Ollama) and injects them into the LLM prompt. Per the paper:

1. Query Rewriting: LLM generates 2 root causes + 5 candidate function names
2. Dataset Creation: Collect functions from same file + type-related files
3. Indexing: nomic-embed-text encodes signature || docstring
4. Retrieval: Cosine similarity, top 25
5. Generation: Repeat 20 times (each with fresh query rewriting), 1 patch/iter
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from .base_repair import BugInfo, PatchCandidate, ValidatorFn
from .code_parser import (
    FunctionInfo,
    extract_functions_from_directory,
    extract_functions_from_file,
    get_variable_types,
)
from .config import Config
from .llm import OllamaClient
from .prompt_utils import code_with_linenos, fault_context
from .prompts import SIG_QUERY_REWRITE_PROMPT, SIG_REPAIR_PROMPT, format_signatures_block
from .retrieval import retrieve_similar_signatures
from .validator import (
    apply_patch,
    extract_function_from_response,
    validate_patch,
    validate_syntax,
)

logger = logging.getLogger(__name__)


def _build_candidate_dataset(bug: BugInfo) -> list[FunctionInfo]:
    """Build the candidate function dataset for SigRepair.

    Collects:
    1. File-based functions: all functions in the same file as the buggy function
    2. Variable-based functions: functions from files defining types used in the buggy function
    """
    candidates = []
    seen = set()

    # 1. File-based functions (same file)
    file_functions = extract_functions_from_file(bug.resolved_file_path)
    for func in file_functions:
        if func.name != bug.function_name:
            key = (str(func.file_path), func.name, func.start_line)
            if key not in seen:
                candidates.append(func)
                seen.add(key)

    # 2. Variable-based functions (type-related)
    user_types = get_variable_types(bug.buggy_function)
    if user_types and bug.project_dir:
        all_project_functions = extract_functions_from_directory(
            bug.project_dir, exclude_tests=True
        )
        for func in all_project_functions:
            if func.name == bug.function_name and str(func.file_path) == str(bug.file_path):
                continue
            # Check if the function is related to any user-defined type
            for utype in user_types:
                if (
                    utype.lower() in func.class_name.lower() if func.class_name else False
                ) or utype.lower() in func.name.lower():
                    key = (str(func.file_path), func.name, func.start_line)
                    if key not in seen:
                        candidates.append(func)
                        seen.add(key)
                    break

    logger.info(
        "SigRepair dataset: %d candidates (%d file-based, %d type-based)",
        len(candidates),
        len(file_functions) - 1,
        len(candidates) - (len(file_functions) - 1),
    )
    return candidates


def _query_rewrite(llm: OllamaClient, bug: BugInfo) -> tuple[str, int, int]:
    """Use LLM to generate root causes and candidate function names.

    Returns (query_text, prompt_tokens, completion_tokens).
    """
    fault_line_relative = bug.fault_line - bug.start_line + 1
    prompt = SIG_QUERY_REWRITE_PROMPT.format(
        buggy_function=bug.buggy_function,
        fault_line=fault_line_relative,
    )

    response = llm.generate(prompt, temperature=0.7, num_return=1)[0]
    text = response.text.strip()

    root_causes = []
    functions = []

    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("ROOT_CAUSE_1:"):
            root_causes.append(line.split(":", 1)[1].strip())
        elif line.startswith("ROOT_CAUSE_2:"):
            root_causes.append(line.split(":", 1)[1].strip())
        elif line.startswith("FUNCTIONS:"):
            funcs_str = line.split(":", 1)[1].strip()
            functions = [f.strip() for f in funcs_str.split(",")]

    query = " ".join(root_causes + functions)
    if not query.strip():
        query = f"{bug.function_name} {bug.error_message[:200]}"

    logger.debug("SigRepair query rewrite: %s", query[:200])
    return query, response.prompt_tokens, response.completion_tokens


def sig_repair(
    bug: BugInfo,
    config: Config,
    validator: ValidatorFn | None = None,
) -> tuple[list[PatchCandidate], dict[str, int]]:
    """Execute the SigRepair stage.

    Runs config.sig_num_iterations iterations (default: 20), each time:
    1. Rewrites the query with the LLM
    2. Retrieves top-25 similar function signatures
    3. Generates config.sig_num_patches_per_iter candidate patches

    Returns:
        (candidates, token_stats) where token_stats has keys
        'prompt_tokens' and 'completion_tokens'.
    """
    llm = OllamaClient(config)
    candidates: list[PatchCandidate] = []
    token_stats: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

    candidate_functions = _build_candidate_dataset(bug)
    if not candidate_functions:
        logger.warning("SigRepair: no candidate functions found, skipping")
        return candidates, token_stats

    logger.info(
        "SigRepair: starting %d iterations for %s",
        config.sig_num_iterations,
        bug.bug_id,
    )

    fault_line_relative = bug.fault_line - bug.start_line + 1
    fault_content, fault_ctx = fault_context(bug.buggy_function, fault_line_relative)

    for i in range(config.sig_num_iterations):
        logger.info("SigRepair iteration %d/%d", i + 1, config.sig_num_iterations)

        # 1. Query rewriting
        query, qpt, qct = _query_rewrite(llm, bug)
        token_stats["prompt_tokens"] += qpt
        token_stats["completion_tokens"] += qct

        # 2. Retrieve top-K signatures
        retrieved = retrieve_similar_signatures(
            query_text=query,
            candidate_functions=candidate_functions,
            top_k=config.sig_top_k,
            embed_model=config.embed_model,
            base_url=config.ollama_base_url,
        )

        sig_block = format_signatures_block([
            {"signature": func.signature, "docstring": func.docstring}
            for func, _ in retrieved
        ])

        # 3. Generate patch(es)
        prompt = SIG_REPAIR_PROMPT.format(
            file_path=bug.file_path,
            function_name=bug.function_name,
            buggy_function_with_linenos=code_with_linenos(bug.buggy_function),
            fault_line_relative=fault_line_relative,
            fault_line_content=fault_content,
            fault_context=fault_ctx,
            error_message=bug.error_message,
            test_output=bug.test_output,
            signatures=sig_block,
        )

        responses = llm.generate(prompt, num_return=config.sig_num_patches_per_iter)
        for response in responses:
            token_stats["prompt_tokens"] += response.prompt_tokens
            token_stats["completion_tokens"] += response.completion_tokens

            patched_func = extract_function_from_response(response.text)
            if patched_func is None:
                logger.warning("SigRepair iter %d: could not extract function", i + 1)
                continue

            if not validate_syntax(patched_func):
                logger.warning("SigRepair iter %d: syntax error in patch", i + 1)
                continue

            candidate = PatchCandidate(patch_code=patched_func, stage="SigRepair")

            if bug.project_dir and bug.resolved_file_path.exists():
                patched_source = apply_patch(
                    original_file=bug.resolved_file_path,
                    buggy_function_name=bug.function_name,
                    patched_function=patched_func,
                    start_line=bug.start_line,
                    end_line=bug.end_line,
                )
                result = validate_patch(
                    project_dir=bug.project_dir,
                    original_file=bug.resolved_file_path,
                    patched_source=patched_source,
                    test_file=bug.test_file,
                    config=config,
                )
                candidate.validation = result
                if result.passed:
                    logger.info("SigRepair iter %d: patch PASSED all tests!", i + 1)
                    candidates.append(candidate)
                    return candidates, token_stats

            elif validator is not None:
                result = validator(candidate)
                if result is not None:
                    candidate.validation = result
                    if result.passed:
                        logger.info("SigRepair iter %d: patch PASSED (external validator)!", i + 1)
                        candidates.append(candidate)
                        return candidates, token_stats

            candidates.append(candidate)

    return candidates, token_stats
