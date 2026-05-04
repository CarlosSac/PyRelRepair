"""BaseRepair: Stage 1 of the PyRelRepair pipeline.

Generates a single candidate patch using only the buggy function,
error messages, and failing test output.
This is the first repair attempt and the no-retrieval baseline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import Config
from .llm import OllamaClient
from .prompts import BASE_REPAIR_PROMPT
from .prompt_utils import code_with_linenos, fault_context
from .validator import (
    ValidationResult,
    apply_patch,
    extract_function_from_response,
    validate_patch,
    validate_syntax,
)

logger = logging.getLogger(__name__)

ValidatorFn = Callable[["PatchCandidate"], "ValidationResult | None"]


@dataclass
class BugInfo:
    """All information needed to describe a bug for repair."""
    bug_id: str
    file_path: Path
    function_name: str
    buggy_function: str
    fault_line: int
    start_line: int
    end_line: int
    error_message: str
    test_output: str
    test_file: Path | None = None
    project_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.start_line < 1:
            raise ValueError("start_line must be >= 1")
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        if self.fault_line < self.start_line or self.fault_line > self.end_line:
            raise ValueError("fault_line must be within [start_line, end_line]")
        lines = self.buggy_function.splitlines()
        if not lines:
            raise ValueError("buggy_function must not be empty")
        fault_line_relative = self.fault_line - self.start_line + 1
        if fault_line_relative < 1 or fault_line_relative > len(lines):
            raise ValueError(
                "fault_line is out of bounds for buggy_function content "
                f"(relative line {fault_line_relative}, function has {len(lines)} lines)"
            )

    @property
    def resolved_file_path(self) -> Path:
        """Absolute path to the bug file, resolved against project_dir if needed."""
        if self.file_path.is_absolute():
            return self.file_path
        if self.project_dir is not None:
            return self.project_dir / self.file_path
        return self.file_path


@dataclass
class PatchCandidate:
    """A generated patch candidate with its validation result."""
    patch_code: str
    stage: str
    validation: ValidationResult | None = None

    @property
    def is_valid(self) -> bool:
        return self.validation is not None and self.validation.passed


def base_repair(
    bug: BugInfo,
    config: Config,
    validator: ValidatorFn | None = None,
) -> tuple[list[PatchCandidate], dict[str, int]]:
    """Execute the BaseRepair stage.

    Generates config.base_num_patches candidate patches (default: 1)
    using only the buggy function, error message, and test output.

    Returns:
        (candidates, token_stats) where token_stats has keys
        'prompt_tokens' and 'completion_tokens'.
    """
    llm = OllamaClient(config)
    candidates = []
    token_stats: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

    fault_line_relative = bug.fault_line - bug.start_line + 1
    fault_content, fault_ctx = fault_context(bug.buggy_function, fault_line_relative)
    prompt = BASE_REPAIR_PROMPT.format(
        file_path=bug.file_path,
        function_name=bug.function_name,
        buggy_function_with_linenos=code_with_linenos(bug.buggy_function),
        fault_line_relative=fault_line_relative,
        fault_line_content=fault_content,
        fault_context=fault_ctx,
        error_message=bug.error_message,
        test_output=bug.test_output,
    )

    logger.info(
        "BaseRepair: generating %d patch(es) for %s",
        config.base_num_patches,
        bug.bug_id,
    )

    responses = llm.generate(prompt, num_return=config.base_num_patches)

    for i, response in enumerate(responses):
        token_stats["prompt_tokens"] += response.prompt_tokens
        token_stats["completion_tokens"] += response.completion_tokens
        logger.debug("BaseRepair response %d: %s", i, response.text[:200])

        patched_func = extract_function_from_response(response.text)
        if patched_func is None:
            logger.warning("BaseRepair: could not extract function from response %d", i)
            continue

        if not validate_syntax(patched_func):
            logger.warning("BaseRepair: patch %d has syntax errors", i)
            continue

        candidate = PatchCandidate(patch_code=patched_func, stage="BaseRepair")

        if bug.project_dir and bug.file_path.exists():
            patched_source = apply_patch(
                original_file=bug.file_path,
                buggy_function_name=bug.function_name,
                patched_function=patched_func,
                start_line=bug.start_line,
                end_line=bug.end_line,
            )
            result = validate_patch(
                project_dir=bug.project_dir,
                original_file=bug.file_path,
                patched_source=patched_source,
                test_file=bug.test_file,
                config=config,
            )
            candidate.validation = result
            if result.passed:
                logger.info("BaseRepair: patch %d PASSED all tests!", i)
                candidates.append(candidate)
                return candidates, token_stats
            else:
                logger.info(
                    "BaseRepair: patch %d failed (%d passed, %d failed, %d errors)",
                    i, result.num_passed, result.num_failed, result.num_errors,
                )

        elif validator is not None:
            result = validator(candidate)
            if result is not None:
                candidate.validation = result
                if result.passed:
                    logger.info("BaseRepair: patch %d PASSED (external validator)!", i)
                    candidates.append(candidate)
                    return candidates, token_stats

        candidates.append(candidate)

    return candidates, token_stats
