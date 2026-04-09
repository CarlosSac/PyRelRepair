"""BaseRepair: Stage 1 of the PyRelRepair pipeline.

Generates a single candidate patch using only the buggy function,
error messages, and failing test output — no retrieval augmentation.
This serves as both the first repair attempt and the no-retrieval baseline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .llm import OllamaClient
from .prompts import BASE_REPAIR_PROMPT
from .validator import (
    ValidationResult,
    apply_patch,
    extract_function_from_response,
    validate_patch,
    validate_syntax,
)

logger = logging.getLogger(__name__)


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


@dataclass
class PatchCandidate:
    """A generated patch candidate with its validation result."""
    patch_code: str
    stage: str
    validation: ValidationResult | None = None

    @property
    def is_valid(self) -> bool:
        return self.validation is not None and self.validation.passed


def base_repair(bug: BugInfo, config: Config) -> list[PatchCandidate]:
    """Execute the BaseRepair stage.

    Generates config.base_num_patches candidate patches (default: 1)
    using only the buggy function, error message, and test output.

    Returns:
        List of PatchCandidate objects (validated).
    """
    llm = OllamaClient(config)
    candidates = []

    prompt = BASE_REPAIR_PROMPT.format(
        file_path=bug.file_path,
        fault_line=bug.fault_line,
        buggy_function=bug.buggy_function,
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
        logger.debug("BaseRepair response %d: %s", i, response.text[:200])

        # Extract the function from LLM output
        patched_func = extract_function_from_response(response.text)
        if patched_func is None:
            logger.warning("BaseRepair: could not extract function from response %d", i)
            continue

        # Syntax check
        if not validate_syntax(patched_func):
            logger.warning("BaseRepair: patch %d has syntax errors", i)
            continue

        candidate = PatchCandidate(patch_code=patched_func, stage="BaseRepair")

        # Validate against test suite if project info is available
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
            else:
                logger.info(
                    "BaseRepair: patch %d failed (%d passed, %d failed, %d errors)",
                    i, result.num_passed, result.num_failed, result.num_errors,
                )

        candidates.append(candidate)

    return candidates
