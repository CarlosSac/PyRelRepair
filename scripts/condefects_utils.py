"""Shared utilities for ConDefects run scripts."""
from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from pathlib import Path

from pyrelrepair.base_repair import BugInfo, PatchCandidate
from pyrelrepair.code_parser import extract_functions
from pyrelrepair.condefects_loader import ConDefectsBug
from pyrelrepair.validator import ValidationResult

logger = logging.getLogger(__name__)


def set_verbose() -> None:
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def condefects_to_buginfo(bug: ConDefectsBug) -> BugInfo | None:
    """Convert a ConDefectsBug to BugInfo.

    Prefers the innermost function containing the fault line.
    Falls back to the full script when the fault is at module level.
    """
    functions = extract_functions(bug.buggy_code, bug.buggy_file)
    containing = [f for f in functions if f.start_line <= bug.fault_line <= f.end_line]

    if containing:
        func = min(containing, key=lambda f: f.end_line - f.start_line)
        return BugInfo(
            bug_id=f"{bug.task_id}_{bug.submission_id}",
            file_path=bug.buggy_file,
            function_name=func.name,
            buggy_function=func.body,
            fault_line=bug.fault_line,
            start_line=func.start_line,
            end_line=func.end_line,
            error_message="Wrong Answer",
            test_output="",
            project_dir=None,
        )

    # Module-level code: use the entire script as the "function"
    lines = bug.buggy_code.splitlines()
    end_line = len(lines)
    if not lines or not (1 <= bug.fault_line <= end_line):
        logger.debug(
            "Skipping %s/%s: fault line %d out of range (script has %d lines)",
            bug.task_id, bug.submission_id, bug.fault_line, end_line,
        )
        return None

    logger.debug(
        "Module-level bug %s/%s: using full script (fault line %d)",
        bug.task_id, bug.submission_id, bug.fault_line,
    )
    return BugInfo(
        bug_id=f"{bug.task_id}_{bug.submission_id}",
        file_path=bug.buggy_file,
        function_name="<module>",
        buggy_function=bug.buggy_code,
        fault_line=bug.fault_line,
        start_line=1,
        end_line=end_line,
        error_message="Wrong Answer",
        test_output="",
        project_dir=None,
    )


def get_test_dir(bug: ConDefectsBug, condefects_dir: Path) -> Path | None:
    """Resolve the Test/in directory for a bug's task."""
    parts = bug.task_id.split("_")
    if len(parts) != 2:
        return None
    contest, letter = parts[0], parts[1].upper()
    test_in = condefects_dir / "Test" / contest / letter / "in"
    if test_in.is_dir():
        return test_in
    ex_in = condefects_dir / "Test" / contest / "Ex" / "in"
    return ex_in if ex_in.is_dir() else None


def run_on_tests(code: str, test_in_dir: Path, timeout: int = 10) -> tuple[int, int]:
    """Run code against all input files and compare to expected output.

    Returns (passed, total).
    """
    out_dir = test_in_dir.parent / "out"
    passed = total = 0
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", encoding="utf-8", delete=False) as tmp:
        tmp.write(code)
        tmp_path = Path(tmp.name)
    try:
        for in_file in sorted(test_in_dir.iterdir()):
            expected_file = out_dir / in_file.name
            if not expected_file.exists():
                continue
            total += 1
            try:
                result = subprocess.run(
                    [sys.executable, str(tmp_path)],
                    input=in_file.read_text(encoding="utf-8", errors="ignore"),
                    capture_output=True, text=True, timeout=timeout,
                )
                if result.stdout.rstrip("\n") == expected_file.read_text(
                    encoding="utf-8", errors="ignore"
                ).rstrip("\n"):
                    passed += 1
            except (subprocess.TimeoutExpired, Exception):
                pass
    finally:
        tmp_path.unlink(missing_ok=True)
    return passed, total


def patch_script(original: str, patch_code: str, start_line: int, end_line: int) -> str:
    """Splice patch_code into original, replacing lines start_line..end_line (1-based)."""
    lines = original.splitlines()
    return "\n".join(lines[: start_line - 1] + patch_code.splitlines() + lines[end_line:])


def make_validator(raw_bug: ConDefectsBug, bug: BugInfo, test_in: Path):
    """Return a ValidatorFn that runs stdin/stdout tests for a ConDefects bug."""
    def validate(candidate: PatchCandidate) -> ValidationResult | None:
        original = raw_bug.buggy_file.read_text(encoding="utf-8")
        patched = patch_script(original, candidate.patch_code, bug.start_line, bug.end_line)
        passed, total = run_on_tests(patched, test_in)
        return ValidationResult(
            passed=passed == total and total > 0,
            output=f"{passed}/{total} tests passed",
            return_code=0 if passed == total else 1,
            num_passed=passed,
            num_failed=total - passed,
            num_errors=0,
        )
    return validate
