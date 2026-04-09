"""Patch validation using pytest."""
from __future__ import annotations

import logging
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    passed: bool
    output: str
    return_code: int
    num_passed: int = 0
    num_failed: int = 0
    num_errors: int = 0


def extract_function_from_response(response: str) -> str | None:
    """Extract the Python function from an LLM response.

    Looks for a ```python ... ``` code block and returns its contents.
    Falls back to extracting anything that looks like a function definition.
    """
    # Try to extract from code block
    pattern = r"```python\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: try to find a def statement
    pattern = r"((?:@\w+.*\n)*def\s+\w+\s*\(.*$(?:\n(?:[ \t]+.*))*)"
    match = re.search(pattern, response, re.MULTILINE)
    if match:
        return match.group(1).strip()

    return None


def apply_patch(
    original_file: Path,
    buggy_function_name: str,
    patched_function: str,
    start_line: int,
    end_line: int,
) -> str:
    """Replace the buggy function in the source file with the patched version.

    Returns the full patched source code.
    """
    source = original_file.read_text(encoding="utf-8")
    lines = source.split("\n")

    # Replace lines [start_line-1 : end_line] with patched function
    # Preserve the original indentation level
    original_indent = ""
    if start_line > 0:
        orig_line = lines[start_line - 1]
        original_indent = orig_line[: len(orig_line) - len(orig_line.lstrip())]

    # Indent the patched function to match original
    patched_lines = patched_function.split("\n")
    indented_patched = []
    for i, line in enumerate(patched_lines):
        if i == 0:
            indented_patched.append(original_indent + line.lstrip())
        elif line.strip():
            indented_patched.append(original_indent + line)
        else:
            indented_patched.append("")

    lines[start_line - 1 : end_line] = indented_patched
    return "\n".join(lines)


def validate_patch(
    project_dir: Path,
    original_file: Path,
    patched_source: str,
    test_file: Path | None = None,
    config: Config | None = None,
) -> ValidationResult:
    """Validate a patch by running pytest on the project.

    Writes the patched source to the file, runs pytest, then restores
    the original source.
    """
    timeout = config.test_timeout if config else 300

    # Save original
    original_source = original_file.read_text(encoding="utf-8")

    try:
        # Write patched version
        original_file.write_text(patched_source, encoding="utf-8")

        # Build pytest command
        cmd = [sys.executable, "-m", "pytest", "-x", "--tb=short", "-q"]
        if test_file:
            cmd.append(str(test_file))

        result = subprocess.run(
            cmd,
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # Parse pytest output
        output = result.stdout + result.stderr
        passed = result.returncode == 0

        num_passed = 0
        num_failed = 0
        num_errors = 0

        # Parse summary line like "5 passed, 2 failed"
        summary_match = re.search(r"(\d+) passed", output)
        if summary_match:
            num_passed = int(summary_match.group(1))
        failed_match = re.search(r"(\d+) failed", output)
        if failed_match:
            num_failed = int(failed_match.group(1))
        error_match = re.search(r"(\d+) error", output)
        if error_match:
            num_errors = int(error_match.group(1))

        return ValidationResult(
            passed=passed,
            output=output,
            return_code=result.returncode,
            num_passed=num_passed,
            num_failed=num_failed,
            num_errors=num_errors,
        )

    except subprocess.TimeoutExpired:
        return ValidationResult(
            passed=False,
            output="Test execution timed out",
            return_code=-1,
        )
    except Exception as e:
        return ValidationResult(
            passed=False,
            output=f"Validation error: {e}",
            return_code=-1,
        )
    finally:
        # Always restore original source
        original_file.write_text(original_source, encoding="utf-8")


def validate_syntax(code: str) -> bool:
    """Quick check that a patch is syntactically valid Python."""
    try:
        compile(code, "<patch>", "exec")
        return True
    except SyntaxError:
        return False
