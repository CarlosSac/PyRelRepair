"""Run BaseRepair on a sample of ConDefects Python bugs.

Usage:
    python scripts/run_baserepair.py [--n 10] [--data data/condefects]

Validation modes (auto-detected):
  - tests_passed : patch passes all stdin/stdout test cases (requires Test/ dir)
  - syntax_valid : patched code compiles without SyntaxError (fallback)
"""
from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyrelrepair.config import Config
from pyrelrepair.condefects_loader import ConDefectsBug, load_bugs
from pyrelrepair.llm import OllamaClient
from pyrelrepair.prompts import SCRIPT_REPAIR_PROMPT
from pyrelrepair.validator import validate_syntax

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def extract_script(response: str) -> str | None:
    """Extract a Python script from an LLM response (```python ... ```)."""
    match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    stripped = response.strip()
    return stripped if stripped else None


def get_test_dir(bug: ConDefectsBug, condefects_dir: Path) -> Path | None:
    """Resolve the Test/in directory for a bug's task.

    ConDefects test layout: Test/{contest}/{LETTER}/in/
    """
    test_root = condefects_dir / "Test"
    if not test_root.is_dir():
        return None

    
    parts = bug.task_id.split("_")  
    if len(parts) != 2:
        return None

    # "abc221"
    contest = parts[0]              
    # "F" 
    letter = parts[1].upper()       

    test_in = test_root / contest / letter / "in"
    if test_in.is_dir():
        return test_in

    # Some tasks use "Ex" instead of the letter
    ex_in = test_root / contest / "Ex" / "in"
    if ex_in.is_dir():
        return ex_in

    return None


def run_on_tests(code: str, test_in_dir: Path, timeout: int = 10) -> tuple[int, int]:
    """Run patched code against all input files, compare to expected output.

    Returns (passed, total).
    """
    out_dir = test_in_dir.parent / "out"
    passed = 0
    total = 0

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
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
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                actual = result.stdout.rstrip("\n")
                expected = expected_file.read_text(
                    encoding="utf-8", errors="ignore"
                ).rstrip("\n")
                if actual == expected:
                    passed += 1
            except subprocess.TimeoutExpired:
                pass
            except Exception:
                pass
    finally:
        tmp_path.unlink(missing_ok=True)

    return passed, total


def main():
    parser = argparse.ArgumentParser(description="Run BaseRepair on ConDefects")
    parser.add_argument("--n", type=int, default=10, help="Number of bugs to evaluate")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/condefects"),
        help="Path to ConDefects repo root",
    )
    args = parser.parse_args()

    config = Config()
    client = OllamaClient(config)

    if not client.is_available():
        logger.error("Ollama is not running or model '%s' not found.", config.ollama_model)
        sys.exit(1)

    has_tests = (args.data / "Test").is_dir()
    if has_tests:
        logger.info("Test/ directory found — using stdin/stdout validation.")
    else:
        logger.warning("Test/ directory not found — syntax validation only.")

    logger.info("Loading bugs from %s ...", args.data)
    bugs = load_bugs(args.data, max_bugs=args.n)
    logger.info("Loaded %d bugs. Model: %s", len(bugs), config.ollama_model)

    results = []

    for i, bug in enumerate(bugs, 1):
        logger.info(
            "[%d/%d] %s / %s (fault line %d)",
            i, len(bugs), bug.task_id, bug.submission_id, bug.fault_line,
        )

        prompt = SCRIPT_REPAIR_PROMPT.format(
            task_id=bug.task_id,
            fault_line=bug.fault_line,
            buggy_code=bug.buggy_code,
        )

        responses = client.generate(prompt, num_return=1)
        response_text = responses[0].text if responses else ""
        patched = extract_script(response_text)

        syntax_ok = validate_syntax(patched) if patched else False

        tests_passed = None
        tests_total = None

        if has_tests and syntax_ok and patched:
            test_in = get_test_dir(bug, args.data)
            if test_in:
                tests_passed, tests_total = run_on_tests(patched, test_in)
                logger.info(
                    "  syntax_valid=%s  tests=%s/%s", syntax_ok, tests_passed, tests_total
                )
            else:
                logger.info("  syntax_valid=%s  tests=no test dir found", syntax_ok)
        else:
            logger.info("  syntax_valid=%s", syntax_ok)

        results.append({
            "task_id": bug.task_id,
            "submission_id": bug.submission_id,
            "syntax_valid": syntax_ok,
            "tests_passed": tests_passed,
            "tests_total": tests_total,
            "all_tests_passed": (
                tests_passed == tests_total and tests_total is not None and tests_total > 0
            ),
        })

    # Summary
    total = len(results)
    n_syntax = sum(r["syntax_valid"] for r in results)

    print("\n=== BaseRepair Results ===")
    print(f"Model          : {config.ollama_model}")
    print(f"Bugs evaluated : {total}")
    print(f"Syntax valid   : {n_syntax}/{total} ({100*n_syntax/total:.1f}%)")

    if has_tests:
        n_all_pass = sum(r["all_tests_passed"] for r in results)
        n_with_tests = sum(1 for r in results if r["tests_total"] is not None)
        print(f"Bugs with tests: {n_with_tests}/{total}")
        if n_with_tests:
            print(
                f"All tests pass : {n_all_pass}/{n_with_tests} "
                f"({100*n_all_pass/n_with_tests:.1f}%)"
            )
    else:
        print("(Download Test.zip from the ConDefects OneDrive link for full validation)")


if __name__ == "__main__":
    main()
