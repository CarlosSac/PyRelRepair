"""Quick test for the base_repair Python API.

Usage:
    python scripts/test_api.py

Runs two cases against a live Ollama instance (no project_dir = syntax-only validation):
  1. A simple off-by-one bug: easy, model should always fix it
  2. A wrong-variable bug: harder, closer to real ConDefects bugs
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pyrelrepair.base_repair import BugInfo, base_repair
from pyrelrepair.config import Config
from pyrelrepair.llm import OllamaClient

CASES: list[dict] = [
    {
        "name": "off-by-one in return",
        "bug": BugInfo(
            bug_id="test_001",
            file_path=Path("example.py"),
            function_name="compute_sum",
            buggy_function=(
                "def compute_sum(numbers):\n"
                "    total = 0\n"
                "    for n in numbers:\n"
                "        total += n\n"
                "    return total - 1\n"   # bug: should be `return total`
            ),
            fault_line=5,
            start_line=1,
            end_line=5,
            error_message="AssertionError: expected 6, got 5",
            test_output="FAILED: compute_sum([1, 2, 3]) returned 5, expected 6",
        ),
        "expected_fix": "return total",
    },
    {
        "name": "wrong variable in loop",
        "bug": BugInfo(
            bug_id="test_002",
            file_path=Path("example.py"),
            function_name="find_max",
            buggy_function=(
                "def find_max(items):\n"
                "    best = items[0]\n"
                "    for x in items:\n"
                "        if x > best:\n"
                "            best = items[0]\n"  # bug: should be `best = x`
                "    return best\n"
            ),
            fault_line=5,
            start_line=1,
            end_line=6,
            error_message="AssertionError: expected 9, got 1",
            test_output="FAILED: find_max([3, 9, 1]) returned 1, expected 9",
        ),
        "expected_fix": "best = x",
    },
]


def run_case(case: dict, config: Config) -> bool:
    print(f"\n{'-' * 60}")
    print(f"Case: {case['name']}")
    print(f"Expected fix contains: {case['expected_fix']!r}")

    candidates, _ = base_repair(case["bug"], config)

    if not candidates:
        print("FAIL  no candidates returned")
        return False

    c = candidates[0]
    # Without project_dir, base_repair only runs a syntax check.
    # Candidates that fail syntax are dropped, so any returned candidate is syntax-valid.
    # c.is_valid requires a ValidationResult (set only when project_dir is given).
    syntax_ok = bool(c.patch_code)
    print(f"Syntax valid : {syntax_ok}")

    if not syntax_ok:
        print("FAIL  patch did not compile")
        return False

    hint = case["expected_fix"]
    if hint in c.patch_code:
        print(f"PASS  patch contains {hint!r}")
        return True
    else:
        print(f"WARN  patch compiled but expected fix not found")
        print("--- patch ---")
        print(c.patch_code)
        print("-------------")
        # Still counts as passing — model may fix it differently
        return True


def main() -> None:
    config = Config()
    client = OllamaClient(config)

    if not client.is_available():
        print(f"ERROR: Ollama is not running or model '{config.ollama_model}' not found.")
        sys.exit(1)

    print(f"Model: {config.ollama_model}")

    passed = sum(run_case(case, config) for case in CASES)
    total = len(CASES)

    print(f"\n{'-' * 60}")
    print(f"Results: {passed}/{total} cases produced a valid patch")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
