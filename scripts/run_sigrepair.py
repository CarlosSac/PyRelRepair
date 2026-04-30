"""Run SigRepair on a sample of ConDefects Python bugs.

Each ConDefects bug is a standalone script. We find the function that contains
the fault line and use all other functions in that same script as the signature
retrieval pool (option 1: treat the script as the project).

Usage:
    python scripts/run_sigrepair.py [--n 10] [--data data/condefects]
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pyrelrepair.base_repair import BugInfo, PatchCandidate
from pyrelrepair.code_parser import extract_functions
from pyrelrepair.config import Config
from pyrelrepair.condefects_loader import ConDefectsBug, load_bugs
from pyrelrepair.llm import OllamaClient
from pyrelrepair.sig_repair import sig_repair
from pyrelrepair.validator import ValidationResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _set_verbose() -> None:
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def condefects_to_buginfo(bug: ConDefectsBug) -> BugInfo | None:
    """Convert a ConDefectsBug to BugInfo by finding the function at the fault line."""
    functions = extract_functions(bug.buggy_code, bug.buggy_file)

    containing = [
        f for f in functions
        if f.start_line <= bug.fault_line <= f.end_line
    ]

    if not containing:
        logger.debug(
            "Skipping %s/%s: fault line %d not inside any function",
            bug.task_id, bug.submission_id, bug.fault_line,
        )
        return None

    # Innermost function if nested
    func = min(containing, key=lambda f: f.end_line - f.start_line)

    return BugInfo(
        bug_id=f"{bug.task_id}_{bug.submission_id}",
        file_path=bug.buggy_file,          # absolute path — resolved_file_path returns it as-is
        function_name=func.name,
        buggy_function=func.body,
        fault_line=bug.fault_line,
        start_line=func.start_line,
        end_line=func.end_line,
        error_message="Wrong Answer",
        test_output="",
        project_dir=None,                  # no pytest suite; skips in-stage validation
    )


def make_validator(raw_bug: ConDefectsBug, bug: BugInfo, test_in: Path):
    def validate(candidate: PatchCandidate) -> ValidationResult | None:
        original = raw_bug.buggy_file.read_text(encoding="utf-8")
        lines = original.splitlines()
        patched = "\n".join(
            lines[: bug.start_line - 1] + candidate.patch_code.splitlines() + lines[bug.end_line :]
        )
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


def get_test_dir(bug: ConDefectsBug, condefects_dir: Path) -> Path | None:
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
                actual = result.stdout.rstrip("\n")
                expected = expected_file.read_text(encoding="utf-8", errors="ignore").rstrip("\n")
                if actual == expected:
                    passed += 1
            except (subprocess.TimeoutExpired, Exception):
                pass
    finally:
        tmp_path.unlink(missing_ok=True)
    return passed, total


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SigRepair on ConDefects")
    parser.add_argument("--n", type=int, default=10, help="Number of bugs to evaluate")
    parser.add_argument("--data", type=Path, default=Path("data/condefects"))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        _set_verbose()

    config = Config()
    client = OllamaClient(config)
    if not client.is_available():
        logger.error("Ollama is not running or model '%s' not found.", config.ollama_model)
        sys.exit(1)

    has_tests = (args.data / "Test").is_dir()
    if not has_tests:
        logger.warning("Test/ directory not found — syntax validation only.")

    logger.info("Loading bugs from %s ...", args.data)
    raw_bugs = load_bugs(args.data, max_bugs=args.n * 3)  # load extra to account for skips

    results = []
    for raw_bug in raw_bugs:
        if len(results) >= args.n:
            break

        bug = condefects_to_buginfo(raw_bug)
        if bug is None:
            continue

        logger.info(
            "[%d/%d] %s (fault line %d, function %s)",
            len(results) + 1, args.n, bug.bug_id, bug.fault_line, bug.function_name,
        )

        test_in = get_test_dir(raw_bug, args.data) if has_tests else None
        validator = make_validator(raw_bug, bug, test_in) if test_in else None
        candidates = sig_repair(bug, config, validator=validator)

        best = None
        if candidates:
            passing = [c for c in candidates if c.is_valid]
            best = passing[0] if passing else candidates[-1]

        tests_passed = tests_total = None
        all_pass = False

        if best and test_in:
            # Reuse validation already done in-pipeline if available.
            if best.validation is not None:
                tests_passed = best.validation.num_passed
                tests_total = best.validation.num_passed + best.validation.num_failed
            else:
                original = raw_bug.buggy_file.read_text(encoding="utf-8")
                lines = original.splitlines()
                patched_script = "\n".join(
                    lines[: bug.start_line - 1] + best.patch_code.splitlines() + lines[bug.end_line :]
                )
                tests_passed, tests_total = run_on_tests(patched_script, test_in)
            all_pass = tests_passed == tests_total and tests_total is not None and tests_total > 0
            logger.info("  candidates=%d  tests=%s/%s", len(candidates), tests_passed, tests_total)
        else:
            logger.info("  candidates=%d", len(candidates))

        results.append({
            "bug_id": bug.bug_id,
            "function": bug.function_name,
            "num_candidates": len(candidates),
            "tests_passed": tests_passed,
            "tests_total": tests_total,
            "all_tests_passed": all_pass,
        })

    total = len(results)
    n_with_candidates = sum(1 for r in results if r["num_candidates"] > 0)

    print("\n=== SigRepair Results ===")
    print(f"Model            : {config.ollama_model}")
    print(f"Bugs evaluated   : {total}")
    print(f"Bugs with patches: {n_with_candidates}/{total}")

    if has_tests:
        n_all_pass = sum(r["all_tests_passed"] for r in results)
        n_with_tests = sum(1 for r in results if r["tests_total"] is not None)
        print(f"Bugs with tests  : {n_with_tests}/{total}")
        if n_with_tests:
            print(
                f"All tests pass   : {n_all_pass}/{n_with_tests} "
                f"({100 * n_all_pass / n_with_tests:.1f}%)"
            )
    else:
        print("(Download Test.zip for full validation)")


if __name__ == "__main__":
    main()
