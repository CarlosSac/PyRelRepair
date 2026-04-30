"""Run the full repair pipeline (BaseRepair → SigRepair) on ConDefects bugs.

Usage:
    python scripts/run_pipeline.py [--n 10] [--data data/condefects]
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pyrelrepair.base_repair import BugInfo
from pyrelrepair.code_parser import extract_functions
from pyrelrepair.config import Config
from pyrelrepair.condefects_loader import ConDefectsBug, load_bugs
from pyrelrepair.llm import OllamaClient
from pyrelrepair.pipeline import PipelineResult, run_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _set_verbose() -> None:
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def condefects_to_buginfo(bug: ConDefectsBug) -> BugInfo | None:
    functions = extract_functions(bug.buggy_code, bug.buggy_file)
    containing = [f for f in functions if f.start_line <= bug.fault_line <= f.end_line]
    if not containing:
        logger.debug(
            "Skipping %s/%s: fault line %d not inside any function",
            bug.task_id, bug.submission_id, bug.fault_line,
        )
        return None
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


def patch_script(original: str, patch_code: str, start_line: int, end_line: int) -> str:
    lines = original.splitlines()
    return "\n".join(
        lines[: start_line - 1] + patch_code.splitlines() + lines[end_line:]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full pipeline on ConDefects")
    parser.add_argument("--n", type=int, default=10)
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
        logger.warning("Test/ directory not found — reporting candidate counts only.")

    logger.info("Loading bugs from %s ...", args.data)
    raw_bugs = load_bugs(args.data, max_bugs=args.n * 3)

    results = []
    for raw_bug in raw_bugs:
        if len(results) >= args.n:
            break

        bug = condefects_to_buginfo(raw_bug)
        if bug is None:
            continue

        logger.info("[%d/%d] %s — function: %s", len(results) + 1, args.n, bug.bug_id, bug.function_name)

        pipeline_result: PipelineResult = run_pipeline(bug, config)

        row = {
            "bug_id": bug.bug_id,
            "base_candidates": len(pipeline_result.base_candidates),
            "sig_candidates": len(pipeline_result.sig_candidates),
            "tests_passed": None,
            "tests_total": None,
            "all_tests_passed": False,
            "best_stage": None,
        }

        best = pipeline_result.best_candidate
        if best and has_tests:
            test_in = get_test_dir(raw_bug, args.data)
            if test_in:
                original = raw_bug.buggy_file.read_text(encoding="utf-8")
                patched = patch_script(original, best.patch_code, bug.start_line, bug.end_line)
                passed, total = run_on_tests(patched, test_in)
                row["tests_passed"] = passed
                row["tests_total"] = total
                row["all_tests_passed"] = passed == total and total > 0
                row["best_stage"] = best.stage
                logger.info("  base=%d  sig=%d  tests=%d/%d  stage=%s",
                    row["base_candidates"], row["sig_candidates"], passed, total, best.stage)
            else:
                logger.info("  base=%d  sig=%d  no test dir", row["base_candidates"], row["sig_candidates"])
        else:
            logger.info("  base=%d  sig=%d", row["base_candidates"], row["sig_candidates"])

        results.append(row)

    # Summary
    total = len(results)
    n_base = sum(1 for r in results if r["base_candidates"] > 0)
    n_sig = sum(1 for r in results if r["sig_candidates"] > 0)

    print("\n=== Pipeline Results ===")
    print(f"Model             : {config.ollama_model}")
    print(f"Bugs evaluated    : {total}")
    print(f"BaseRepair patches: {n_base}/{total}")
    print(f"SigRepair patches : {n_sig}/{total}")

    if has_tests:
        n_pass = sum(r["all_tests_passed"] for r in results)
        n_with_tests = sum(1 for r in results if r["tests_total"] is not None)
        by_stage = {}
        for r in results:
            if r["all_tests_passed"] and r["best_stage"]:
                by_stage[r["best_stage"]] = by_stage.get(r["best_stage"], 0) + 1
        print(f"Bugs with tests   : {n_with_tests}/{total}")
        if n_with_tests:
            print(f"All tests pass    : {n_pass}/{n_with_tests} ({100*n_pass/n_with_tests:.1f}%)")
        for stage, count in sorted(by_stage.items()):
            print(f"  solved by {stage}: {count}")


if __name__ == "__main__":
    main()
