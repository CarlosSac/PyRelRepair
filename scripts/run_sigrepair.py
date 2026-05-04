"""Run SigRepair on a sample of ConDefects Python bugs.

Each ConDefects bug is a standalone script. We find the function that contains
the fault line and use all other functions in that same script as the signature
retrieval pool.

Usage:
    python scripts/run_sigrepair.py [--n 10] [--data data/condefects]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # repo root
sys.path.insert(0, str(Path(__file__).parent))          # scripts/

from pyrelrepair.config import Config
from pyrelrepair.condefects_loader import load_bugs
from pyrelrepair.llm import OllamaClient
from pyrelrepair.sig_repair import sig_repair
from condefects_utils import (
    condefects_to_buginfo,
    get_test_dir,
    make_validator,
    patch_script,
    run_on_tests,
    set_verbose,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SigRepair on ConDefects")
    parser.add_argument("--n", type=int, default=10, help="Number of bugs to evaluate")
    parser.add_argument("--data", type=Path, default=Path("data/condefects"))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        set_verbose()

    config = Config()
    client = OllamaClient(config)

    if not client.is_available():
        logger.error("Ollama is not running or model '%s' not found.", config.ollama_model)
        sys.exit(1)

    if not client.is_model_available(config.embed_model):
        logger.error(
            "Embedding model '%s' not found. Run: ollama pull %s",
            config.embed_model, config.embed_model,
        )
        sys.exit(1)

    has_tests = (args.data / "Test").is_dir()
    if not has_tests:
        logger.warning("Test/ directory not found — syntax validation only.")

    logger.info("Loading bugs from %s ...", args.data)
    raw_bugs = load_bugs(args.data, max_bugs=args.n * 3)

    results = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

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
        candidates, token_stats = sig_repair(bug, config, validator=validator)
        total_prompt_tokens += token_stats["prompt_tokens"]
        total_completion_tokens += token_stats["completion_tokens"]

        best = None
        if candidates:
            passing = [c for c in candidates if c.is_valid]
            best = passing[0] if passing else candidates[-1]

        tests_passed = tests_total = None
        all_pass = False

        if best and test_in:
            if best.validation is not None:
                tests_passed = best.validation.num_passed
                tests_total = best.validation.num_passed + best.validation.num_failed
            else:
                original = raw_bug.buggy_file.read_text(encoding="utf-8")
                patched_script = patch_script(
                    original, best.patch_code, bug.start_line, bug.end_line
                )
                tests_passed, tests_total = run_on_tests(patched_script, test_in)
            all_pass = tests_passed == tests_total and tests_total is not None and tests_total > 0
            logger.info(
                "  candidates=%d  tests=%s/%s", len(candidates), tests_passed, tests_total
            )
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
    print(f"Tokens (prompt)  : {total_prompt_tokens:,}")
    print(f"Tokens (output)  : {total_completion_tokens:,}")
    print(f"Tokens (total)   : {total_prompt_tokens + total_completion_tokens:,}")

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
