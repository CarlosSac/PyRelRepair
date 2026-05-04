"""Run the repair pipeline on ConDefects bugs (quick sanity check).

Usage:
    python scripts/run_condefects.py [--n 10] [--data data/condefects] [--base-only] [--debug]
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from pyrelrepair.base_repair import base_repair
from pyrelrepair.config import Config
from pyrelrepair.condefects_loader import load_bugs
from pyrelrepair.llm import OllamaClient
from pyrelrepair.pipeline import PipelineResult, run_pipeline
from condefects_utils import (
    condefects_to_buginfo,
    get_test_dir,
    make_validator,
    patch_script,
    run_on_tests,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _enable_debug(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(levelname)s %(name)s — %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    print(f"Debug log → {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repair pipeline on ConDefects")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--data", type=Path, default=Path("data/condefects"))
    parser.add_argument("--model", type=str, default=None, help="Ollama model to use (overrides config)")
    parser.add_argument("--base-only", action="store_true", help="Run BaseRepair only, skip SigRepair")
    parser.add_argument("--debug", action="store_true", help="Save full prompts and LLM responses to a log file")
    args = parser.parse_args()

    config = Config()
    if args.model:
        config.ollama_model = args.model

    model_slug = config.ollama_model.replace(":", "-").replace("/", "-")

    if args.debug:
        stage = "baserepair" if args.base_only else "pipeline"
        log_path = Path("results") / f"condefects_{stage}_{model_slug}_{datetime.now():%Y%m%d_%H%M%S}_debug.log"
        _enable_debug(log_path)

    client = OllamaClient(config)

    if not client.is_available():
        logger.error("Ollama is not running or model '%s' not found.", config.ollama_model)
        sys.exit(1)
    if not args.base_only and not client.is_model_available(config.embed_model):
        logger.error(
            "Embedding model '%s' not found. Run: ollama pull %s",
            config.embed_model, config.embed_model,
        )
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

        logger.info(
            "[%d/%d] %s — function: %s", len(results) + 1, args.n, bug.bug_id, bug.function_name
        )

        test_in = get_test_dir(raw_bug, args.data) if has_tests else None
        validator = make_validator(raw_bug, bug, test_in) if test_in else None

        if args.base_only:
            candidates, token_stats = base_repair(bug, config, validator=validator)
            best = next((c for c in candidates if c.is_valid), candidates[-1] if candidates else None)
            row = {
                "bug_id": bug.bug_id,
                "base_candidates": len(candidates),
                "sig_candidates": 0,
                "prompt_tokens": token_stats["prompt_tokens"],
                "completion_tokens": token_stats["completion_tokens"],
                "tests_passed": None,
                "tests_total": None,
                "all_tests_passed": False,
                "best_stage": "BaseRepair" if any(c.is_valid for c in candidates) else None,
            }
            if best and test_in:
                if best.validation is not None:
                    passed = best.validation.num_passed
                    total = best.validation.num_passed + best.validation.num_failed
                else:
                    original = raw_bug.buggy_file.read_text(encoding="utf-8")
                    patched = patch_script(original, best.patch_code, bug.start_line, bug.end_line)
                    passed, total = run_on_tests(patched, test_in)
                row["tests_passed"] = passed
                row["tests_total"] = total
                row["all_tests_passed"] = passed == total and total > 0
                logger.info("  base=%d  tests=%d/%d", len(candidates), passed, total)
            else:
                logger.info("  base=%d", len(candidates))
        else:
            pipeline_result: PipelineResult = run_pipeline(bug, config, validator=validator)
            best = pipeline_result.best_candidate
            row = {
                "bug_id": bug.bug_id,
                "base_candidates": len(pipeline_result.base_candidates),
                "sig_candidates": len(pipeline_result.sig_candidates),
                "prompt_tokens": pipeline_result.total_prompt_tokens,
                "completion_tokens": pipeline_result.total_completion_tokens,
                "tests_passed": None,
                "tests_total": None,
                "all_tests_passed": False,
                "best_stage": None,
            }
            if best and test_in:
                if best.validation is not None:
                    passed = best.validation.num_passed
                    total = best.validation.num_passed + best.validation.num_failed
                else:
                    original = raw_bug.buggy_file.read_text(encoding="utf-8")
                    patched = patch_script(original, best.patch_code, bug.start_line, bug.end_line)
                    passed, total = run_on_tests(patched, test_in)
                row["tests_passed"] = passed
                row["tests_total"] = total
                row["all_tests_passed"] = passed == total and total > 0
                row["best_stage"] = best.stage
                logger.info(
                    "  base=%d  sig=%d  tests=%d/%d  stage=%s",
                    row["base_candidates"], row["sig_candidates"], passed, total, best.stage,
                )
            else:
                logger.info("  base=%d  sig=%d", row["base_candidates"], row["sig_candidates"])

        results.append(row)

    total = len(results)
    n_base = sum(1 for r in results if r["base_candidates"] > 0)
    n_sig = sum(1 for r in results if r["sig_candidates"] > 0)
    total_prompt = sum(r["prompt_tokens"] for r in results)
    total_completion = sum(r["completion_tokens"] for r in results)

    label = "BaseRepair" if args.base_only else "Pipeline"
    print(f"\n=== {label} Results (ConDefects) ===")
    print(f"Model             : {config.ollama_model}")
    print(f"Bugs evaluated    : {total}")
    print(f"BaseRepair patches: {n_base}/{total}")
    if not args.base_only:
        print(f"SigRepair patches : {n_sig}/{total}")
    print(f"Tokens (prompt)   : {total_prompt:,}")
    print(f"Tokens (output)   : {total_completion:,}")
    print(f"Tokens (total)    : {total_prompt + total_completion:,}")

    if has_tests:
        n_pass = sum(r["all_tests_passed"] for r in results)
        n_with_tests = sum(1 for r in results if r["tests_total"] is not None)
        by_stage: dict[str, int] = {}
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
