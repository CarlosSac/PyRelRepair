"""Run the repair pipeline on ConDefects bugs (quick sanity check).

Usage:
    python scripts/run_condefects.py [--n 10] [--base-only] [--debug]
                                     [--model qwen2.5-coder:3b deepseek-r1:7b ...]

Results saved to results/condefects_<stage>_<model>_<timestamp>/
  <bug_id>.json   — per-bug patch info and test results
  summary.json    — aggregate stats
"""
from __future__ import annotations

import argparse
import dataclasses
import json
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


def _run_for_model(model: str, raw_bugs, config: Config, args) -> dict:
    cfg = dataclasses.replace(config, ollama_model=model)
    has_tests = (args.data / "Test").is_dir()

    model_slug = model.replace(":", "-").replace("/", "-")
    stage = "baserepair" if args.base_only else "pipeline"
    out_dir = Path("results") / f"condefects_{stage}_{model_slug}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("--- Model: %s  →  %s/ ---", model, out_dir)

    results = []
    for raw_bug in raw_bugs:
        if len(results) >= args.n:
            break
        bug = condefects_to_buginfo(raw_bug)
        if bug is None:
            continue

        logger.info("[%d/%d] %s — %s()", len(results) + 1, args.n, bug.bug_id, bug.function_name)
        test_in = get_test_dir(raw_bug, args.data) if has_tests else None
        validator = make_validator(raw_bug, bug, test_in) if test_in else None

        if args.base_only:
            candidates, token_stats = base_repair(bug, cfg, validator=validator)
            best = next((c for c in candidates if c.is_valid), candidates[-1] if candidates else None)
            row = {
                "bug_id": bug.bug_id,
                "function_name": bug.function_name,
                "base_candidates": len(candidates),
                "sig_candidates": 0,
                "prompt_tokens": token_stats["prompt_tokens"],
                "completion_tokens": token_stats["completion_tokens"],
                "tests_passed": None, "tests_total": None,
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
            pr: PipelineResult = run_pipeline(bug, cfg, validator=validator)
            best = pr.best_candidate
            row = {
                "bug_id": bug.bug_id,
                "function_name": bug.function_name,
                "base_candidates": len(pr.base_candidates),
                "sig_candidates": len(pr.sig_candidates),
                "prompt_tokens": pr.total_prompt_tokens,
                "completion_tokens": pr.total_completion_tokens,
                "tests_passed": None, "tests_total": None,
                "all_tests_passed": False, "best_stage": None,
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
                logger.info("  base=%d  sig=%d  tests=%d/%d  stage=%s",
                            row["base_candidates"], row["sig_candidates"], passed, total, best.stage)
            else:
                logger.info("  base=%d  sig=%d", row["base_candidates"], row["sig_candidates"])

        (out_dir / f"{bug.bug_id}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        results.append(row)

    n_base = sum(1 for r in results if r["base_candidates"] > 0)
    n_sig = sum(1 for r in results if r["sig_candidates"] > 0)
    n_pass = sum(r["all_tests_passed"] for r in results)
    total_prompt = sum(r["prompt_tokens"] for r in results)
    total_completion = sum(r["completion_tokens"] for r in results)

    summary = {
        "model": model,
        "bugs_evaluated": len(results),
        "n_base": n_base,
        "n_sig": n_sig,
        "n_pass": n_pass,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "has_tests": has_tests,
        "bugs": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["out_dir"] = str(out_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repair pipeline on ConDefects")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--data", type=Path, default=Path("data/condefects"))
    parser.add_argument("--model", nargs="+", default=None, help="One or more Ollama models to compare")
    parser.add_argument("--base-only", action="store_true", help="Run BaseRepair only, skip SigRepair")
    parser.add_argument("--debug", action="store_true", help="Save full prompts and LLM responses to a log file")
    args = parser.parse_args()

    config = Config()
    models = args.model or [config.ollama_model]

    if args.debug:
        stage = "baserepair" if args.base_only else "pipeline"
        model_slug = "_".join(m.replace(":", "-").replace("/", "-") for m in models)
        log_path = Path("results") / f"condefects_{stage}_{model_slug}_{datetime.now():%Y%m%d_%H%M%S}_debug.log"
        _enable_debug(log_path)

    for model in list(models):
        cfg = dataclasses.replace(config, ollama_model=model)
        if not OllamaClient(cfg).is_available():
            logger.error("Model '%s' not found in Ollama — skipping.", model)
            models = [m for m in models if m != model]

    if not models:
        sys.exit(1)

    if not args.base_only and not OllamaClient(config).is_model_available(config.embed_model):
        logger.error("Embedding model '%s' not found. Run: ollama pull %s",
                     config.embed_model, config.embed_model)
        sys.exit(1)

    logger.info("Loading bugs from %s ...", args.data)
    raw_bugs = load_bugs(args.data, max_bugs=args.n * 3)

    summaries = [_run_for_model(m, raw_bugs, config, args) for m in models]

    label = "BaseRepair" if args.base_only else "Pipeline"
    if len(summaries) == 1:
        s = summaries[0]
        total = s["bugs_evaluated"]
        n_with = sum(1 for r in s["bugs"] if r["tests_total"] is not None)
        print(f"\n=== {label} Results (ConDefects) ===")
        print(f"Model             : {s['model']}")
        print(f"Bugs evaluated    : {total}")
        print(f"BaseRepair patches: {s['n_base']}/{total}")
        if not args.base_only:
            print(f"SigRepair patches : {s['n_sig']}/{total}")
        print(f"Tokens (prompt)   : {s['total_prompt_tokens']:,}")
        print(f"Tokens (output)   : {s['total_completion_tokens']:,}")
        if n_with:
            print(f"All tests pass    : {s['n_pass']}/{n_with}")
        print(f"Results saved     : {s['out_dir']}/")
    else:
        print(f"\n=== {label} Comparison (ConDefects) ===")
        col = 26
        print(f"{'Model':<{col}}  {'Base patches':<14}  {'Tests pass':<12}  {'Prompt tok':>12}  {'Output tok':>12}  Results")
        print("-" * 100)
        for s in summaries:
            total = s["bugs_evaluated"]
            n_with = sum(1 for r in s["bugs"] if r["tests_total"] is not None)
            tests = f"{s['n_pass']}/{n_with}" if n_with else "n/a"
            print(f"{s['model']:<{col}}  {s['n_base']}/{total:<12}  {tests:<12}  {s['total_prompt_tokens']:>12,}  {s['total_completion_tokens']:>12,}  {s['out_dir']}/")


if __name__ == "__main__":
    main()
