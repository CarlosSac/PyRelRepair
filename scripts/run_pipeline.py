"""Run the full repair pipeline (BaseRepair → SigRepair) on BugsInPy bugs.

Usage:
    python scripts/run_pipeline.py [--n 6] [--model qwen2.5-coder:3b deepseek-r1:7b ...]

Results saved to results/pipeline_<model>_<timestamp>/
  <bug_id>.json   — per-bug patch candidates + validation output
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

from pyrelrepair.base_repair import BugInfo, PatchCandidate
from pyrelrepair.bugsinpy_loader import load_bugs
from pyrelrepair.config import Config
from pyrelrepair.llm import OllamaClient
from pyrelrepair.pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _candidate_to_dict(c: PatchCandidate) -> dict:
    v = c.validation
    return {
        "stage": c.stage,
        "patch_code": c.patch_code,
        "passed": v.passed if v else None,
        "tests_passed": v.num_passed if v else None,
        "tests_failed": v.num_failed if v else None,
        "num_errors": v.num_errors if v else None,
        "return_code": v.return_code if v else None,
        "validation_output": v.output if v else None,
    }


def _run_for_model(model: str, bugs: list[BugInfo], config: Config) -> dict:
    cfg = dataclasses.replace(config, ollama_model=model)
    model_slug = model.replace(":", "-").replace("/", "-")
    out_dir = Path("results") / f"pipeline_{model_slug}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("--- Model: %s  →  %s/ ---", model, out_dir)

    results = []
    for i, bug in enumerate(bugs, 1):
        logger.info("[%d/%d] %s — %s()", i, len(bugs), bug.bug_id, bug.function_name)
        pr = run_pipeline(bug, cfg)

        repaired = any(c.is_valid for c in pr.all_candidates)
        best = pr.best_candidate
        val = best.validation if best else None

        (out_dir / f"{bug.bug_id}.json").write_text(json.dumps({
            "bug_id": bug.bug_id,
            "function_name": bug.function_name,
            "file_path": str(bug.file_path),
            "fault_line": bug.fault_line,
            "repaired": repaired,
            "passing_stage": pr.passing_stage,
            "prompt_tokens": pr.total_prompt_tokens,
            "completion_tokens": pr.total_completion_tokens,
            "base_candidates": [_candidate_to_dict(c) for c in pr.base_candidates],
            "sig_candidates": [_candidate_to_dict(c) for c in pr.sig_candidates],
        }, indent=2), encoding="utf-8")

        results.append({
            "bug_id": bug.bug_id,
            "base_candidates": len(pr.base_candidates),
            "sig_candidates": len(pr.sig_candidates),
            "passing_stage": pr.passing_stage,
            "tests_passed": val.num_passed if val else None,
            "tests_failed": val.num_failed if val else None,
            "repaired": repaired,
            "prompt_tokens": pr.total_prompt_tokens,
            "completion_tokens": pr.total_completion_tokens,
        })
        logger.info(
            "  base=%d  sig=%d  repaired=%s  stage=%s",
            len(pr.base_candidates), len(pr.sig_candidates), repaired, pr.passing_stage,
        )

    n_repaired = sum(r["repaired"] for r in results)
    total_prompt = sum(r["prompt_tokens"] for r in results)
    total_completion = sum(r["completion_tokens"] for r in results)
    total = len(results)
    by_stage: dict[str, int] = {}
    for r in results:
        if r["passing_stage"]:
            by_stage[r["passing_stage"]] = by_stage.get(r["passing_stage"], 0) + 1

    summary = {
        "model": model,
        "bugs_evaluated": total,
        "repaired": n_repaired,
        "repair_rate": round(n_repaired / total, 4) if total else 0,
        "by_stage": by_stage,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "bugs": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["out_dir"] = str(out_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full pipeline on BugsInPy")
    parser.add_argument("--n", type=int, default=6, help="Number of bugs to evaluate")
    parser.add_argument("--data", type=Path, default=Path("data/bugsinpy_checked"))
    parser.add_argument("--model", nargs="+", default=None, help="One or more Ollama models to compare")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

    config = Config()
    models = args.model or [config.ollama_model]

    if not OllamaClient(config).is_model_available(config.embed_model):
        logger.error("Embedding model '%s' not found. Run: ollama pull %s",
                     config.embed_model, config.embed_model)
        sys.exit(1)

    for model in list(models):
        if not OllamaClient(dataclasses.replace(config, ollama_model=model)).is_available():
            logger.error("Model '%s' not found in Ollama — skipping.", model)
            models = [m for m in models if m != model]

    if not models:
        sys.exit(1)

    bugs = load_bugs(args.data, max_bugs=args.n)
    if not bugs:
        logger.error("No bugs found in %s — run bugsinpy_setup.py first.", args.data)
        sys.exit(1)

    logger.info("Loaded %d bugs. Running %d model(s).", len(bugs), len(models))

    summaries = [_run_for_model(m, bugs, config) for m in models]

    if len(summaries) == 1:
        s = summaries[0]
        total = s["bugs_evaluated"]
        print("\n=== Pipeline Results (BugsInPy) ===")
        print(f"Model          : {s['model']}")
        print(f"Bugs evaluated : {total}")
        print(f"Repaired       : {s['repaired']}/{total} ({100*s['repair_rate']:.1f}%)")
        for stage, count in sorted(s["by_stage"].items()):
            print(f"  solved by {stage}: {count}")
        print(f"Tokens (prompt): {s['total_prompt_tokens']:,}")
        print(f"Tokens (output): {s['total_completion_tokens']:,}")
        print(f"Results saved  : {s['out_dir']}/")
    else:
        print("\n=== Pipeline Comparison (BugsInPy) ===")
        col = 26
        print(f"{'Model':<{col}}  {'Repaired':<12}  {'Prompt tok':>12}  {'Output tok':>12}  Results")
        print("-" * 90)
        for s in summaries:
            total = s["bugs_evaluated"]
            repaired = f"{s['repaired']}/{total} ({100*s['repair_rate']:.0f}%)"
            print(f"{s['model']:<{col}}  {repaired:<12}  {s['total_prompt_tokens']:>12,}  {s['total_completion_tokens']:>12,}  {s['out_dir']}/")


if __name__ == "__main__":
    main()
