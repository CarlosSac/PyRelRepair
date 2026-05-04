"""Run BaseRepair baseline on BugsInPy bugs.

Usage:
    python scripts/run_baserepair.py [--n 6] [--data data/bugsinpy_checked]

Results saved to results/baserepair_<timestamp>/
  <bug_id>.json   — per-bug patch candidates + validation output
  summary.json    — aggregate stats
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pyrelrepair.base_repair import PatchCandidate, base_repair
from pyrelrepair.bugsinpy_loader import load_bugs
from pyrelrepair.config import Config
from pyrelrepair.llm import OllamaClient

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BaseRepair on BugsInPy")
    parser.add_argument("--n", type=int, default=6, help="Number of bugs to evaluate")
    parser.add_argument("--data", type=Path, default=Path("data/bugsinpy_checked"))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

    config = Config()
    client = OllamaClient(config)

    if not client.is_available():
        logger.error("Ollama is not running or model '%s' not found.", config.ollama_model)
        sys.exit(1)

    bugs = load_bugs(args.data, max_bugs=args.n)
    if not bugs:
        logger.error("No bugs found in %s — run bugsinpy_setup.py first.", args.data)
        sys.exit(1)

    out_dir = Path("results") / f"baserepair_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving results to %s/", out_dir)
    logger.info("Loaded %d bugs. Model: %s", len(bugs), config.ollama_model)

    results = []
    for i, bug in enumerate(bugs, 1):
        logger.info("[%d/%d] %s — %s()", i, len(bugs), bug.bug_id, bug.function_name)
        candidates, token_stats = base_repair(bug, config)

        repaired = any(c.is_valid for c in candidates)
        best = next((c for c in candidates if c.is_valid), candidates[-1] if candidates else None)
        val = best.validation if best else None

        bug_result = {
            "bug_id": bug.bug_id,
            "function_name": bug.function_name,
            "file_path": str(bug.file_path),
            "fault_line": bug.fault_line,
            "repaired": repaired,
            "prompt_tokens": token_stats["prompt_tokens"],
            "completion_tokens": token_stats["completion_tokens"],
            "candidates": [_candidate_to_dict(c) for c in candidates],
        }
        (out_dir / f"{bug.bug_id}.json").write_text(
            json.dumps(bug_result, indent=2), encoding="utf-8"
        )

        row = {
            "bug_id": bug.bug_id,
            "num_candidates": len(candidates),
            "repaired": repaired,
            "tests_passed": val.num_passed if val else None,
            "tests_failed": val.num_failed if val else None,
            "prompt_tokens": token_stats["prompt_tokens"],
            "completion_tokens": token_stats["completion_tokens"],
        }
        results.append(row)
        logger.info(
            "  candidates=%d  repaired=%s  tests=%s/%s",
            len(candidates), repaired,
            val.num_passed if val else "-",
            (val.num_passed + val.num_failed) if val else "-",
        )

    total = len(results)
    n_repaired = sum(r["repaired"] for r in results)
    total_prompt = sum(r["prompt_tokens"] for r in results)
    total_completion = sum(r["completion_tokens"] for r in results)

    summary = {
        "model": config.ollama_model,
        "bugs_evaluated": total,
        "repaired": n_repaired,
        "repair_rate": round(n_repaired / total, 4) if total else 0,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "bugs": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== BaseRepair Results (BugsInPy) ===")
    print(f"Model          : {config.ollama_model}")
    print(f"Bugs evaluated : {total}")
    print(f"Repaired       : {n_repaired}/{total} ({100*n_repaired/total:.1f}%)" if total else "Repaired: 0")
    print(f"Tokens (prompt): {total_prompt:,}")
    print(f"Tokens (output): {total_completion:,}")
    print(f"Tokens (total) : {total_prompt + total_completion:,}")
    print(f"Results saved  : {out_dir}/")


if __name__ == "__main__":
    main()
