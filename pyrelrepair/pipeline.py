"""Full repair pipeline: BaseRepair → SigRepair.

Runs stages in order and stops as soon as a passing patch is found
(only when pytest validation is available via bug.project_dir).
For ConDefects (no project_dir), both stages always run and
validation is handled externally by the run script.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .base_repair import BugInfo, PatchCandidate, ValidatorFn, base_repair
from .config import Config
from .sig_repair import sig_repair

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    bug_id: str
    base_candidates: list[PatchCandidate] = field(default_factory=list)
    sig_candidates: list[PatchCandidate] = field(default_factory=list)
    passing_stage: str | None = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    @property
    def passed(self) -> bool:
        return self.passing_stage is not None

    @property
    def all_candidates(self) -> list[PatchCandidate]:
        return self.base_candidates + self.sig_candidates

    @property
    def best_candidate(self) -> PatchCandidate | None:
        passing = [c for c in self.all_candidates if c.is_valid]
        if passing:
            return passing[0]
        return self.all_candidates[-1] if self.all_candidates else None


def run_pipeline(
    bug: BugInfo,
    config: Config,
    validator: ValidatorFn | None = None,
) -> PipelineResult:
    """Run BaseRepair then SigRepair, stopping early on a passing patch."""
    result = PipelineResult(bug_id=bug.bug_id)

    # Stage 1: BaseRepair
    logger.info("Pipeline [%s]: running BaseRepair", bug.bug_id)
    result.base_candidates, base_tokens = base_repair(bug, config, validator=validator)
    result.total_prompt_tokens += base_tokens["prompt_tokens"]
    result.total_completion_tokens += base_tokens["completion_tokens"]

    if any(c.is_valid for c in result.base_candidates):
        result.passing_stage = "BaseRepair"
        logger.info("Pipeline [%s]: solved at BaseRepair", bug.bug_id)
        return result

    # Stage 2: SigRepair
    logger.info("Pipeline [%s]: BaseRepair failed, escalating to SigRepair", bug.bug_id)
    result.sig_candidates, sig_tokens = sig_repair(bug, config, validator=validator)
    result.total_prompt_tokens += sig_tokens["prompt_tokens"]
    result.total_completion_tokens += sig_tokens["completion_tokens"]

    if any(c.is_valid for c in result.sig_candidates):
        result.passing_stage = "SigRepair"
        logger.info("Pipeline [%s]: solved at SigRepair", bug.bug_id)

    return result
