# calibration/optimizer.py
"""
Calibration optimizer (EV-37).

Runs the analyst on train, turns its proposals into candidate configs, scores
them on train, picks the best by κ, then evaluates the winner and the current
version on holdout exactly once. It applies nothing — it returns a draft config
and the evidence; a human (or the platform) creates the version.

Overfitting guards: holdout never enters the analysis or candidate selection and
is scored only once; few-shot examples come only from train; an improvement that
sits inside the bootstrap CI is reported as ``not_significant``.
"""

import asyncio
from typing import Any, Callable, List, Optional

from pydantic import BaseModel, Field

from eval_lib.calibration.analyst import AnalysisResult, CalibrationAnalyst
from eval_lib.calibration.budget import Budget, InsufficientLabels
from eval_lib.calibration.candidates import (
    Comparison,
    ConfigDiff,
    apply_proposals,
    compare,
    diff_configs,
    evaluate_config,
)
from eval_lib.calibration.models import (
    CalibrationSet,
    CustomEvalConfig,
    LabeledCase,
    Proposal,
    Split,
)
from eval_lib.calibration.stats import _judge_verdict

_INDEX_KINDS = {"rewrite_criterion", "remove_criterion", "split_criterion"}


class CandidateResult(BaseModel):
    schema_version: int = 1
    proposal_ids: List[str] = Field(default_factory=list)
    n_edits: int = 0
    train_kappa: Optional[float] = None
    error: Optional[str] = None


class OptimizationResult(BaseModel):
    schema_version: int = 1
    proposed_config: CustomEvalConfig
    diff: ConfigDiff
    holdout_comparison: Optional[Comparison] = None
    analysis: AnalysisResult
    candidates: List[CandidateResult] = Field(default_factory=list)
    cost_usd: float = 0.0
    stopped_by_budget: bool = False
    verdict: str = "not_significant"  # improved | not_significant | worse


class CalibrationOptimizer:
    def __init__(
        self,
        model: Any,
        max_candidates: int = 6,
        min_labels: int = 40,
        min_per_class: int = 10,
        analyst: Optional[CalibrationAnalyst] = None,
    ):
        self.model = model
        self.max_candidates = max(1, max_candidates)
        self.min_labels = min_labels
        self.min_per_class = min_per_class
        self.analyst = analyst or CalibrationAnalyst(model)

    async def optimize(
        self,
        calibration_set: CalibrationSet,
        split: Split,
        budget: Optional[Budget] = None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> OptimizationResult:
        budget = budget if budget is not None else Budget()
        base = calibration_set.config
        by_id = {c.id: c for c in calibration_set.cases}
        train = [by_id[i] for i in split.train if i in by_id]
        holdout = [by_id[i] for i in split.holdout if i in by_id]

        self._check_labels(train)

        # 1. Analyst on train (holdout never passed in).
        analysis = await self.analyst.analyze(
            calibration_set, cases=split.train, budget=budget, on_progress=on_progress
        )
        stopped = analysis.stopped_by_budget

        # 2. Candidates from proposals.
        candidate_sets = self._build_candidates(analysis.proposals)

        # 3. Score each candidate on train.
        scored: List[CandidateResult] = []
        best: Optional[tuple] = None  # (kappa, -n_edits, config, result)
        for cand in candidate_sets:
            cr = CandidateResult(proposal_ids=[p.id for p in cand], n_edits=len(cand))
            try:
                cfg = apply_proposals(base, cand, cases=train)
                run = await evaluate_config(
                    cfg,
                    train,
                    self.model,
                    budget=budget,
                    concurrency=8,
                    on_progress=_stage(on_progress, "evaluate_train"),
                )
                stopped = stopped or run.stopped_by_budget
                cr.train_kappa = run.agreement.cohen_kappa
                key = (cr.train_kappa if cr.train_kappa is not None else -2.0, -cr.n_edits)
                if best is None or key > best[0]:
                    best = (key, cfg, cand)
            except Exception as e:  # a bad candidate must not kill the run
                cr.error = f"{type(e).__name__}: {e}"
            scored.append(cr)

        # 4. Pick winner; evaluate it + current once on holdout.
        comparison: Optional[Comparison] = None
        verdict = "not_significant"
        proposed = base
        if best is not None and best[1] is not base and holdout and not stopped:
            proposed = best[1]
            after_run = await evaluate_config(
                proposed,
                holdout,
                self.model,
                budget=budget,
                on_progress=_stage(on_progress, "evaluate_holdout"),
            )
            before_run = await evaluate_config(
                base,
                holdout,
                self.model,
                budget=budget,
                on_progress=_stage(on_progress, "evaluate_holdout"),
            )
            stopped = stopped or after_run.stopped_by_budget or before_run.stopped_by_budget
            comparison = compare(before_run, after_run)
            verdict = self._verdict(comparison)

        return OptimizationResult(
            proposed_config=proposed,
            diff=diff_configs(base, proposed),
            holdout_comparison=comparison,
            analysis=analysis,
            candidates=scored,
            cost_usd=round(budget.cost_usd, 6),
            stopped_by_budget=stopped,
            verdict=verdict,
        )

    # ── helpers ──────────────────────────────────────────────────────────────
    def _check_labels(self, train: List[LabeledCase]) -> None:
        labelled = [c for c in train if _judge_verdict(c) is not None]
        if len(labelled) < self.min_labels:
            raise InsufficientLabels(
                f"{len(labelled)} usable labels < min_labels={self.min_labels}"
            )
        n_fail = sum(1 for c in train if c.label.verdict == "fail")
        n_pass = sum(1 for c in train if c.label.verdict == "pass")
        if min(n_fail, n_pass) < self.min_per_class:
            raise InsufficientLabels(
                f"a class has fewer than min_per_class={self.min_per_class} "
                f"(pass={n_pass}, fail={n_fail})"
            )

    def _build_candidates(self, proposals: List[Proposal]) -> List[List[Proposal]]:
        if not proposals:
            return []
        candidates: List[List[Proposal]] = []
        # one combined candidate of all mutually non-conflicting proposals
        combined: List[Proposal] = []
        used_index = set()
        for p in proposals:
            if p.kind in _INDEX_KINDS:
                if p.criterion_index in used_index:
                    continue
                used_index.add(p.criterion_index)
            combined.append(p)
        if len(combined) > 1:
            candidates.append(combined)
        # each proposal on its own
        candidates.extend([p] for p in proposals)
        return candidates[: self.max_candidates]

    @staticmethod
    def _verdict(cmp: Optional[Comparison]) -> str:
        if cmp is None:
            return "not_significant"
        delta = cmp.delta_kappa if cmp.delta_kappa is not None else cmp.delta_accuracy
        if delta is None:
            return "not_significant"
        if cmp.significant and delta > 0:
            return "improved"
        if cmp.significant and delta < 0:
            return "worse"
        return "not_significant"


def _stage(cb: Optional[Callable[[str, int, int], None]], stage: str):
    if cb is None:
        return None

    def wrapped(_stage_name: str, done: int, total: int) -> None:
        cb(stage, done, total)

    return wrapped
