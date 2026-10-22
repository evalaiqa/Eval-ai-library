# calibration/improve.py
"""
Small-set calibration: try a few candidate configs and keep the best, tuning the
threshold on each candidate's own scores.

Built for the 10–30 labelled rows case, where a train/holdout split leaves too
few rows to decide anything. Instead of one bundle, ``improve`` scores the
current config once ("before"), then compares up to four candidates on the rows
they all share:

  A. Threshold only — swept on the "before" scores (no LLM calls).
  B. Criteria edits (rewrite/add/remove/split, + strategy/temperature) — scored
     once, then the threshold tuned on *these* scores.
  C. B + few-shot examples (only if proposed); the rows used as examples are
     left out of C's scoring.
  D. Everything else the analyst proposed, if it differs from B and C.

The winner is the candidate with the most agreeing rows (ties → fewer edits,
A before B). If nothing beats "before" by at least one row, the verdict is
``no_improvement`` and the config is unchanged. ``improve`` applies nothing.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from eval_lib.calibration.analyst import AnalysisResult, CalibrationAnalyst
from eval_lib.calibration.budget import Budget
from eval_lib.calibration.candidates import (
    ConfigDiff,
    EvaluationRun,
    apply_proposals,
    diff_configs,
    evaluate_config,
)
from eval_lib.calibration.models import CalibrationSet, CustomEvalConfig, LabeledCase, Proposal
from eval_lib.calibration.stats import (
    AgreementReport,
    ThresholdReport,
    agreement,
    threshold_sweep_run,
)

_CRITERIA_KINDS = {
    "rewrite_criterion",
    "add_criterion",
    "remove_criterion",
    "split_criterion",
    "set_strategy",
    "set_temperature",
}
_INDEX_KINDS = {"rewrite_criterion", "remove_criterion", "split_criterion"}
_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3}


class CandidateReport(BaseModel):
    schema_version: int = 1
    label: str  # A | B | C | D
    proposal_ids: List[str] = Field(default_factory=list)
    n_edits: int = 0
    diff: ConfigDiff
    agreement: AgreementReport
    threshold: Optional[ThresholdReport] = None
    agree: int = 0  # agreeing rows on the shared comparison set
    n: int = 0  # size of the shared comparison set
    fixed: List[str] = Field(default_factory=list)
    broken: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class ImproveResult(BaseModel):
    schema_version: int = 1
    before: EvaluationRun
    candidates: List[CandidateReport] = Field(default_factory=list)
    winner: Optional[str] = None
    verdict: str = "no_improvement"  # improved | no_improvement
    proposed_config: CustomEvalConfig
    analysis: Optional[AnalysisResult] = None
    cost_usd: float = 0.0
    llm_calls: int = 0
    stopped_by_budget: bool = False


class _Cand:
    """Internal working record for one candidate before it becomes a report."""

    def __init__(self, label, config, proposal_ids, n_edits, scores, scored_ids, threshold, tr):
        self.label = label
        self.config = config
        self.proposal_ids = proposal_ids
        self.n_edits = n_edits
        self.scores: Dict[str, float] = scores
        self.scored_ids: List[str] = scored_ids
        self.threshold = threshold
        self.tr: Optional[ThresholdReport] = tr
        self.excluded: Set[str] = set()


def _judge(score: Optional[float], threshold: float) -> Optional[str]:
    if score is None:
        return None
    return "fail" if score < threshold else "pass"


async def improve(
    calibration_set: CalibrationSet,
    learn_cases: List[str],
    measure_cases: List[str],
    model: Any,
    budget: Optional[Budget] = None,
    on_progress: Optional[Callable[[str, int, int], None]] = None,
    analyst: Optional[CalibrationAnalyst] = None,
) -> ImproveResult:
    budget = budget if budget is not None else Budget()
    analyst = analyst or CalibrationAnalyst(model)
    base = calibration_set.config
    by_id = {c.id: c for c in calibration_set.cases}
    measure = [by_id[i] for i in measure_cases if i in by_id]
    calls0, cost0 = budget.llm_calls, budget.cost_usd
    stopped = False

    # 1. Score the current config on measure_cases once.
    before = await evaluate_config(
        base, measure, model, budget=budget, on_progress=_stage(on_progress, "before")
    )
    stopped = stopped or before.stopped_by_budget
    before_scores = {r.case_id: r.score for r in before.results}
    human_by_id = {r.case_id: r.human_verdict for r in before.results}
    measure_ids = [r.case_id for r in before.results if r.human_verdict is not None]

    # 2. Analyst on learn_cases.
    learn = [i for i in learn_cases if i in by_id]
    analysis = await analyst.analyze(
        calibration_set, cases=learn, budget=budget, on_progress=on_progress
    )
    stopped = stopped or analysis.stopped_by_budget

    # 3. Partition proposals (threshold is handled per-candidate, never bundled).
    props = [p for p in analysis.proposals if p.kind not in ("set_threshold", "review_labels")]
    criteria_props = _dedupe_index([p for p in props if p.kind in _CRITERIA_KINDS])
    fewshot_props = [p for p in props if p.kind == "add_few_shot"]

    cands: List[_Cand] = []

    # A. Threshold only — swept on the "before" scores, no LLM.
    tr_a = threshold_sweep_run(before, base.threshold)
    thr_a = tr_a.best.threshold if tr_a.best else base.threshold
    cands.append(
        _Cand(
            "A",
            base.model_copy(update={"threshold": thr_a}),
            [],
            0,
            before_scores,
            list(measure_ids),
            thr_a,
            tr_a,
        )
    )

    # B. Criteria edits.
    if criteria_props and not stopped:
        try:
            cfg_b0 = apply_proposals(base, criteria_props, cases=list(calibration_set.cases))
            cand_b, stopped_b = await _score_candidate(
                "B", cfg_b0, criteria_props, measure, model, budget, by_id, on_progress
            )
            stopped = stopped or stopped_b
            cands.append(cand_b)
        except Exception:
            pass

    # C. B + few-shot (rows used as examples are left out of C's scoring).
    if fewshot_props and not stopped:
        try:
            combo = criteria_props + fewshot_props
            cfg_c0 = apply_proposals(base, combo, cases=list(calibration_set.cases))
            excluded = _fewshot_source_ids(fewshot_props)
            measure_c = [c for c in measure if c.id not in excluded]
            cand_c, stopped_c = await _score_candidate(
                "C", cfg_c0, combo, measure_c, model, budget, by_id, on_progress
            )
            cand_c.excluded = excluded
            stopped = stopped or stopped_c
            cands.append(cand_c)
        except Exception:
            pass

    # D. Everything else, if it differs from B and C.
    if props and not stopped:
        try:
            cfg_d0 = apply_proposals(base, props, cases=list(calibration_set.cases))
            existing = {_cfg_key(c.config) for c in cands}
            if _cfg_key(cfg_d0) not in existing:
                excluded = _fewshot_source_ids(fewshot_props)
                measure_d = [c for c in measure if c.id not in excluded]
                cand_d, stopped_d = await _score_candidate(
                    "D", cfg_d0, props, measure_d, model, budget, by_id, on_progress
                )
                cand_d.excluded = excluded
                stopped = stopped or stopped_d
                cands.append(cand_d)
        except Exception:
            pass

    # 4. Common comparison set = measure rows no candidate dropped.
    dropped: Set[str] = set().union(*[c.excluded for c in cands]) if cands else set()
    common = [cid for cid in measure_ids if cid not in dropped]

    # "before" baseline agreement on the common rows.
    before_correct = {
        cid: (_judge(before_scores.get(cid), base.threshold) == human_by_id.get(cid))
        for cid in common
    }
    before_agree = sum(1 for cid in common if before_correct[cid])

    # 5. Build reports + rank.
    reports: List[CandidateReport] = []
    ranked: List[tuple] = []
    for c in cands:
        agree, fixed, broken = _score_on_common(c, common, human_by_id, before_correct)
        reports.append(
            CandidateReport(
                label=c.label,
                proposal_ids=c.proposal_ids,
                n_edits=c.n_edits,
                diff=diff_configs(base, c.config),
                agreement=_agreement_for(c, by_id),
                threshold=c.tr,
                agree=agree,
                n=len(common),
                fixed=fixed,
                broken=broken,
            )
        )
        ranked.append((-agree, c.n_edits, _ORDER.get(c.label, 9), c))

    ranked.sort(key=lambda t: (t[0], t[1], t[2]))

    # 6. Decide.
    winner_label: Optional[str] = None
    verdict = "no_improvement"
    proposed = base
    if ranked and common:
        best_agree = -ranked[0][0]
        best_cand: _Cand = ranked[0][3]
        if best_agree >= before_agree + 1:
            winner_label = best_cand.label
            verdict = "improved"
            proposed = best_cand.config

    return ImproveResult(
        before=before,
        candidates=reports,
        winner=winner_label,
        verdict=verdict,
        proposed_config=proposed,
        analysis=analysis,
        cost_usd=round(budget.cost_usd - cost0, 6),
        llm_calls=budget.llm_calls - calls0,
        stopped_by_budget=stopped,
    )


# ── helpers ──────────────────────────────────────────────────────────────────
async def _score_candidate(label, cfg0, proposals, measure, model, budget, by_id, on_progress):
    run = await evaluate_config(
        cfg0, measure, model, budget=budget, on_progress=_stage(on_progress, f"score_{label}")
    )
    tr = threshold_sweep_run(run, cfg0.threshold)
    thr = tr.best.threshold if tr.best else cfg0.threshold
    scores = {r.case_id: r.score for r in run.results}
    scored_ids = [r.case_id for r in run.results]
    cfg = cfg0.model_copy(update={"threshold": thr})
    return (
        _Cand(label, cfg, [p.id for p in proposals], len(proposals), scores, scored_ids, thr, tr),
        run.stopped_by_budget,
    )


def _score_on_common(cand: _Cand, common, human_by_id, before_correct):
    agree = 0
    fixed: List[str] = []
    broken: List[str] = []
    for cid in common:
        j = _judge(cand.scores.get(cid), cand.threshold)
        ok = j is not None and j == human_by_id.get(cid)
        if ok:
            agree += 1
        if not before_correct.get(cid) and ok:
            fixed.append(cid)
        elif before_correct.get(cid) and not ok:
            broken.append(cid)
    return agree, fixed, broken


def _agreement_for(cand: _Cand, by_id) -> AgreementReport:
    cases = []
    for cid in cand.scored_ids:
        lc = by_id.get(cid)
        if lc is None:
            continue
        s = cand.scores.get(cid)
        jr = {
            "score": s,
            "success": (s is not None and s >= cand.threshold),
            "evaluation_log": {"kept_criteria": ["_"], "threshold": cand.threshold},
        }
        cases.append(lc.model_copy(update={"judge_result": jr}))
    return agreement(cases, min_n=1)


def _dedupe_index(proposals: List[Proposal]) -> List[Proposal]:
    """Keep only the first index-edit per criterion index so a bundle never
    conflicts in apply_proposals."""
    out: List[Proposal] = []
    used: Set[int] = set()
    for p in proposals:
        if p.kind in _INDEX_KINDS and p.criterion_index is not None:
            if p.criterion_index in used:
                continue
            used.add(p.criterion_index)
        out.append(p)
    return out


def _fewshot_source_ids(fewshot_props: List[Proposal]) -> Set[str]:
    ids: Set[str] = set()
    for p in fewshot_props:
        ids.update(p.case_ids or [])
    return ids


def _cfg_key(cfg: CustomEvalConfig) -> tuple:
    return (
        tuple(cfg.evaluation_criteria),
        cfg.strategy,
        cfg.temperature,
        len(cfg.few_shot_examples),
    )


def _stage(cb: Optional[Callable[[str, int, int], None]], stage: str):
    if cb is None:
        return None

    def wrapped(_name: str, done: int, total: int) -> None:
        cb(stage, done, total)

    return wrapped
