# calibration/improve.py
"""
Small-set calibration: try a few candidate configs, keep the best, and run
several rounds — keeping the user's threshold fixed by default.

Built for the 10–30 labelled-row case. Each round scores the current config once
("before"), runs the analyst on that run's results (so every label is learned
from, as the current version sees it), then compares up to four candidates on the
rows they all share:

  A. Threshold only — swept on the "before" scores (only when ``keep_threshold`` is
     False; no LLM calls).
  B. Criteria edits (rewrite/add/remove/split, + strategy/temperature).
  C. B + few-shot examples (only if proposed); rows used as examples leave C's
     comparison set.
  D. Everything else the analyst proposed, if it differs from B and C.

The winner needs at least one more agreeing row than the round's "before" (ties →
fewer edits, A before B). Rounds repeat from the winner until every comparable row
agrees, a round gains nothing, ``max_rounds`` is reached, or the budget runs out.
``improve`` applies nothing — it returns a draft and the evidence.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from eval_lib.calibration.analyst import AnalysisResult, CalibrationAnalyst
from eval_lib.calibration.budget import Budget
from eval_lib.calibration.candidates import (
    CaseEvalResult,
    ConfigDiff,
    EvaluationRun,
    apply_proposals,
    diff_configs,
    evaluate_config,
)
from eval_lib.calibration.models import CalibrationSet, CustomEvalConfig, LabeledCase, Proposal
from eval_lib.calibration.stats import (
    AgreementReport,
    ThresholdPoint,
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


class RoundReport(BaseModel):
    schema_version: int = 1
    round: int
    candidates: List[CandidateReport] = Field(default_factory=list)
    winner: Optional[str] = None
    agree: int = 0  # winner's agreeing rows on the shared set (or before's if no winner)
    n: int = 0
    fixed: List[str] = Field(default_factory=list)
    broken: List[str] = Field(default_factory=list)


class ImproveResult(BaseModel):
    schema_version: int = 1
    before: EvaluationRun
    candidates: List[CandidateReport] = Field(default_factory=list)  # round 1 (back-compat)
    rounds: List[RoundReport] = Field(default_factory=list)
    winner: Optional[str] = None  # last adopted round's winner label
    verdict: str = "no_improvement"  # improved | no_improvement
    proposed_config: CustomEvalConfig
    analysis: Optional[AnalysisResult] = None  # round 1
    remaining: List[Dict[str, Any]] = Field(default_factory=list)  # {case_id, reason}
    cost_usd: float = 0.0
    llm_calls: int = 0
    stopped_by_budget: bool = False


class _Cand:
    """Internal working record for one candidate before it becomes a report."""

    def __init__(self, label, config, proposal_ids, n_edits, scores, result_by_id, threshold, tr):
        self.label = label
        self.config = config
        self.proposal_ids = proposal_ids
        self.n_edits = n_edits
        self.scores: Dict[str, Optional[float]] = scores
        self.result_by_id: Dict[str, CaseEvalResult] = result_by_id
        self.scored_ids: List[str] = list(result_by_id.keys())
        self.threshold = threshold
        self.tr: Optional[ThresholdReport] = tr
        self.excluded: Set[str] = set()


class _RoundOutcome:
    def __init__(self, report, winner_cand, winner_agree, before_agree, common, stopped):
        self.report = report
        self.winner_cand: Optional[_Cand] = winner_cand
        self.winner_agree = winner_agree
        self.before_agree = before_agree
        self.common = common
        self.stopped = stopped


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
    keep_threshold: bool = True,
    max_rounds: int = 3,
    rejudge: bool = True,
) -> ImproveResult:
    budget = budget if budget is not None else Budget()
    analyst = analyst or CalibrationAnalyst(model)
    base = calibration_set.config
    by_id = {c.id: c for c in calibration_set.cases}
    measure = [by_id[i] for i in measure_cases if i in by_id]
    measure_ids = [c.id for c in measure]
    calls0, cost0 = budget.llm_calls, budget.cost_usd

    current = base
    before0: Optional[EvaluationRun] = None
    analysis0: Optional[AnalysisResult] = None
    rounds: List[RoundReport] = []
    last_winner: Optional[_Cand] = None
    stopped = False

    for r in range(max(1, max_rounds)):
        before_run = await evaluate_config(
            current,
            measure,
            model,
            budget=budget,
            on_progress=_stage(on_progress, f"round{r + 1}_before"),
        )
        if r == 0:
            before0 = before_run
        stopped = stopped or before_run.stopped_by_budget

        human_by_id = {res.case_id: res.human_verdict for res in before_run.results}
        comparable = [
            res.case_id
            for res in before_run.results
            if res.human_verdict is not None and res.judge_verdict is not None
        ]
        disagreeing = [
            cid
            for cid in comparable
            if _result_verdict(before_run, cid, current.threshold) != human_by_id[cid]
        ]
        if comparable and not disagreeing:
            break  # every comparable row already agrees
        if stopped:
            break

        analyst_set, learn_ids = _analyst_inputs(
            calibration_set, current, before_run, by_id, measure_ids, learn_cases, rejudge
        )
        analysis = await analyst.analyze(
            analyst_set,
            cases=learn_ids,
            budget=budget,
            on_progress=on_progress,
            threshold_target=(current.threshold if keep_threshold else None),
        )
        if r == 0:
            analysis0 = analysis
        stopped = stopped or analysis.stopped_by_budget

        outcome = await _build_round(
            r + 1,
            current,
            before_run,
            analysis,
            measure,
            by_id,
            model,
            budget,
            keep_threshold,
            on_progress,
        )
        stopped = stopped or outcome.stopped
        rounds.append(outcome.report)

        if outcome.winner_cand is not None and outcome.winner_agree >= outcome.before_agree + 1:
            current = outcome.winner_cand.config
            last_winner = outcome.winner_cand
        else:
            break  # this round gained nothing → stop
        if stopped:
            break

    remaining = _remaining(last_winner, before0, base, measure_ids, by_id)
    verdict = "improved" if last_winner is not None else "no_improvement"
    # the last adopted round is the last round that named a winner
    winner_label = next((rr.winner for rr in reversed(rounds) if rr.winner is not None), None)

    return ImproveResult(
        before=before0,
        candidates=rounds[0].candidates if rounds else [],
        rounds=rounds,
        winner=winner_label,
        verdict=verdict,
        proposed_config=current,
        analysis=analysis0,
        remaining=remaining,
        cost_usd=round(budget.cost_usd - cost0, 6),
        llm_calls=budget.llm_calls - calls0,
        stopped_by_budget=stopped,
    )


# ── one round ──────────────────────────────────────────────────────────────
async def _build_round(
    round_idx,
    start,
    before_run,
    analysis,
    measure,
    by_id,
    model,
    budget,
    keep_threshold,
    on_progress,
) -> _RoundOutcome:
    before_scores = {r.case_id: r.score for r in before_run.results}
    before_results = {r.case_id: r for r in before_run.results}
    human_by_id = {r.case_id: r.human_verdict for r in before_run.results}
    measure_ids = [r.case_id for r in before_run.results if r.human_verdict is not None]

    props = [p for p in analysis.proposals if p.kind not in ("set_threshold", "review_labels")]
    criteria_props = _dedupe_index([p for p in props if p.kind in _CRITERIA_KINDS])
    fewshot_props = [p for p in props if p.kind == "add_few_shot"]

    cands: List[_Cand] = []
    stopped = False

    # A. Threshold only (only when the threshold may move).
    if not keep_threshold:
        tr_a = threshold_sweep_run(before_run, start.threshold)
        thr_a = tr_a.best.threshold if tr_a.best else start.threshold
        cands.append(
            _Cand(
                "A",
                start.model_copy(update={"threshold": thr_a}),
                [],
                0,
                before_scores,
                dict(before_results),
                thr_a,
                tr_a,
            )
        )

    # B. Criteria edits.
    if criteria_props and not stopped:
        cand_b, s = await _score_candidate(
            "B",
            apply_proposals(start, criteria_props, cases=_all(by_id)),
            criteria_props,
            measure,
            model,
            budget,
            by_id,
            human_by_id,
            keep_threshold,
            start.threshold,
            on_progress,
        )
        stopped = stopped or s
        if cand_b is not None:
            cands.append(cand_b)

    # C. B + few-shot (example rows leave C's comparison set).
    if fewshot_props and not stopped:
        combo = criteria_props + fewshot_props
        excluded = _fewshot_source_ids(fewshot_props)
        measure_c = [c for c in measure if c.id not in excluded]
        cand_c, s = await _score_candidate(
            "C",
            apply_proposals(start, combo, cases=_all(by_id)),
            combo,
            measure_c,
            model,
            budget,
            by_id,
            human_by_id,
            keep_threshold,
            start.threshold,
            on_progress,
        )
        stopped = stopped or s
        if cand_c is not None:
            cand_c.excluded = excluded
            cands.append(cand_c)

    # D. Everything else, if it differs from B and C.
    if props and not stopped:
        cfg_d0 = apply_proposals(start, props, cases=_all(by_id))
        if _cfg_key(cfg_d0) not in {_cfg_key(c.config) for c in cands}:
            excluded = _fewshot_source_ids(fewshot_props)
            measure_d = [c for c in measure if c.id not in excluded]
            cand_d, s = await _score_candidate(
                "D",
                cfg_d0,
                props,
                measure_d,
                model,
                budget,
                by_id,
                human_by_id,
                keep_threshold,
                start.threshold,
                on_progress,
            )
            stopped = stopped or s
            if cand_d is not None:
                cand_d.excluded = excluded
                cands.append(cand_d)

    # Common comparison set = measure rows no candidate dropped.
    dropped: Set[str] = set().union(*[c.excluded for c in cands]) if cands else set()
    common = [cid for cid in measure_ids if cid not in dropped]

    before_correct = {
        cid: (_judge(before_scores.get(cid), start.threshold) == human_by_id.get(cid))
        for cid in common
    }
    before_agree = sum(1 for cid in common if before_correct[cid])

    reports: List[CandidateReport] = []
    ranked: List[tuple] = []
    for c in cands:
        agree, fixed, broken = _score_on_common(c, common, human_by_id, before_correct)
        reports.append(
            CandidateReport(
                label=c.label,
                proposal_ids=c.proposal_ids,
                n_edits=c.n_edits,
                diff=diff_configs(start, c.config),
                agreement=_agreement_for(c, by_id),
                threshold=c.tr,
                agree=agree,
                n=len(common),
                fixed=fixed,
                broken=broken,
            )
        )
        ranked.append((-agree, c.n_edits, _ORDER.get(c.label, 9), c, fixed, broken))

    ranked.sort(key=lambda t: (t[0], t[1], t[2]))

    winner_cand = None
    winner_agree = before_agree
    winner_label = None
    w_fixed: List[str] = []
    w_broken: List[str] = []
    if ranked and common and -ranked[0][0] >= before_agree + 1:
        winner_cand = ranked[0][3]
        winner_agree = -ranked[0][0]
        winner_label = winner_cand.label
        w_fixed, w_broken = ranked[0][4], ranked[0][5]

    report = RoundReport(
        round=round_idx,
        candidates=reports,
        winner=winner_label,
        agree=winner_agree,
        n=len(common),
        fixed=w_fixed,
        broken=w_broken,
    )
    return _RoundOutcome(report, winner_cand, winner_agree, before_agree, common, stopped)


# ── helpers ──────────────────────────────────────────────────────────────────
async def _score_candidate(
    label,
    cfg0,
    proposals,
    measure,
    model,
    budget,
    by_id,
    human_by_id,
    keep_threshold,
    fixed_threshold,
    on_progress,
):
    run = await evaluate_config(
        cfg0, measure, model, budget=budget, on_progress=_stage(on_progress, f"score_{label}")
    )
    result_by_id = {r.case_id: r for r in run.results}
    scores = {r.case_id: r.score for r in run.results}
    if keep_threshold:
        thr = fixed_threshold
        tr = _info_threshold_report(scores, human_by_id, thr)
    else:
        tr = threshold_sweep_run(run, cfg0.threshold)
        thr = tr.best.threshold if tr.best else cfg0.threshold
    cfg = cfg0.model_copy(update={"threshold": thr})
    cand = _Cand(
        label, cfg, [p.id for p in proposals], len(proposals), scores, result_by_id, thr, tr
    )
    return cand, run.stopped_by_budget


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


def _info_threshold_report(scores_by_id, human_by_id, threshold) -> ThresholdReport:
    pairs = [
        (human_by_id[cid], s)
        for cid, s in scores_by_id.items()
        if s is not None and human_by_id.get(cid) is not None
    ]
    n = len(pairs)
    agree = sum(1 for h, s in pairs if ("fail" if s < threshold else "pass") == h)
    pt = ThresholdPoint(
        threshold=threshold, accuracy=round(agree / n, 4) if n else 0.0, agree=agree, n=n
    )
    return ThresholdReport(
        objective="kept",
        curve=[pt],
        best=pt,
        current=pt,
        rationale_text=f"Threshold kept at {threshold}: {agree} of {n} rows match your labels.",
    )


def _result_verdict(run: EvaluationRun, cid: str, threshold: float) -> Optional[str]:
    for r in run.results:
        if r.case_id == cid:
            return r.judge_verdict
    return None


def _analyst_inputs(calibration_set, current, before_run, by_id, measure_ids, learn_cases, rejudge):
    if rejudge:
        res_by_id = {r.case_id: r for r in before_run.results}
        rejudged: List[LabeledCase] = []
        for cid, r in res_by_id.items():
            lc = by_id.get(cid)
            if lc is None:
                continue
            if r.error:
                jr: Dict[str, Any] = {"error": r.error}
            else:
                log = r.evaluation_log or {
                    "kept_criteria": (["_"] if r.judge_verdict is not None else [])
                }
                jr = {
                    "score": r.score,
                    "success": r.success,
                    "reason": r.reason,
                    "evaluation_log": log,
                }
            rejudged.append(lc.model_copy(update={"judge_result": jr}))
        analyst_set = calibration_set.model_copy(update={"config": current, "cases": rejudged})
        return analyst_set, list(measure_ids)
    analyst_set = calibration_set.model_copy(update={"config": current})
    return analyst_set, [i for i in learn_cases if i in by_id]


def _remaining(last_winner, before0, base, measure_ids, by_id) -> List[Dict[str, Any]]:
    if last_winner is not None:
        result_by_id = last_winner.result_by_id
        threshold = last_winner.threshold
        ids = last_winner.scored_ids
    elif before0 is not None:
        result_by_id = {r.case_id: r for r in before0.results}
        threshold = base.threshold
        ids = measure_ids
    else:
        return []
    out: List[Dict[str, Any]] = []
    for cid in ids:
        r = result_by_id.get(cid)
        if r is None or r.human_verdict is None:
            continue
        j = _judge(r.score, threshold)
        if j is None or j == r.human_verdict:
            continue
        out.append({"case_id": cid, "reason": r.reason or ""})
    return out


def _all(by_id) -> List[LabeledCase]:
    return list(by_id.values())


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
