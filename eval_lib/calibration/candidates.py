# calibration/candidates.py
"""
Apply proposed config edits, diff configs, re-evaluate without the app, and
compare two runs with a significance test.

``apply_proposals`` and ``diff_configs`` are pure. ``evaluate_config`` is the
only async entry point and the only one that calls an LLM — it re-runs the judge
over saved test cases (never the application under test) and pairs the fresh
judge results with the existing human labels to produce an ``AgreementReport``.
``compare`` is a pure significance test over two runs of the same cases.
"""

import asyncio
import json
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

from pydantic import BaseModel, Field

from eval_lib.calibration.budget import Budget, BudgetExceeded, _BudgetedModel
from eval_lib.calibration.models import CustomEvalConfig, LabeledCase, Proposal
from eval_lib.calibration.stats import AgreementReport, _mcnemar_exact, agreement
from eval_lib.few_shot import FewShotExample

_INDEX_KINDS = {"rewrite_criterion", "remove_criterion", "split_criterion"}


# ── Result models ────────────────────────────────────────────────────────────
class ConfigDiff(BaseModel):
    schema_version: int = 1
    criteria_added: List[str] = Field(default_factory=list)
    criteria_removed: List[str] = Field(default_factory=list)
    criteria_changed: List[Dict[str, Any]] = Field(default_factory=list)
    params_changed: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    few_shot_added: int = 0
    few_shot_removed: int = 0


class CaseEvalResult(BaseModel):
    case_id: str
    human_verdict: Optional[str] = None
    judge_verdict: Optional[str] = None  # None == excluded (no verdict)
    score: Optional[float] = None
    success: Optional[bool] = None
    error: Optional[str] = None


class EvaluationRun(BaseModel):
    schema_version: int = 1
    results: List[CaseEvalResult] = Field(default_factory=list)
    agreement: AgreementReport
    llm_calls: int = 0
    cost_usd: float = 0.0
    stopped_by_budget: bool = False


class Comparison(BaseModel):
    schema_version: int = 1
    n_paired: int
    delta_accuracy: Optional[float] = None
    delta_kappa: Optional[float] = None
    delta_recall_fail: Optional[float] = None
    mcnemar_p: Optional[float] = None
    accuracy_delta_ci95: Optional[Tuple[float, float]] = None
    significant: bool = False
    fixed: List[str] = Field(default_factory=list)
    broken: List[str] = Field(default_factory=list)


# ── apply_proposals ──────────────────────────────────────────────────────────
def apply_proposals(
    config: CustomEvalConfig,
    proposals: List[Proposal],
    cases: Optional[List[LabeledCase]] = None,
) -> CustomEvalConfig:
    """Return a new config with the proposals applied; never mutates ``config``.

    Index-based criterion edits are applied back-to-front so earlier indices
    stay valid. Two edits targeting the same criterion index raise ``ValueError``.
    """
    criteria = list(config.evaluation_criteria)
    few_shot = [e.model_copy(deep=True) for e in config.few_shot_examples]
    temperature, strategy, threshold = config.temperature, config.strategy, config.threshold

    seen_index: Dict[int, str] = {}
    for p in proposals:
        if p.kind in _INDEX_KINDS:
            if p.criterion_index is None:
                raise ValueError(f"proposal {p.id} ({p.kind}) needs criterion_index")
            if p.criterion_index in seen_index:
                raise ValueError(
                    f"conflicting edits to criterion {p.criterion_index}: "
                    f"{seen_index[p.criterion_index]} and {p.id}"
                )
            seen_index[p.criterion_index] = p.id

    cases_by_id = {c.id: c for c in (cases or [])}
    indexed: List[Proposal] = []
    adds: List[str] = []
    for p in proposals:
        if p.kind in _INDEX_KINDS:
            indexed.append(p)
        elif p.kind == "add_criterion":
            if not p.text:
                raise ValueError(f"proposal {p.id} (add_criterion) needs text")
            adds.append(p.text)
        elif p.kind == "set_temperature":
            if p.temperature is None:
                raise ValueError(f"proposal {p.id} (set_temperature) needs temperature")
            temperature = float(p.temperature)
        elif p.kind == "set_strategy":
            if not p.strategy:
                raise ValueError(f"proposal {p.id} (set_strategy) needs strategy")
            strategy = p.strategy
        elif p.kind == "set_threshold":
            if p.threshold is None:
                raise ValueError(f"proposal {p.id} (set_threshold) needs threshold")
            threshold = float(p.threshold)
        elif p.kind == "add_few_shot":
            if cases is None:
                raise ValueError("add_few_shot requires cases=")
            for cid in p.case_ids or []:
                lc = cases_by_id.get(cid)
                if lc is None:
                    raise ValueError(f"add_few_shot references unknown case {cid!r}")
                few_shot.append(_example_from_case(lc))
        elif p.kind == "review_labels":
            continue  # informational; changes nothing

    for p in sorted(indexed, key=lambda q: cast(int, q.criterion_index), reverse=True):
        idx = cast(int, p.criterion_index)
        if idx < 0 or idx >= len(criteria):
            raise ValueError(f"criterion_index {idx} out of range (0..{len(criteria) - 1})")
        if p.kind == "rewrite_criterion":
            if not p.text:
                raise ValueError(f"proposal {p.id} (rewrite_criterion) needs text")
            criteria[idx] = p.text
        elif p.kind == "remove_criterion":
            del criteria[idx]
        elif p.kind == "split_criterion":
            if not p.texts:
                raise ValueError(f"proposal {p.id} (split_criterion) needs texts")
            criteria[idx : idx + 1] = list(p.texts)

    criteria.extend(adds)
    return cast(
        CustomEvalConfig,
        config.model_copy(
            update={
                "evaluation_criteria": criteria,
                "few_shot_examples": few_shot,
                "temperature": temperature,
                "strategy": strategy,
                "threshold": threshold,
            }
        ),
    )


def _example_from_case(lc: LabeledCase) -> FewShotExample:
    tc = lc.test_case
    data: Dict[str, str] = {}
    if tc.extra_fields:
        for k, v in tc.extra_fields.items():
            data[k] = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return FewShotExample(
        input=tc.input,
        actual_output=tc.actual_output,
        expected_output=tc.expected_output,
        data=data,
        verdict=lc.label.verdict,
        reason=lc.label.reason or "",
        source_case_id=lc.id,
    )


# ── diff_configs ─────────────────────────────────────────────────────────────
def diff_configs(a: CustomEvalConfig, b: CustomEvalConfig) -> ConfigDiff:
    ca, cb = a.evaluation_criteria, b.evaluation_criteria
    changed: List[Dict[str, Any]] = []
    changed_a, changed_b = set(), set()
    for i in range(min(len(ca), len(cb))):
        if ca[i] != cb[i] and ca[i] not in cb and cb[i] not in ca:
            changed.append({"index": i, "before": ca[i], "after": cb[i]})
            changed_a.add(i)
            changed_b.add(i)
    added = [cb[i] for i in range(len(cb)) if cb[i] not in ca and i not in changed_b]
    removed = [ca[i] for i in range(len(ca)) if ca[i] not in cb and i not in changed_a]

    params: Dict[str, Dict[str, Any]] = {}
    for field in ("name", "strategy", "n_runs", "aggregation", "temperature", "threshold"):
        va, vb = getattr(a, field), getattr(b, field)
        if va != vb:
            params[field] = {"before": va, "after": vb}

    return ConfigDiff(
        criteria_added=added,
        criteria_removed=removed,
        criteria_changed=changed,
        params_changed=params,
        few_shot_added=max(0, len(b.few_shot_examples) - len(a.few_shot_examples)),
        few_shot_removed=max(0, len(a.few_shot_examples) - len(b.few_shot_examples)),
    )


# ── evaluate_config ──────────────────────────────────────────────────────────
async def evaluate_config(
    config: CustomEvalConfig,
    cases: List[LabeledCase],
    model: Any,
    budget: Optional[Budget] = None,
    concurrency: int = 8,
    on_progress: Optional[Callable[[str, int, int], None]] = None,
) -> EvaluationRun:
    """Re-score every case's saved ``test_case`` with ``config``'s judge.

    Only the judge is called — never the app. Per-case errors are captured and
    do not abort the run. A supplied ``budget`` is shared and caps the work;
    without one an internal uncapped Budget is used purely to count calls/cost.
    """
    counting = budget if budget is not None else Budget()
    budgeted = _BudgetedModel(model, counting)
    metric = config.to_metric(budgeted)

    sem = asyncio.Semaphore(max(1, concurrency))
    total = len(cases)
    done = 0
    stopped = False

    async def worker(case: LabeledCase) -> CaseEvalResult:
        nonlocal done, stopped
        async with sem:
            tc = case.test_case
            if case.meta is not None:
                try:
                    tc._meta = case.meta
                except Exception:
                    pass
            out = CaseEvalResult(case_id=case.id, human_verdict=case.label.verdict)
            try:
                res = await metric.evaluate(tc)
                out.score = res.get("score")
                out.success = res.get("success")
                out.judge_verdict = _verdict_from_result(res)
            except BudgetExceeded as e:
                stopped = True
                out.error = f"budget: {e}"
            except Exception as e:  # noqa: BLE001 — a bad case must not kill the run
                out.error = f"{type(e).__name__}: {e}"
        done += 1
        if on_progress:
            on_progress("evaluate", done, total)
        return out

    results = await asyncio.gather(*[worker(c) for c in cases])

    # Pair fresh judge results with the existing human labels for agreement.
    by_id = {r.case_id: r for r in results}
    scored_cases = []
    for c in cases:
        r = by_id[c.id]
        jr = (
            {"error": r.error}
            if r.error
            else {
                "score": r.score,
                "success": r.success,
                "evaluation_log": {"kept_criteria": ["_"]},  # non-empty ⇒ counts as a verdict
            }
        )
        scored_cases.append(c.model_copy(update={"judge_result": jr}))
    report = agreement(scored_cases)

    return EvaluationRun(
        results=list(results),
        agreement=report,
        llm_calls=counting.llm_calls,
        cost_usd=round(counting.cost_usd, 6),
        stopped_by_budget=stopped,
    )


def _verdict_from_result(res: Dict[str, Any]) -> Optional[str]:
    log = res.get("evaluation_log") or {}
    if "kept_criteria" in log and len(log.get("kept_criteria") or []) == 0:
        return None
    success = res.get("success")
    if success is None:
        return None
    return "pass" if success else "fail"


# ── compare ──────────────────────────────────────────────────────────────────
def compare(before: EvaluationRun, after: EvaluationRun) -> Comparison:
    """Significance test over two runs of the *same* cases."""
    a = {r.case_id: r for r in before.results}
    b = {r.case_id: r for r in after.results}
    common = [cid for cid in a if cid in b]

    paired: List[Tuple[bool, bool]] = []
    fixed: List[str] = []
    broken: List[str] = []
    for cid in common:
        ra, rb = a[cid], b[cid]
        if ra.human_verdict is None or ra.judge_verdict is None or rb.judge_verdict is None:
            continue
        ca = ra.judge_verdict == ra.human_verdict
        cb = rb.judge_verdict == rb.human_verdict
        paired.append((ca, cb))
        if not ca and cb:
            fixed.append(cid)
        elif ca and not cb:
            broken.append(cid)

    n_paired = len(paired)
    b_count = len(broken)  # before correct, after wrong
    c_count = len(fixed)  # before wrong, after correct
    mcnemar_p = _mcnemar_exact(b_count, c_count) if n_paired else None

    def _delta(x: Optional[float], y: Optional[float]) -> Optional[float]:
        if x is None or y is None:
            return None
        return round(y - x, 4)

    return Comparison(
        n_paired=n_paired,
        delta_accuracy=_delta(before.agreement.accuracy, after.agreement.accuracy),
        delta_kappa=_delta(before.agreement.cohen_kappa, after.agreement.cohen_kappa),
        delta_recall_fail=_delta(before.agreement.recall_fail, after.agreement.recall_fail),
        mcnemar_p=mcnemar_p,
        accuracy_delta_ci95=_delta_ci(paired),
        significant=bool(mcnemar_p is not None and mcnemar_p < 0.05),
        fixed=fixed,
        broken=broken,
    )


def _delta_ci(paired: List[Tuple[bool, bool]]) -> Optional[Tuple[float, float]]:
    n = len(paired)
    if n < 2:
        return None
    rng = random.Random(12345)
    deltas: List[float] = []
    for _ in range(1000):
        pick = [paired[rng.randrange(n)] for _ in range(n)]
        before_acc = sum(1 for ca, _ in pick if ca) / n
        after_acc = sum(1 for _, cb in pick if cb) / n
        deltas.append(after_acc - before_acc)
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[min(len(deltas) - 1, int(0.975 * len(deltas)))]
    return (round(lo, 4), round(hi, 4))
