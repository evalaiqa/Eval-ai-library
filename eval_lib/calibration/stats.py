# calibration/stats.py
"""
Agreement statistics and threshold tuning for Custom Eval calibration.

Pure functions — no LLM calls, no scipy/numpy. The positive class is **fail**
(the team cares most about not letting bad answers through). Cases where the
judge produced no verdict (an error, or every criterion skipped) are left out of
the agreement maths and counted in ``excluded``.

All math helpers (Wilson interval, Cohen's/Fleiss' κ, Spearman, MAE, percentile
bootstrap, exact McNemar) are implemented here from the standard library so the
package pulls in no heavy numeric dependency.
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple, cast

from pydantic import BaseModel, Field

from eval_lib.metrics.custom_metric.custom_eval import VERDICT_WEIGHTS
from eval_lib.utils import score_agg

_Z95 = 1.959963984540054
_BOOTSTRAP_N = 1000
_BOOTSTRAP_SEED = 12345
_NEGATIVE_VERDICTS = {"minor", "none"}


# ── Report models ────────────────────────────────────────────────────────────
class ConfusionCell(BaseModel):
    count: int = 0
    case_ids: List[str] = Field(default_factory=list)


class Confusion(BaseModel):
    true_fail: ConfusionCell = Field(default_factory=ConfusionCell)
    false_pass: ConfusionCell = Field(default_factory=ConfusionCell)
    false_fail: ConfusionCell = Field(default_factory=ConfusionCell)
    true_pass: ConfusionCell = Field(default_factory=ConfusionCell)


class AgreementReport(BaseModel):
    schema_version: int = 1
    n: int
    status: str  # "ok" | "insufficient"
    accuracy: Optional[float] = None
    accuracy_ci95: Optional[Tuple[float, float]] = None
    precision_fail: Optional[float] = None
    recall_fail: Optional[float] = None
    f1_fail: Optional[float] = None
    cohen_kappa: Optional[float] = None
    kappa_ci95: Optional[Tuple[float, float]] = None
    confusion: Confusion = Field(default_factory=Confusion)
    spearman: Optional[float] = None
    mae: Optional[float] = None
    excluded: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class InterAnnotatorReport(BaseModel):
    schema_version: int = 1
    n: int
    status: str
    percent_agreement: Optional[float] = None
    cohen_kappa: Optional[float] = None
    fleiss_kappa: Optional[float] = None
    reviewers: Optional[int] = None
    notes: List[str] = Field(default_factory=list)


class ThresholdPoint(BaseModel):
    threshold: float
    accuracy: float
    kappa: Optional[float] = None
    precision_fail: Optional[float] = None
    recall_fail: Optional[float] = None
    agree: Optional[int] = None  # rows whose judge verdict matches the human label
    n: Optional[int] = None  # rows scored


class ThresholdReport(BaseModel):
    schema_version: int = 1
    objective: str
    curve: List[ThresholdPoint] = Field(default_factory=list)
    best: Optional[ThresholdPoint] = None
    current: Optional[ThresholdPoint] = None
    unstable: bool = False
    # Plain-language explanation of a proposed threshold move (populated by
    # threshold_sweep_run; empty for the plain threshold_sweep).
    flipped: List[Dict[str, Any]] = Field(default_factory=list)
    margin: Optional[float] = None
    rationale_text: str = ""
    notes: List[str] = Field(default_factory=list)


class CriterionStats(BaseModel):
    schema_version: int = 1
    index: int
    criterion: str
    verdict_distribution: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    skip_rate: float = 0.0
    false_fail_culprit_case_ids: List[str] = Field(default_factory=list)
    fully_on_false_pass_case_ids: List[str] = Field(default_factory=list)


# ── Extraction helpers ───────────────────────────────────────────────────────
def _judge_verdict(case: Any) -> Optional[str]:
    """Judge's pass/fail decision, or None when the judge gave no verdict."""
    jr = getattr(case, "judge_result", None)
    if not jr:
        return None
    if jr.get("error"):
        return None
    log = jr.get("evaluation_log") or {}
    if log.get("not_applicable_only"):
        return None  # every criterion not_applicable → no verdict
    if "kept_criteria" in log and len(log.get("kept_criteria") or []) == 0:
        return None  # every criterion skipped → n/a
    success = jr.get("success")
    if success is None:
        return None
    return "pass" if success else "fail"


def _judge_score(case: Any) -> Optional[float]:
    jr = getattr(case, "judge_result", None)
    if not jr:
        return None
    s = jr.get("score")
    return None if s is None else float(s)


def _current_threshold(cases: List[Any], default: float = 0.5) -> float:
    for c in cases:
        jr = getattr(c, "judge_result", None) or {}
        log = jr.get("evaluation_log") or {}
        t = log.get("threshold")
        if t is not None:
            return float(t)
    return default


def _split_included(cases: List[Any]):
    """Return (included_cases, excluded_ids)."""
    included, excluded = [], []
    for c in cases:
        if _judge_verdict(c) is None:
            excluded.append(c.id)
        else:
            included.append(c)
    return included, excluded


# ── Math primitives ──────────────────────────────────────────────────────────
def _wilson(k: int, n: int) -> Optional[Tuple[float, float]]:
    if n == 0:
        return None
    p = k / n
    z = _Z95
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, round(center - half, 4)), min(1.0, round(center + half, 4)))


def _cohen_kappa(a: List[str], b: List[str]) -> Optional[float]:
    n = len(a)
    if n == 0:
        return None
    cats = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    if abs(1 - pe) < 1e-12:
        return None
    return round((po - pe) / (1 - pe), 4)


def _fleiss_kappa(items: List[List[str]]) -> Optional[float]:
    """Fleiss' κ generalised to a variable number of raters per item."""
    cats = sorted({v for row in items for v in row})
    if not cats:
        return None
    p_bar_terms = []
    cat_total = {c: 0 for c in cats}
    total_ratings = 0
    for row in items:
        n_i = len(row)
        if n_i < 2:
            continue
        counts = {c: row.count(c) for c in cats}
        pi = (sum(v * v for v in counts.values()) - n_i) / (n_i * (n_i - 1))
        p_bar_terms.append(pi)
        for c in cats:
            cat_total[c] += counts[c]
        total_ratings += n_i
    if not p_bar_terms or total_ratings == 0:
        return None
    p_bar = sum(p_bar_terms) / len(p_bar_terms)
    pe = sum((cat_total[c] / total_ratings) ** 2 for c in cats)
    if abs(1 - pe) < 1e-12:
        return None
    return round((p_bar - pe) / (1 - pe), 4)


def _rank(values: List[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # average rank (1-based)
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(x: List[float], y: List[float]) -> Optional[float]:
    n = len(x)
    if n < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def _spearman(x: List[float], y: List[float]) -> Optional[float]:
    r = _pearson(_rank(x), _rank(y))
    return None if r is None else round(r, 4)


def _mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5**n)
    return min(1.0, 2 * tail)


# ── Binary (fail-positive) metric bundle ─────────────────────────────────────
def _binary_metrics(human: List[str], judge: List[str]) -> Dict[str, Any]:
    tf = sum(1 for h, j in zip(human, judge) if h == "fail" and j == "fail")
    tp = sum(1 for h, j in zip(human, judge) if h == "pass" and j == "pass")
    fp_ = sum(1 for h, j in zip(human, judge) if h == "fail" and j == "pass")  # false_pass
    ff = sum(1 for h, j in zip(human, judge) if h == "pass" and j == "fail")  # false_fail
    n = len(human)
    accuracy = (tf + tp) / n if n else None
    precision_fail = tf / (tf + ff) if (tf + ff) else None
    recall_fail = tf / (tf + fp_) if (tf + fp_) else None
    if precision_fail and recall_fail and (precision_fail + recall_fail):
        f1_fail = 2 * precision_fail * recall_fail / (precision_fail + recall_fail)
    else:
        f1_fail = None
    recall_pass = tp / (tp + ff) if (tp + ff) else None
    if recall_fail is not None and recall_pass is not None:
        balanced = (recall_fail + recall_pass) / 2
    else:
        balanced = None
    return {
        "true_fail": tf,
        "true_pass": tp,
        "false_pass": fp_,
        "false_fail": ff,
        "accuracy": accuracy,
        "precision_fail": precision_fail,
        "recall_fail": recall_fail,
        "f1_fail": f1_fail,
        "balanced_accuracy": balanced,
        "kappa": _cohen_kappa(human, judge),
    }


# ── Public: agreement ────────────────────────────────────────────────────────
def agreement(cases: List[Any], min_n: int = 20) -> AgreementReport:
    included, excluded = _split_included(cases)
    n = len(included)

    # _split_included guarantees every included case has a non-None verdict.
    human: List[str] = [c.label.verdict for c in included]
    judge: List[str] = [cast(str, _judge_verdict(c)) for c in included]

    confusion = Confusion()
    for c, h, j in zip(included, human, judge):
        if h == "fail" and j == "fail":
            cell = confusion.true_fail
        elif h == "pass" and j == "pass":
            cell = confusion.true_pass
        elif h == "fail" and j == "pass":
            cell = confusion.false_pass
        else:
            cell = confusion.false_fail
        cell.count += 1
        cell.case_ids.append(c.id)

    notes: List[str] = []
    if n < min_n:
        notes.append(f"n={n} < min_n={min_n}: numeric metrics withheld.")
        return AgreementReport(
            n=n, status="insufficient", confusion=confusion, excluded=excluded, notes=notes
        )

    m = _binary_metrics(human, judge)
    if len(set(human)) < 2:
        notes.append("human labels are a single class; κ is undefined.")
    if len(set(judge)) < 2:
        notes.append("judge verdicts are a single class; κ is undefined.")

    report = AgreementReport(
        n=n,
        status="ok",
        accuracy=round(m["accuracy"], 4) if m["accuracy"] is not None else None,
        accuracy_ci95=_wilson(m["true_fail"] + m["true_pass"], n),
        precision_fail=round(m["precision_fail"], 4) if m["precision_fail"] is not None else None,
        recall_fail=round(m["recall_fail"], 4) if m["recall_fail"] is not None else None,
        f1_fail=round(m["f1_fail"], 4) if m["f1_fail"] is not None else None,
        cohen_kappa=m["kappa"],
        kappa_ci95=_kappa_bootstrap_ci(human, judge),
        confusion=confusion,
        excluded=excluded,
        notes=notes,
    )

    hs = [c.label.score for c in included]
    js = [_judge_score(c) for c in included]
    if all(s is not None for s in hs) and all(s is not None for s in js) and n >= 2:
        hf = [float(cast(float, s)) for s in hs]
        jf = [float(cast(float, s)) for s in js]
        report.spearman = _spearman(hf, jf)
        report.mae = round(sum(abs(a - b) for a, b in zip(hf, jf)) / n, 4)
    return report


def _kappa_bootstrap_ci(human: List[str], judge: List[str]) -> Optional[Tuple[float, float]]:
    n = len(human)
    if n < 2:
        return None
    rng = random.Random(_BOOTSTRAP_SEED)
    samples: List[float] = []
    idx = range(n)
    for _ in range(_BOOTSTRAP_N):
        pick = [rng.randrange(n) for _ in idx]
        k = _cohen_kappa([human[i] for i in pick], [judge[i] for i in pick])
        if k is not None:
            samples.append(k)
    if len(samples) < 20:
        return None
    samples.sort()
    lo = samples[int(0.025 * len(samples))]
    hi = samples[min(len(samples) - 1, int(0.975 * len(samples)))]
    return (round(lo, 4), round(hi, 4))


# ── Public: inter-annotator ──────────────────────────────────────────────────
def inter_annotator(cases: List[Any]) -> InterAnnotatorReport:
    per_case: List[List[str]] = []
    for c in cases:
        labels = [c.label] + list(c.extra_labels or [])
        verdicts = [l.verdict for l in labels if l and l.verdict]
        if len(verdicts) >= 2:
            per_case.append(verdicts)

    n = len(per_case)
    if n == 0:
        return InterAnnotatorReport(
            n=0, status="insufficient", notes=["no cases with ≥2 reviewers."]
        )

    pct = round(sum(1.0 for v in per_case if len(set(v)) == 1) / n, 4)
    max_raters = max(len(v) for v in per_case)
    cohen = None
    fleiss = None
    if all(len(v) == 2 for v in per_case):
        cohen = _cohen_kappa([v[0] for v in per_case], [v[1] for v in per_case])
    if max_raters >= 3:
        fleiss = _fleiss_kappa(per_case)
    return InterAnnotatorReport(
        n=n,
        status="ok",
        percent_agreement=pct,
        cohen_kappa=cohen,
        fleiss_kappa=fleiss,
        reviewers=max_raters,
    )


# ── Public: threshold sweep ──────────────────────────────────────────────────
def threshold_sweep(
    cases: List[Any],
    objective: str = "kappa",
    min_recall_fail: Optional[float] = None,
) -> ThresholdReport:
    included, _ = _split_included(cases)
    pairs: List[Tuple[str, float]] = [
        (c.label.verdict, cast(float, _judge_score(c)))
        for c in included
        if _judge_score(c) is not None
    ]
    notes: List[str] = []
    if not pairs:
        return ThresholdReport(objective=objective, notes=["no judge scores available."])

    current = _current_threshold(cases)
    scores = sorted({s for _, s in pairs})
    candidates = sorted(
        {round((scores[i] + scores[i + 1]) / 2, 6) for i in range(len(scores) - 1)}
        | {round(current, 6)}
    )

    def point(t: float) -> ThresholdPoint:
        human = [h for h, _ in pairs]
        judge = ["fail" if s < t else "pass" for _, s in pairs]
        m = _binary_metrics(human, judge)
        return ThresholdPoint(
            threshold=t,
            accuracy=round(m["accuracy"], 4),
            kappa=m["kappa"],
            precision_fail=(
                round(m["precision_fail"], 4) if m["precision_fail"] is not None else None
            ),
            recall_fail=round(m["recall_fail"], 4) if m["recall_fail"] is not None else None,
            agree=sum(1 for a, b in zip(human, judge) if a == b),
            n=len(pairs),
        )

    curve = [point(t) for t in candidates]

    def obj_value(p: ThresholdPoint) -> float:
        if min_recall_fail is not None:
            # max precision subject to a recall floor
            if (p.recall_fail or 0.0) < min_recall_fail:
                return -1.0 + (p.recall_fail or 0.0)  # below floor → rank by recall
            return 1.0 + (p.precision_fail or 0.0)
        if objective == "accuracy":
            return p.accuracy
        if objective == "balanced_accuracy":
            human = [h for h, _ in pairs]
            judge = ["fail" if s < p.threshold else "pass" for _, s in pairs]
            return _binary_metrics(human, judge)["balanced_accuracy"] or 0.0
        return p.kappa if p.kappa is not None else -1.0  # default: kappa

    objs = [obj_value(p) for p in curve]
    best_val = max(objs)

    def tiebreak(i: int) -> Tuple[float, float]:
        return (abs(curve[i].threshold - current), curve[i].threshold)

    best_i = min((i for i, o in enumerate(objs) if o == best_val), key=tiebreak)

    near_thresholds = [curve[i].threshold for i, o in enumerate(objs) if o >= best_val - 0.02]
    unstable = bool(near_thresholds) and (max(near_thresholds) - min(near_thresholds)) >= 0.02

    current_point = min(curve, key=lambda p: abs(p.threshold - current))
    if min_recall_fail is not None and all((p.recall_fail or 0.0) < min_recall_fail for p in curve):
        notes.append(f"no threshold reaches recall_fail ≥ {min_recall_fail}; best by recall shown.")

    return ThresholdReport(
        objective=objective if min_recall_fail is None else f"precision@recall≥{min_recall_fail}",
        curve=curve,
        best=curve[best_i],
        current=current_point,
        unstable=unstable,
        notes=notes,
    )


def threshold_sweep_run(run: Any, current: float, objective: str = "accuracy") -> ThresholdReport:
    """Tune the pass/fail threshold on an EvaluationRun's own scores.

    ``run`` is any object exposing ``results`` with ``case_id`` / ``human_verdict``
    / ``score`` per row (an ``EvaluationRun``) — kept duck-typed to avoid a import
    cycle with the candidates module. Unlike :func:`threshold_sweep`, the threshold
    moves only when it strictly gains at least one more agreeing row, and the
    report carries a human-readable explanation (``flipped`` rows, ``margin`` to the
    nearest score, ``rationale_text``).
    """
    pairs: List[Tuple[str, str, float]] = [
        (r.case_id, r.human_verdict, r.score)
        for r in run.results
        if r.human_verdict is not None and r.score is not None
    ]
    if not pairs:
        return ThresholdReport(objective=objective, notes=["no scored rows with a human label."])

    n = len(pairs)
    scores = sorted({s for _, _, s in pairs})
    candidates = sorted(
        {round((scores[i] + scores[i + 1]) / 2, 6) for i in range(len(scores) - 1)}
        | {round(current, 6)}
    )

    def judge_at(t: float) -> Dict[str, str]:
        return {cid: ("fail" if s < t else "pass") for cid, _, s in pairs}

    def point(t: float) -> ThresholdPoint:
        human = [h for _, h, _ in pairs]
        judge = ["fail" if s < t else "pass" for _, _, s in pairs]
        m = _binary_metrics(human, judge)
        return ThresholdPoint(
            threshold=t,
            accuracy=round(m["accuracy"], 4),
            kappa=m["kappa"],
            precision_fail=(
                round(m["precision_fail"], 4) if m["precision_fail"] is not None else None
            ),
            recall_fail=round(m["recall_fail"], 4) if m["recall_fail"] is not None else None,
            agree=sum(1 for a, b in zip(human, judge) if a == b),
            n=n,
        )

    curve = [point(t) for t in candidates]
    current_point = point(current)

    def obj(p: ThresholdPoint) -> float:
        if objective == "kappa":
            return p.kappa if p.kappa is not None else -1.0
        return float(p.agree or 0)  # "accuracy" == agreeing-row count

    best_val = max(obj(p) for p in curve)
    best_p = min(
        (p for p in curve if obj(p) == best_val),
        key=lambda p: (abs(p.threshold - current), p.threshold),
    )

    # Move only on a strict gain of ≥1 agreeing row.
    moved = (best_p.agree or 0) >= (current_point.agree or 0) + 1 and abs(
        best_p.threshold - current
    ) > 1e-9
    best_point = best_p if moved else current_point
    best_t = best_point.threshold

    flipped: List[Dict[str, Any]] = []
    if moved:
        jcur, jbest = judge_at(current), judge_at(best_t)
        for cid, h, s in pairs:
            if jcur[cid] != jbest[cid]:
                flipped.append(
                    {
                        "case_id": cid,
                        "score": s,
                        "human_verdict": h,
                        "old_verdict": jcur[cid],
                        "new_verdict": jbest[cid],
                    }
                )

    margin = round(min(abs(best_t - s) for s in scores), 6) if scores else None
    unstable = margin is not None and margin < 0.02
    rationale_text = _threshold_rationale(
        moved, best_t, current, best_point, current_point, n, flipped
    )

    return ThresholdReport(
        objective=objective,
        curve=curve,
        best=best_point,
        current=current_point,
        unstable=unstable,
        flipped=flipped,
        margin=margin,
        rationale_text=rationale_text,
    )


def _threshold_rationale(moved, best_t, current, best_point, current_point, n, flipped) -> str:
    if not moved:
        return (
            f"Threshold stays at {current:.2f}: {current_point.agree} of {n} rows match your "
            "labels, and no threshold matches at least one more."
        )
    gained = [f for f in flipped if f["new_verdict"] == f["human_verdict"]]
    sentence = (
        f"At {best_t:.2f}, {best_point.agree} of {n} rows match your labels "
        f"(was {current_point.agree} at {current:.2f})."
    )
    if gained:
        ids = ", ".join(str(f["case_id"]) for f in gained[:6])
        sc = ", ".join(f"{f['score']:.2f}" for f in gained[:6])
        verb = gained[0]["new_verdict"]
        sentence += f" Rows {ids} (scores {sc}) become {verb}, as you labeled them."
    return sentence


# ── Public: criterion attribution ────────────────────────────────────────────
def criterion_attribution(cases: List[Any]) -> List[CriterionStats]:
    included = [c for c in cases if _judge_verdict(c) is not None]
    # Derive the config criteria template from the first available judge result.
    template: List[str] = []
    for c in included:
        log = (c.judge_result or {}).get("evaluation_log") or {}
        if log.get("strategy") == "verdict" and log.get("evaluation_criteria_template"):
            template = list(log["evaluation_criteria_template"])
            break
    if not template:
        return []

    stats = [CriterionStats(index=i, criterion=t) for i, t in enumerate(template)]
    skip_counts = [0] * len(template)
    n_verdict_cases = 0

    for c in included:
        log = (c.judge_result or {}).get("evaluation_log") or {}
        if log.get("strategy") != "verdict":
            continue
        n_verdict_cases += 1
        kept = list(log.get("kept_criteria") or [])
        verdicts = list(log.get("verdicts") or [])
        threshold = float(
            log.get("threshold", c.judge_result.get("evaluation_log", {}).get("threshold", 0.5))
        )
        temperature = float(log.get("temperature", 0.8))
        weights = _weights_over_kept(log, kept, verdicts)

        human = c.label.verdict
        judge = cast(str, _judge_verdict(c))  # included ⇒ non-None
        cell = _cell_name(human, judge)

        for i, text in enumerate(template):
            pos = kept.index(text) if text in kept else None
            if pos is None:
                skip_counts[i] += 1
                continue
            label = (verdicts[pos] or {}).get("verdict", "none") if pos < len(verdicts) else "none"
            dist = stats[i].verdict_distribution.setdefault(cell, {})
            dist[label] = dist.get(label, 0) + 1

            if cell == "false_fail" and label in _NEGATIVE_VERDICTS:
                remaining = [w for k, w in enumerate(weights) if k != pos]
                if remaining and score_agg(remaining, temperature=temperature) >= threshold:
                    stats[i].false_fail_culprit_case_ids.append(c.id)
            if cell == "false_pass" and label == "fully":
                stats[i].fully_on_false_pass_case_ids.append(c.id)

    if n_verdict_cases:
        for i in range(len(template)):
            stats[i].skip_rate = round(skip_counts[i] / n_verdict_cases, 4)
    return stats


def _cell_name(human: str, judge: str) -> str:
    if human == "fail" and judge == "fail":
        return "true_fail"
    if human == "pass" and judge == "pass":
        return "true_pass"
    if human == "fail" and judge == "pass":
        return "false_pass"
    return "false_fail"


def _weights_over_kept(log: dict, kept: List[str], verdicts: List[dict]) -> List[float]:
    raw = log.get("verdict_weights")
    weights: List[float] = []
    for pos in range(len(kept)):
        w = None
        if isinstance(raw, dict):
            w = raw.get(pos, raw.get(str(pos)))
        if w is None and pos < len(verdicts):
            w = VERDICT_WEIGHTS.get((verdicts[pos] or {}).get("verdict", "none"), 0.0)
        weights.append(float(w if w is not None else 0.0))
    return weights
