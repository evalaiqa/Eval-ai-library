# calibration/models.py
"""
Data model for Custom Eval calibration.

Everything here is pydantic v2 and round-trips losslessly through
``model_dump(mode="json")`` ⇄ ``model_validate`` so the platform can store the
objects in JSONB. Every model carries ``schema_version`` so a stored object can
still be read after the library is upgraded.

These models describe *how the platform versions a metric* (`CustomEvalConfig`),
*what a human label looks like* (`HumanLabel`, `LabeledCase`), and *the material
a calibration job works over* (`CalibrationSet`, `Split`, `Proposal`). None of
them call an LLM.
"""

import hashlib
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, field_validator

from eval_lib.few_shot import FEW_SHOT_MAX_EXAMPLES, FewShotExample
from eval_lib.testcases_schema import EvalTestCase

_ALLOWED_STRATEGIES = {"verdict", "direct"}
_ALLOWED_AGGREGATIONS = {"majority", "median", "mean"}
_MAX_RUNS = 15
_PROPOSAL_KINDS = {
    "rewrite_criterion",
    "add_criterion",
    "remove_criterion",
    "split_criterion",
    "set_temperature",
    "set_strategy",
    "set_threshold",
    "add_few_shot",
    "review_labels",
}


class CustomEvalConfig(BaseModel):
    """A full, versionable description of a CustomEvalMetric configuration.

    Validators mirror the metric constructor's limits so a config that passes
    here will also construct a metric. ``to_metric(model)`` builds the live
    metric (model is supplied separately — a config is model-agnostic);
    ``from_params(dict)`` reads the platform's metric-params dict.
    """

    schema_version: int = 1
    name: str
    evaluation_criteria: List[str]
    strategy: str = "verdict"
    n_runs: int = 1
    aggregation: str = "median"
    temperature: float = 0.8
    threshold: float = 0.5
    few_shot_examples: List[FewShotExample] = Field(default_factory=list)

    @field_validator("evaluation_criteria")
    @classmethod
    def _clean_criteria(cls, v: Any) -> List[str]:
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("evaluation_criteria must be a non-empty list.")
        cleaned = [str(c).strip() for c in v if str(c).strip()]
        if not cleaned:
            raise ValueError("evaluation_criteria contained only empty/whitespace items.")
        return cleaned

    @field_validator("strategy")
    @classmethod
    def _check_strategy(cls, v: str) -> str:
        if v not in _ALLOWED_STRATEGIES:
            raise ValueError(f"strategy must be one of {sorted(_ALLOWED_STRATEGIES)}, got {v!r}")
        return v

    @field_validator("n_runs")
    @classmethod
    def _check_runs(cls, v: int) -> int:
        if not isinstance(v, int) or v < 1:
            raise ValueError(f"n_runs must be a positive integer, got {v!r}")
        if v > _MAX_RUNS:
            raise ValueError(f"n_runs={v} exceeds safety cap of {_MAX_RUNS}")
        return v

    @field_validator("aggregation")
    @classmethod
    def _check_agg(cls, v: str) -> str:
        if v not in _ALLOWED_AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {sorted(_ALLOWED_AGGREGATIONS)}, got {v!r}"
            )
        return v

    @field_validator("few_shot_examples")
    @classmethod
    def _check_fewshot(cls, v: List[FewShotExample]) -> List[FewShotExample]:
        if v and len(v) > FEW_SHOT_MAX_EXAMPLES:
            raise ValueError(
                f"few_shot_examples has {len(v)} items; the maximum is {FEW_SHOT_MAX_EXAMPLES}."
            )
        return v

    def to_metric(self, model: Any):
        """Build the live ``CustomEvalMetric`` for this config against ``model``."""
        from eval_lib.metrics.custom_metric.custom_eval import CustomEvalMetric

        return CustomEvalMetric(
            model=model,
            threshold=self.threshold,
            name=self.name,
            evaluation_criteria=list(self.evaluation_criteria),
            strategy=self.strategy,
            n_runs=self.n_runs,
            aggregation=self.aggregation,
            temperature=self.temperature,
            few_shot_examples=[e.model_copy(deep=True) for e in self.few_shot_examples],
        )

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "CustomEvalConfig":
        """Read the platform's metric-params dict (registry names); ignore unknowns
        such as ``judge_backend``."""
        return cls(
            name=params.get("name", "custom"),
            evaluation_criteria=params["evaluation_criteria"],
            strategy=params.get("strategy", "verdict"),
            n_runs=params.get("n_runs", 1),
            aggregation=params.get("aggregation", "median"),
            temperature=params.get("temperature", 0.8),
            threshold=params.get("threshold", 0.5),
            few_shot_examples=params.get("few_shot_examples") or [],
        )


class HumanLabel(BaseModel):
    """A reviewer's final decision on a case.

    ``criterion_verdicts`` (optional) carries per-criterion pass/fail (or None
    where the reviewer did not judge that criterion); its index matches the
    config's criterion index.
    """

    schema_version: int = 1
    verdict: str  # "pass" | "fail"
    reason: Optional[str] = None
    score: Optional[float] = None
    criterion_verdicts: Optional[List[Optional[str]]] = None
    reviewer_id: Optional[str] = None

    @field_validator("verdict")
    @classmethod
    def _check_verdict(cls, v: str) -> str:
        if v not in ("pass", "fail"):
            raise ValueError(f"verdict must be 'pass' or 'fail', got {v!r}")
        return v


class LabeledCase(BaseModel):
    """A single human-labelled case.

    ``test_case`` carries the data (built-in fields + ``extra_fields``).
    Connector-only ``_meta`` (dataset_row / system_prompt / template_variable_map)
    is captured here in ``meta`` so it survives JSON persistence and can be
    re-attached before re-evaluation. ``judge_result`` is a previously saved
    ``evaluate()`` output, so a run need not re-score what it already has.
    """

    schema_version: int = 1
    id: str
    test_case: EvalTestCase
    label: HumanLabel
    extra_labels: List[HumanLabel] = Field(default_factory=list)
    judge_result: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None

    @classmethod
    def from_test_case(
        cls,
        id: str,
        test_case: EvalTestCase,
        label: HumanLabel,
        **kwargs: Any,
    ) -> "LabeledCase":
        """Build a LabeledCase, capturing any connector ``_meta`` off the test case."""
        meta = kwargs.pop("meta", None) or getattr(test_case, "_meta", None)
        return cls(id=id, test_case=test_case, label=label, meta=meta, **kwargs)


class Split(BaseModel):
    """Result of :meth:`CalibrationSet.split` — case ids assigned to each part."""

    schema_version: int = 1
    train: List[str]
    holdout: List[str]
    seed: int


class CalibrationSet(BaseModel):
    """Config + labelled cases + the placeholder names available in the data."""

    schema_version: int = 1
    config: CustomEvalConfig
    cases: List[LabeledCase]
    placeholders: Set[str] = Field(default_factory=set)

    def split(
        self,
        holdout_ratio: float = 0.3,
        seed: int = 0,
        pinned_train: Optional[List[str]] = None,
    ) -> Split:
        """Deterministic, stratified train/holdout split.

        Assignment is by a stable hash of ``seed:case.id`` compared against
        ``holdout_ratio`` — per case, not by rank — so adding new cases never
        moves existing ones between parts. Cases are stratified by the human
        verdict (pass/fail). ``pinned_train`` ids and every few-shot example's
        ``source_case_id`` are forced into train (never holdout). If a stratum
        with ≥2 eligible members would otherwise have an empty holdout, its
        lowest-hash member is forced into holdout so both classes appear.
        """
        pinned = set(pinned_train or [])
        fs_ids = {ex.source_case_id for ex in self.config.few_shot_examples if ex.source_case_id}
        forced_train = pinned | fs_ids

        order = {c.id: i for i, c in enumerate(self.cases)}
        strata: Dict[str, List[LabeledCase]] = {}
        forced: List[str] = []
        for c in self.cases:
            if c.id in forced_train:
                forced.append(c.id)
                continue
            strata.setdefault(c.label.verdict, []).append(c)

        train: List[str] = list(forced)
        holdout: List[str] = []
        for members in strata.values():
            scored = sorted(members, key=lambda c: _hash01(seed, c.id))
            local_hold = [c.id for c in scored if _hash01(seed, c.id) < holdout_ratio]
            local_train = [c.id for c in scored if _hash01(seed, c.id) >= holdout_ratio]
            if not local_hold and len(scored) >= 2:
                forced_hold = scored[0].id
                local_hold = [forced_hold]
                local_train = [c.id for c in scored if c.id != forced_hold]
            holdout.extend(local_hold)
            train.extend(local_train)

        train.sort(key=lambda i: order.get(i, 1 << 30))
        holdout.sort(key=lambda i: order.get(i, 1 << 30))
        return Split(train=train, holdout=holdout, seed=seed)


class Proposal(BaseModel):
    """A single proposed edit to a config.

    Produced by the analyst (0.8.1) and consumed by ``apply_proposals``. Which
    fields are read depends on ``kind``:

      * rewrite_criterion   → criterion_index, text
      * add_criterion       → text
      * remove_criterion    → criterion_index
      * split_criterion     → criterion_index, texts
      * set_temperature     → temperature
      * set_strategy        → strategy
      * set_threshold       → threshold
      * add_few_shot        → case_ids (ids of LabeledCases, from train)
      * review_labels       → case_ids (informational; changes nothing)
    """

    schema_version: int = 1
    id: str
    kind: str
    criterion_index: Optional[int] = None
    text: Optional[str] = None
    texts: Optional[List[str]] = None
    temperature: Optional[float] = None
    strategy: Optional[str] = None
    threshold: Optional[float] = None
    case_ids: Optional[List[str]] = None
    rationale: str = ""
    finding_ids: List[str] = Field(default_factory=list)
    evidence_case_ids: List[str] = Field(default_factory=list)
    expected_fixes: List[str] = Field(default_factory=list)
    risk_case_ids: List[str] = Field(default_factory=list)

    @field_validator("kind")
    @classmethod
    def _check_kind(cls, v: str) -> str:
        if v not in _PROPOSAL_KINDS:
            raise ValueError(f"kind must be one of {sorted(_PROPOSAL_KINDS)}, got {v!r}")
        return v


def _hash01(seed: int, case_id: str) -> float:
    """Stable hash of (seed, case_id) → float in [0, 1)."""
    digest = hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).hexdigest()
    return int(digest, 16) / float(1 << 256)
