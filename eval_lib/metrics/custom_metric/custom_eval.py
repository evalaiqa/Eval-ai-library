# custom_eval.py
"""
Custom Evaluation Metric with pluggable scoring strategies.

Two evaluation strategies are supported:

  strategy="verdict" (default, backward-compatible)
      The judge returns one verdict per criterion, chosen from
      (fully / mostly / partial / minor / none). Verdict labels are mapped
      to weights and aggregated into a final score via TCVA
      (Temperature-Controlled Verdict Aggregation).

  strategy="direct"
      The judge returns a single integer score 0-10 for the whole test case
      taken against ALL criteria together. That raw score is normalized to
      the 0.0-1.0 range (raw / 10).

Both strategies can be repeated `n_runs` times for consensus scoring
(default n_runs=1 → single call, no aggregation). When n_runs > 1 the
per-run judge results are combined with `aggregation`:

  "majority"  – most common verdict / most common integer score
  "median"    – median of the per-run weights / scores
  "mean"      – arithmetic mean of the per-run weights / scores

When n_runs > 1 the LLM sampling temperature is raised automatically so the
runs actually differ; otherwise consensus voting would be a no-op.

Placeholder / filtering rules apply to both strategies:
    - A criterion with no {{placeholders}} at all is skipped.
    - A criterion referencing an unknown placeholder is skipped.
    - If every criterion is filtered out, the metric returns score=0.
"""

import asyncio
import json
import re
from collections import Counter
from statistics import mean as _mean
from statistics import median as _median
from typing import Any, Dict, List, Optional, Tuple

from eval_lib.few_shot import FEW_SHOT_DEFAULT_MAX_CHARS
from eval_lib.llm_client import chat_complete
from eval_lib.metric_pattern import MetricPattern
from eval_lib.testcases_schema import EvalTestCase
from eval_lib.utils import extract_json_block, score_agg

_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


# Hard safety ceiling on the number of criteria sent to the judge. Not a
# tunable knob — the goal is only to catch accidental disasters (e.g. a user
# pasting 500 rows from a spreadsheet into the criteria list) before they
# turn into a runaway LLM bill. Real chat-list sizes are 3-10; if you legitimately
# need more, raise this constant.
_HARD_SAFETY_CAP = 50

# Safety ceiling on the number of consensus runs per test case. Same rationale
# as _HARD_SAFETY_CAP — catches typos (n_runs=100) before the bill catches you.
_MAX_RUNS = 15

# LLM sampling temperature used when n_runs > 1. Non-zero variance is required
# for consensus voting to be meaningful; at 0.0 the judge would return the
# same answer every run.
_CONSENSUS_LLM_TEMPERATURE = 0.7


# Verdict weights for scoring.
VERDICT_WEIGHTS = {
    "fully": 1.0,
    "mostly": 0.9,
    "partial": 0.7,
    "minor": 0.3,
    "none": 0.0,
}

_ALLOWED_STRATEGIES = {"verdict", "direct"}
_ALLOWED_AGGREGATIONS = {"majority", "median", "mean"}

# A verdict the judge may use for a conditional criterion whose condition does
# not hold for this input. not_applicable criteria are dropped from the
# aggregation (like skipped criteria); they carry no weight.
_NOT_APPLICABLE = "not_applicable"


class CustomEvalMetric(MetricPattern):
    """Custom evaluation against user-defined criteria.

    See module docstring for the two scoring strategies and consensus
    aggregation. The metric is reusable across test cases — per-row data is
    resolved fresh every `evaluate()` call.
    """

    name = "customEval"

    def __init__(
        self,
        model: str,
        threshold: float,
        name: str,
        evaluation_criteria: List[str],
        strategy: str = "verdict",
        n_runs: int = 1,
        aggregation: str = "median",
        temperature: float = 0.8,
        verbose: bool = False,
        few_shot_examples: Optional[List] = None,
        few_shot_max_chars: int = FEW_SHOT_DEFAULT_MAX_CHARS,
    ):
        """
        Args:
            model: LLM model id (e.g. "openai:gpt-4o-mini").
            threshold: Pass/fail threshold for the aggregated score (0.0-1.0).
            name: Human-readable metric name; surfaced as "Custom: <name>".
            evaluation_criteria: Non-empty list of criteria. Each item should
                reference data via {{placeholders}}; criteria without any
                placeholder or with unknown placeholders are skipped at
                evaluate-time (with a warning in evaluation_log).
            strategy: "verdict" (default) — per-criterion verdicts aggregated
                via TCVA. "direct" — single overall integer score 0-10 from
                the judge, normalized to 0.0-1.0.
            n_runs: Number of times to call the judge per test case. Default 1
                (no consensus). When >1, the LLM sampling temperature is
                raised so the runs actually differ.
            aggregation: Only used when n_runs > 1. "majority" | "median" |
                "mean". For "verdict" strategy the aggregation is applied
                per-criterion across runs (majority=mode of verdict labels;
                median/mean=of the corresponding verdict weights). For
                "direct" strategy it is applied to the raw 0-10 scores.
            temperature: TCVA aggregation temperature (NOT LLM sampling — see
                n_runs for that). Only used by strategy="verdict". Low (~0.1)
                ≈ strict (min), 0.5 ≈ arithmetic mean, high (~1.5) ≈ lenient
                (max). Default 0.8.
            verbose: Enable per-result console logging.
            few_shot_examples: Optional list of FewShotExample (or dicts) shown
                to the judge as calibration examples — how strictly the team
                grades. Reference only; they are not scored and placeholders in
                them are not substituted. At most 20 (ValueError otherwise).
                Omitting them keeps the prompt identical to earlier versions.
            few_shot_max_chars: Character budget for the whole examples block;
                trailing examples are dropped to fit (default 12000).
        """
        super().__init__(
            model=model,
            threshold=threshold,
            verbose=verbose,
            few_shot_examples=few_shot_examples,
            few_shot_max_chars=few_shot_max_chars,
        )

        if not isinstance(evaluation_criteria, list) or len(evaluation_criteria) == 0:
            raise ValueError("CustomEvalMetric requires a non-empty list of evaluation_criteria.")
        cleaned = [str(c).strip() for c in evaluation_criteria if str(c).strip()]
        if not cleaned:
            raise ValueError(
                "CustomEvalMetric: evaluation_criteria contained only empty/whitespace items."
            )

        if strategy not in _ALLOWED_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {sorted(_ALLOWED_STRATEGIES)}, got {strategy!r}"
            )
        if not isinstance(n_runs, int) or n_runs < 1:
            raise ValueError(f"n_runs must be a positive integer, got {n_runs!r}")
        if n_runs > _MAX_RUNS:
            raise ValueError(
                f"n_runs={n_runs} exceeds safety cap of {_MAX_RUNS}; "
                f"raise _MAX_RUNS in custom_eval.py if you truly need more."
            )
        if aggregation not in _ALLOWED_AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {sorted(_ALLOWED_AGGREGATIONS)}, got {aggregation!r}"
            )

        self.custom_name = name
        self.evaluation_criteria = cleaned
        self.strategy = strategy
        self.n_runs = n_runs
        self.aggregation = aggregation
        self.temperature = temperature

    # ==================== TEMPLATE CONTEXT ====================

    @staticmethod
    def _build_substitution_context(test_case: EvalTestCase) -> Dict[str, Any]:
        """Collect every value addressable from the test case.

        Priority (later wins):
          1. _meta["dataset_row"]  (raw dataset row from the connector)
          2. extra_fields          (user-provided ad-hoc fields)
          3. built-in EvalTestCase fields
          4. _meta["system_prompt"]

        Connector aliases (`template_variable_map`) are expanded inline so the
        caller only needs to look up by the placeholder name as written.
        """
        meta = getattr(test_case, "_meta", None) or {}
        values: Dict[str, Any] = {}

        row = meta.get("dataset_row") or {}
        if isinstance(row, dict):
            values.update(row)

        if test_case.extra_fields:
            values.update(test_case.extra_fields)

        values["input"] = test_case.input
        values["actual_output"] = test_case.actual_output
        if test_case.expected_output is not None:
            values["expected_output"] = test_case.expected_output
        if test_case.retrieval_context:
            values["retrieval_context"] = "\n".join(test_case.retrieval_context)

        sys_prompt = meta.get("system_prompt")
        if sys_prompt:
            values["system_prompt"] = sys_prompt

        variable_map = meta.get("template_variable_map") or {}
        if isinstance(variable_map, dict):
            for alias, real_name in variable_map.items():
                if real_name in values and alias not in values:
                    values[alias] = values[real_name]

        return values

    @staticmethod
    def _extract_placeholders(text: str) -> List[str]:
        """Return placeholder names in order of first appearance, deduped."""
        seen: Dict[str, None] = {}
        for name in _PLACEHOLDER_RE.findall(text or ""):
            if name not in seen:
                seen[name] = None
        return list(seen.keys())

    # ==================== FILTERING ====================

    @classmethod
    def _filter_criteria(
        cls,
        criteria: List[str],
        values: Dict[str, Any],
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Return (kept_criteria, skipped_info).

        A criterion is kept only if it has at least one placeholder AND every
        placeholder it references is present in `values`.
        """
        kept: List[str] = []
        skipped: List[Dict[str, Any]] = []
        for c in criteria:
            placeholders = cls._extract_placeholders(c)
            if not placeholders:
                skipped.append(
                    {
                        "criterion": c,
                        "reason": "no_placeholders",
                        "detail": "Criterion contains no {{placeholders}}; nothing to evaluate against.",
                    }
                )
                continue
            missing = [p for p in placeholders if p not in values]
            if missing:
                skipped.append(
                    {
                        "criterion": c,
                        "reason": "unknown_placeholders",
                        "detail": f"Criterion references unknown placeholder(s): {missing}",
                        "missing": missing,
                    }
                )
                continue
            kept.append(c)
        return kept, skipped

    @staticmethod
    def _collect_used_placeholders(criteria: List[str]) -> List[str]:
        """Union of all placeholders mentioned across kept criteria, in order."""
        seen: Dict[str, None] = {}
        for c in criteria:
            for p in _PLACEHOLDER_RE.findall(c):
                if p not in seen:
                    seen[p] = None
        return list(seen.keys())

    # ==================== PROMPT ====================

    @staticmethod
    def _format_value(value: Any) -> str:
        """How a single DATA-block value is rendered for the judge."""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, indent=2)
        return str(value)

    @staticmethod
    def _prompt_label_help() -> str:
        return """Rate how well each criterion is satisfied (worst → best):

none    – criterion not satisfied at all
minor   – criterion minimally addressed
partial – criterion partially satisfied
mostly  – criterion largely satisfied with minor gaps
fully   – criterion fully satisfied"""

    @classmethod
    def _build_data_block(
        cls,
        used_names: List[str],
        values: Dict[str, Any],
    ) -> str:
        """Render the DATA: block listing only placeholders actually referenced."""
        lines = []
        for name in used_names:
            rendered = cls._format_value(values[name])
            if "\n" in rendered:
                lines.append(f"- {name}:\n{rendered}")
            else:
                lines.append(f"- {name}: {rendered}")
        return "\n".join(lines)

    @classmethod
    def _prompt_evaluate(
        cls,
        kept_criteria: List[str],
        used_names: List[str],
        values: Dict[str, Any],
        few_shot_block: str = "",
    ) -> str:
        data_block = cls._build_data_block(used_names, values)
        criteria_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(kept_criteria))
        return f"""{cls._prompt_label_help()}{few_shot_block}

DATA:
{data_block}

EVALUATION CRITERIA:
{criteria_text}

Task: For EACH criterion above, decide how well it is satisfied by the DATA.
The {{{{placeholders}}}} inside each criterion refer to entries in the DATA block.
Use exactly one of: fully, mostly, partial, minor, none, not_applicable.
Use not_applicable ONLY when a criterion states a condition (for example "If ...")
that this input does not meet; never use it to avoid judging.

**
Return JSON array with exactly {len(kept_criteria)} verdicts:
[
  {{"verdict": "fully|mostly|partial|minor|none|not_applicable", "reason": "<one sentence>"}},
  ...
]
**

JSON:"""

    @classmethod
    def _prompt_direct(
        cls,
        kept_criteria: List[str],
        used_names: List[str],
        values: Dict[str, Any],
        few_shot_block: str = "",
    ) -> str:
        data_block = cls._build_data_block(used_names, values)
        criteria_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(kept_criteria))
        return f"""You are evaluating how well the DATA satisfies the EVALUATION CRITERIA.{few_shot_block}

DATA:
{data_block}

EVALUATION CRITERIA:
{criteria_text}

Task: Rate on an integer scale from 0 to 10 how well the DATA satisfies ALL
the criteria above, taken together.
    0  = criteria not satisfied at all
    5  = partially satisfied
    10 = fully satisfied
The {{{{placeholders}}}} inside each criterion refer to entries in the DATA block.

**
Return JSON:
{{"score": <integer 0-10>, "reason": "<one-sentence justification>"}}
**

JSON:"""

    # ==================== CORE EVALUATION ====================

    def _llm_temperature(self) -> float:
        """Sampling temperature for the judge LLM.

        n_runs=1 stays deterministic (0.0). n_runs>1 needs variance across
        runs or consensus voting would be a no-op.
        """
        return _CONSENSUS_LLM_TEMPERATURE if self.n_runs > 1 else 0.0

    async def _run_verdicts_once(
        self,
        kept_criteria: List[str],
        used_names: List[str],
        values: Dict[str, Any],
        few_shot_block: str = "",
    ) -> Tuple[List[Dict[str, str]], float]:
        prompt = self._prompt_evaluate(kept_criteria, used_names, values, few_shot_block)
        text, cost = await chat_complete(
            self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._llm_temperature(),
        )
        try:
            raw_json = extract_json_block(text)
            verdicts = json.loads(raw_json)
            if not isinstance(verdicts, list):
                raise ValueError("Expected JSON array of verdicts")
        except Exception as e:
            raise RuntimeError(f"Failed to parse verdicts: {e}\n{text}")

        if len(verdicts) < len(kept_criteria):
            verdicts.extend(
                [{"verdict": "none", "reason": "Missing evaluation"}]
                * (len(kept_criteria) - len(verdicts))
            )
        elif len(verdicts) > len(kept_criteria):
            verdicts = verdicts[: len(kept_criteria)]
        return verdicts, cost or 0.0

    async def _run_direct_once(
        self,
        kept_criteria: List[str],
        used_names: List[str],
        values: Dict[str, Any],
        few_shot_block: str = "",
    ) -> Tuple[int, str, float]:
        prompt = self._prompt_direct(kept_criteria, used_names, values, few_shot_block)
        text, cost = await chat_complete(
            self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._llm_temperature(),
        )
        try:
            raw_json = extract_json_block(text)
            obj = json.loads(raw_json)
            if not isinstance(obj, dict):
                raise ValueError("Expected JSON object with 'score' field")
            raw = obj.get("score")
            if raw is None:
                raise ValueError("Missing 'score' field")
            raw_score = int(round(float(raw)))
            raw_score = max(0, min(10, raw_score))
            reason = str(obj.get("reason", "")).strip()
        except Exception as e:
            raise RuntimeError(f"Failed to parse direct score: {e}\n{text}")
        return raw_score, reason, cost or 0.0

    async def evaluate(self, test_case: EvalTestCase) -> Dict[str, Any]:
        # Step 1: resolve all addressable data for this test case.
        values = self._build_substitution_context(test_case)

        # Step 2: safety ceiling on criteria before any LLM work.
        truncation_warning: Optional[str] = None
        criteria = list(self.evaluation_criteria)
        if len(criteria) > _HARD_SAFETY_CAP:
            truncation_warning = (
                f"evaluation_criteria had {len(criteria)} items; "
                f"truncated to safety ceiling of {_HARD_SAFETY_CAP}."
            )
            criteria = criteria[:_HARD_SAFETY_CAP]

        # Step 3: filter out criteria that cannot be evaluated.
        kept, skipped = self._filter_criteria(criteria, values)
        used_names = self._collect_used_placeholders(kept)

        if not kept:
            return self._empty_kept_result(test_case, skipped, truncation_warning)

        # Step 4: dispatch to strategy.
        if self.strategy == "verdict":
            return await self._evaluate_verdict_mode(
                test_case, criteria, kept, skipped, used_names, values, truncation_warning
            )
        return await self._evaluate_direct_mode(
            test_case, criteria, kept, skipped, used_names, values, truncation_warning
        )

    def _empty_kept_result(
        self,
        test_case: EvalTestCase,
        skipped: List[Dict[str, Any]],
        truncation_warning: Optional[str],
    ) -> Dict[str, Any]:
        reason = (
            "No criteria could be evaluated: "
            "all entries lacked placeholders or referenced unknown data."
        )
        evaluation_log: Dict[str, Any] = {
            "strategy": self.strategy,
            "n_runs": self.n_runs,
            "aggregation": self.aggregation if self.n_runs > 1 else None,
            "input_question": test_case.input,
            "actual_output": test_case.actual_output,
            "evaluation_criteria_template": list(self.evaluation_criteria),
            "kept_criteria": [],
            "skipped_criteria": skipped,
            "comment_skipped_criteria": "Each entry explains why the criterion was excluded from judging.",
            "data_used": {},
            "verdicts": [],
            "final_score": 0.0,
            "threshold": self.threshold,
            "temperature": self.temperature,
            "success": False,
            "summary": reason,
        }
        if truncation_warning:
            evaluation_log["truncation_warning"] = truncation_warning
        result = {
            "name": self.name,
            "score": 0.0,
            "success": False,
            "reason": reason,
            "evaluation_cost": 0.0,
            "evaluation_log": evaluation_log,
        }
        self.print_result(result)
        return result

    def _all_not_applicable_result(
        self,
        test_case: EvalTestCase,
        criteria: List[str],
        kept: List[str],
        skipped: List[Dict[str, Any]],
        used_names: List[str],
        values: Dict[str, Any],
        final_verdicts: List[Dict[str, Any]],
        not_applicable_criteria: List[Dict[str, Any]],
        few_shot_log: Dict[str, Any],
        total_cost: float,
        truncation_warning: Optional[str],
    ) -> Dict[str, Any]:
        """Every kept criterion was not_applicable → the row has no verdict.

        Flagged with ``not_applicable_only`` so the calibration layer excludes it
        from agreement, exactly like an all-skipped row.
        """
        reason = "All criteria were not_applicable to this input; no score."
        evaluation_log: Dict[str, Any] = {
            "strategy": "verdict",
            "n_runs": self.n_runs,
            "aggregation": self.aggregation if self.n_runs > 1 else None,
            "input_question": test_case.input,
            "actual_output": test_case.actual_output,
            "evaluation_criteria_template": list(self.evaluation_criteria),
            "kept_criteria": kept,
            "skipped_criteria": skipped,
            "data_used": {name: values[name] for name in used_names},
            "verdicts": final_verdicts,
            "not_applicable_criteria": not_applicable_criteria,
            "not_applicable_only": True,
            "verdict_weights": {},
            "final_score": 0.0,
            "threshold": self.threshold,
            "temperature": self.temperature,
            "success": False,
            "summary": reason,
        }
        if self.few_shot_examples:
            evaluation_log["few_shot"] = few_shot_log
        if truncation_warning:
            evaluation_log["truncation_warning"] = truncation_warning
        result = {
            "name": self.name,
            "score": 0.0,
            "success": False,
            "reason": reason,
            "evaluation_cost": round(total_cost, 6),
            "evaluation_log": evaluation_log,
        }
        self.print_result(result)
        return result

    # ==================== STRATEGY: VERDICT ====================

    async def _evaluate_verdict_mode(
        self,
        test_case: EvalTestCase,
        criteria: List[str],
        kept: List[str],
        skipped: List[Dict[str, Any]],
        used_names: List[str],
        values: Dict[str, Any],
        truncation_warning: Optional[str],
    ) -> Dict[str, Any]:
        few_shot_block, few_shot_log = self._render_few_shot("verdict")
        runs = await asyncio.gather(
            *[
                self._run_verdicts_once(kept, used_names, values, few_shot_block)
                for _ in range(self.n_runs)
            ]
        )
        # runs: List[(verdicts, cost)]
        total_cost = sum(r[1] for r in runs)
        run_verdicts = [r[0] for r in runs]

        per_run_detail: Optional[List[Dict[str, Any]]] = None

        # not_applicable verdicts are dropped from the aggregation (they carry no
        # weight); weights_by_index maps a scored criterion index to its weight.
        na_indices: List[int] = []
        weights_by_index: Dict[int, float] = {}

        if self.n_runs == 1:
            final_verdicts = run_verdicts[0]
            for i, v in enumerate(final_verdicts):
                lbl = v.get("verdict", "none")
                if lbl == _NOT_APPLICABLE:
                    na_indices.append(i)
                    continue
                weights_by_index[i] = VERDICT_WEIGHTS.get(lbl, 0.0)
        else:
            final_verdicts = []
            per_run_detail = []
            for i in range(len(kept)):
                labels = [rv[i].get("verdict", "none") for rv in run_verdicts]
                counter = Counter(labels)
                mode_label, mode_votes = counter.most_common(1)[0]

                if mode_label == _NOT_APPLICABLE:
                    na_indices.append(i)
                    final_verdicts.append(
                        {
                            "verdict": _NOT_APPLICABLE,
                            "reason": f"Majority not_applicable {mode_votes}/{self.n_runs} across runs.",
                        }
                    )
                    per_run_detail.append(
                        {
                            "criterion_index": i,
                            "per_run_verdicts": labels,
                            "aggregated_verdict": _NOT_APPLICABLE,
                            "aggregated_weight": None,
                        }
                    )
                    continue

                scored_labels = [l for l in labels if l != _NOT_APPLICABLE]
                if self.aggregation == "majority":
                    weight = VERDICT_WEIGHTS.get(mode_label, 0.0)
                    final_verdicts.append(
                        {
                            "verdict": mode_label,
                            "reason": f"Majority vote {mode_votes}/{self.n_runs} across runs.",
                        }
                    )
                else:
                    weights = [VERDICT_WEIGHTS.get(l, 0.0) for l in scored_labels] or [0.0]
                    if self.aggregation == "median":
                        weight = float(_median(weights))
                    else:  # mean
                        weight = float(_mean(weights))
                    final_verdicts.append(
                        {
                            "verdict": mode_label,
                            "reason": (
                                f"{self.aggregation.capitalize()} weight={round(weight, 3)} "
                                f"across {self.n_runs} runs (mode verdict shown)."
                            ),
                        }
                    )
                weights_by_index[i] = weight
                per_run_detail.append(
                    {
                        "criterion_index": i,
                        "per_run_verdicts": labels,
                        "aggregated_verdict": final_verdicts[-1]["verdict"],
                        "aggregated_weight": round(weight, 4),
                    }
                )

        not_applicable_criteria = [
            {"criterion": kept[i], "reason": final_verdicts[i].get("reason", "")}
            for i in na_indices
        ]
        aggregated_weights = [weights_by_index[i] for i in sorted(weights_by_index)]

        if not aggregated_weights:
            # Every criterion was not_applicable → no verdict for this row.
            return self._all_not_applicable_result(
                test_case,
                criteria,
                kept,
                skipped,
                used_names,
                values,
                final_verdicts,
                not_applicable_criteria,
                few_shot_log,
                total_cost,
                truncation_warning,
            )

        final_score = round(score_agg(aggregated_weights, temperature=self.temperature), 4)
        success = final_score >= self.threshold

        scored_verdicts = [v for i, v in enumerate(final_verdicts) if i not in set(na_indices)]
        positive = [v for v in scored_verdicts if v.get("verdict") in ("fully", "mostly")]
        negative = [v for v in scored_verdicts if v.get("verdict") in ("none", "minor", "partial")]
        total = len(scored_verdicts)
        if total > 0 and len(positive) >= total * 0.7:
            summary = (
                f"Strong performance: {len(positive)}/{total} criteria fully or mostly satisfied."
            )
        elif total > 0 and len(negative) >= total * 0.7:
            summary = f"Weak performance: {len(negative)}/{total} criteria not satisfied or minimally addressed."
        else:
            summary = f"Mixed performance: {len(positive)}/{total} criteria satisfied, with room for improvement."

        evaluation_log: Dict[str, Any] = {
            "strategy": "verdict",
            "n_runs": self.n_runs,
            "aggregation": self.aggregation if self.n_runs > 1 else None,
            "input_question": test_case.input,
            "actual_output": test_case.actual_output,
            "evaluation_criteria_template": list(self.evaluation_criteria),
            "comment_evaluation_criteria_template": "Original user-provided criteria with {{placeholders}} preserved.",
            "kept_criteria": kept,
            "comment_kept_criteria": f"Criteria actually scored ({len(kept)} of {len(criteria)} after filtering).",
            "skipped_criteria": skipped,
            "comment_skipped_criteria": "Criteria excluded from judging, with the reason for each.",
            "data_used": {name: values[name] for name in used_names},
            "comment_data_used": "Subset of resolved values that appeared in at least one kept criterion (the DATA block shown to the judge).",
            "verdicts": final_verdicts,
            "comment_verdicts": "Per-criterion verdicts (aggregated across runs if n_runs>1).",
            "verdict_weights": {i: weights_by_index[i] for i in sorted(weights_by_index)},
            "final_score": final_score,
            "comment_final_score": f"TCVA aggregation of verdict weights (temperature={self.temperature}).",
            "threshold": self.threshold,
            "temperature": self.temperature,
            "success": success,
            "summary": summary,
        }
        if per_run_detail is not None:
            evaluation_log["per_run_detail"] = per_run_detail
            evaluation_log["comment_per_run_detail"] = (
                f"Individual verdicts from each of the {self.n_runs} runs before "
                f"{self.aggregation} aggregation."
            )
        if not_applicable_criteria:
            evaluation_log["not_applicable_criteria"] = not_applicable_criteria
            evaluation_log["comment_not_applicable_criteria"] = (
                "Criteria whose condition did not apply to this input; dropped from the score."
            )
        if self.few_shot_examples:
            evaluation_log["few_shot"] = few_shot_log
        if truncation_warning:
            evaluation_log["truncation_warning"] = truncation_warning

        result = {
            "name": self.name,
            "score": final_score,
            "success": success,
            "reason": summary,
            "evaluation_cost": round(total_cost, 6),
            "evaluation_log": evaluation_log,
        }
        self.print_result(result)
        return result

    # ==================== STRATEGY: DIRECT ====================

    async def _evaluate_direct_mode(
        self,
        test_case: EvalTestCase,
        criteria: List[str],
        kept: List[str],
        skipped: List[Dict[str, Any]],
        used_names: List[str],
        values: Dict[str, Any],
        truncation_warning: Optional[str],
    ) -> Dict[str, Any]:
        few_shot_block, few_shot_log = self._render_few_shot("direct")
        runs = await asyncio.gather(
            *[
                self._run_direct_once(kept, used_names, values, few_shot_block)
                for _ in range(self.n_runs)
            ]
        )
        # runs: List[(raw_score_0_10, reason, cost)]
        raw_scores = [r[0] for r in runs]
        reasons = [r[1] for r in runs]
        total_cost = sum(r[2] for r in runs)

        if self.n_runs == 1:
            aggregated_raw: float = float(raw_scores[0])
            chosen_reason = reasons[0]
        elif self.aggregation == "majority":
            winning_raw = Counter(raw_scores).most_common(1)[0][0]
            aggregated_raw = float(winning_raw)
            chosen_reason = next(
                (reasons[i] for i, s in enumerate(raw_scores) if s == winning_raw),
                reasons[0],
            )
        elif self.aggregation == "median":
            aggregated_raw = float(_median(raw_scores))
            chosen_reason = f"Median of {self.n_runs} runs: scores={raw_scores}."
        else:  # mean
            aggregated_raw = float(_mean(raw_scores))
            chosen_reason = f"Mean of {self.n_runs} runs: scores={raw_scores}."

        final_score = round(aggregated_raw / 10.0, 4)
        success = final_score >= self.threshold

        if final_score >= 0.7:
            summary = f"Strong direct judge score: {aggregated_raw}/10 (normalized {final_score})."
        elif final_score >= 0.4:
            summary = f"Mixed direct judge score: {aggregated_raw}/10 (normalized {final_score})."
        else:
            summary = f"Weak direct judge score: {aggregated_raw}/10 (normalized {final_score})."

        evaluation_log: Dict[str, Any] = {
            "strategy": "direct",
            "n_runs": self.n_runs,
            "aggregation": self.aggregation if self.n_runs > 1 else None,
            "input_question": test_case.input,
            "actual_output": test_case.actual_output,
            "evaluation_criteria_template": list(self.evaluation_criteria),
            "comment_evaluation_criteria_template": "Original user-provided criteria with {{placeholders}} preserved.",
            "kept_criteria": kept,
            "comment_kept_criteria": f"Criteria actually included in the direct-score prompt ({len(kept)} of {len(criteria)}).",
            "skipped_criteria": skipped,
            "data_used": {name: values[name] for name in used_names},
            "raw_scores_0_10": raw_scores,
            "comment_raw_scores_0_10": (
                "Judge's integer scores 0-10 from each run."
                if self.n_runs > 1
                else "Judge's integer score 0-10 for this test case."
            ),
            "aggregated_raw_score_0_10": aggregated_raw,
            "final_score": final_score,
            "comment_final_score": "Raw 0-10 score normalized to 0.0-1.0 (raw / 10).",
            "reasons": reasons,
            "chosen_reason": chosen_reason,
            "threshold": self.threshold,
            "success": success,
            "summary": summary,
        }
        if self.few_shot_examples:
            evaluation_log["few_shot"] = few_shot_log
        if truncation_warning:
            evaluation_log["truncation_warning"] = truncation_warning

        result = {
            "name": self.name,
            "score": final_score,
            "success": success,
            "reason": chosen_reason or summary,
            "evaluation_cost": round(total_cost, 6),
            "evaluation_log": evaluation_log,
        }
        self.print_result(result)
        return result

    @property
    def name(self):
        return f"Custom: {self.custom_name}"
