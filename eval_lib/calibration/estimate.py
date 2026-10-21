# calibration/estimate.py
"""
Pre-flight cost/size estimates — no LLM calls.

Token counts are always returned (chars/4 heuristic over the actual prompt the
judge would see, few-shot included). ``cost_usd`` is filled only when the model
is a known string in the pricing catalog; for a CustomLLMClient or an unknown
model it is ``None`` and the platform converts tokens to credits itself.
"""

import math
from typing import Any, List, Optional

from pydantic import BaseModel

from eval_lib.calibration.models import CalibrationSet, CustomEvalConfig, Split
from eval_lib.few_shot import FEW_SHOT_DEFAULT_MAX_CHARS, render_few_shot_block
from eval_lib.llm_client import CustomLLMClient, LLMDescriptor
from eval_lib.model_catalog import get_cost_per_million

_N_CAUSES = 10  # size of the analyst's fixed cause vocabulary (for group estimates)


class CostEstimate(BaseModel):
    schema_version: int = 1
    llm_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: Optional[float] = None


def _model_name(model: Any) -> Optional[str]:
    if model is None or isinstance(model, CustomLLMClient):
        return None
    try:
        return str(LLMDescriptor.parse(model).model)
    except Exception:
        return str(model)


def _cost(model: Any, input_tokens: int, output_tokens: int) -> Optional[float]:
    name = _model_name(model)
    if not name:
        return None
    price = get_cost_per_million(name)
    if not price:
        return None
    return round(
        input_tokens / 1e6 * float(price["input"]) + output_tokens / 1e6 * float(price["output"]),
        6,
    )


def estimate_evaluation(config: CustomEvalConfig, n_cases: int, model: Any = None) -> CostEstimate:
    """Estimate the cost of re-scoring ``n_cases`` with ``config``'s judge.

    ``llm_calls = n_cases × n_runs``. Input tokens are estimated from the judge
    prompt (criteria + few-shot block + template); output tokens from the
    expected verdict/score payload.
    """
    from eval_lib.metrics.custom_metric.custom_eval import CustomEvalMetric

    block, _ = render_few_shot_block(
        config.few_shot_examples, config.strategy, FEW_SHOT_DEFAULT_MAX_CHARS
    )
    criteria = list(config.evaluation_criteria)
    if config.strategy == "direct":
        prompt = CustomEvalMetric._prompt_direct(criteria, [], {}, block)
        out_chars = 60
    else:
        prompt = CustomEvalMetric._prompt_evaluate(criteria, [], {}, block)
        out_chars = 40 * max(1, len(criteria))

    in_per = max(1, len(prompt) // 4)
    out_per = max(1, out_chars // 4)
    calls = n_cases * config.n_runs
    input_tokens = in_per * calls
    output_tokens = out_per * calls
    return CostEstimate(
        llm_calls=calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=_cost(model, input_tokens, output_tokens),
    )


def estimate_analysis(
    calibration_set: CalibrationSet,
    cases: Optional[List[str]] = None,
    batch_size: int = 25,
    model: Any = None,
) -> CostEstimate:
    """Upper-bound estimate for one analyst run: map + reduce + propose calls.

    ``cases`` is the id list the analyst would see (train); ``None`` = all. The
    selection can only shrink this, so using its size is an upper bound.
    """
    config = calibration_set.config
    n = len(cases) if cases is not None else len(calibration_set.cases)
    batch_size = max(1, batch_size)

    map_calls = max(1, math.ceil(n / batch_size))
    est_groups = min(max(1, n), 2 * len(config.evaluation_criteria) + _N_CAUSES)
    reduce_calls = max(1, math.ceil(est_groups / 8))
    propose_calls = 1
    calls = map_calls + reduce_calls + propose_calls

    criteria_chars = sum(len(c) for c in config.evaluation_criteria)
    context_chars = 600 + criteria_chars  # shared map context per batch
    case_chars = 2500  # compacted per-case block (fields capped at 1500)

    in_chars = (
        map_calls * (context_chars + batch_size * case_chars)
        + reduce_calls * 2000
        + propose_calls * (2000 + criteria_chars)
    )
    out_chars = map_calls * (batch_size * 60) + reduce_calls * 800 + propose_calls * 800
    input_tokens = max(1, in_chars // 4)
    output_tokens = max(1, out_chars // 4)
    return CostEstimate(
        llm_calls=calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=_cost(model, input_tokens, output_tokens),
    )


def estimate_optimization(
    calibration_set: CalibrationSet,
    split: Split,
    max_candidates: int = 6,
    model: Any = None,
) -> CostEstimate:
    """Upper-bound estimate: analysis on train + candidate evals on train + two
    holdout evals."""
    config = calibration_set.config
    n_train = len(split.train)
    n_hold = len(split.holdout)

    parts = [
        estimate_analysis(calibration_set, split.train, model=model),
        # max_candidates candidate re-evaluations over train
        _scaled(estimate_evaluation(config, n_train, model), max(1, max_candidates)),
        # the winner and the current version, once each, over holdout
        _scaled(estimate_evaluation(config, n_hold, model), 2),
    ]
    calls = sum(p.llm_calls for p in parts)
    input_tokens = sum(p.input_tokens for p in parts)
    output_tokens = sum(p.output_tokens for p in parts)
    return CostEstimate(
        llm_calls=calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=_cost(model, input_tokens, output_tokens),
    )


def _scaled(est: CostEstimate, factor: int) -> CostEstimate:
    return CostEstimate(
        llm_calls=est.llm_calls * factor,
        input_tokens=est.input_tokens * factor,
        output_tokens=est.output_tokens * factor,
    )
