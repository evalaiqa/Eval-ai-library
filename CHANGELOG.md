# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0] - 2026-10-21

Custom Eval calibration — library-side core (Phase 1). Additions only; existing
`CustomEvalMetric` and every other metric behave exactly as before. With no
few-shot examples the CustomEval judge prompt and result are byte-for-byte
identical to 0.7.30.

### Added

- **Few-shot examples for the judge.** `CustomEvalMetric` accepts
  `few_shot_examples` (list of `FewShotExample` or dicts) and `few_shot_max_chars`.
  Examples are shown to the judge as calibration references (how strictly the
  team grades); they are never copied, never scored, and `{{placeholders}}` in
  them are not substituted. At most 20 examples (`ValueError` otherwise), each
  field capped at 2000 chars, the whole block capped at 12000 chars (trailing
  examples dropped to fit). When examples are present, `evaluation_log` gains a
  `few_shot` entry `{count, used, source_case_ids, truncated}`. The mechanism
  lives in `eval_lib.few_shot` and is wired into the base `MetricPattern`, so it
  can be reused by other metrics later. `FewShotExample` is exported from
  `eval_lib` and `eval_lib.calibration`.
- **`eval_lib.calibration` package.**
  - Models: `CustomEvalConfig` (`to_metric`, `from_params`), `HumanLabel`,
    `LabeledCase`, `CalibrationSet` (stable, stratified `split()`), `Split`,
    `Proposal`. All pydantic v2, JSON round-trippable, with `schema_version`.
  - Statistics (no LLM, no scipy): `agreement`, `inter_annotator`,
    `threshold_sweep`, `criterion_attribution`.
  - Candidates: `apply_proposals`, `diff_configs`, `evaluate_config` (re-score
    saved cases through the judge only — never the app), `compare` (deltas +
    exact McNemar + bootstrap CI).
  - AI analyst `CalibrationAnalyst`: reads every judge-vs-human disagreement,
    diagnoses each case (map, batched), consolidates diagnoses into `Finding`s
    (reduce, hierarchical for many groups), and proposes config edits (`Proposal`).
    The LLM only diagnoses and phrases — all counts are computed in code, the
    threshold is proposed by `threshold_sweep`, and every proposal is validated
    (unknown/absent placeholder, 50-criteria cap, duplicate → `rejected`). Holdout
    is never passed to it. Returns `AnalysisResult`.
  - Optimizer `CalibrationOptimizer` (EV-37): runs the analyst on train, builds
    candidate configs, scores them on train, and evaluates the winner and the
    current version on holdout exactly once; returns a draft `proposed_config`,
    `diff`, `holdout_comparison`, and a `verdict` of `improved` /
    `not_significant` / `worse`. Applies nothing itself.
  - Budget & estimates: `Budget` (asyncio-safe), `BudgetExceeded`,
    `InsufficientLabels`, and `estimate_evaluation` / `estimate_analysis` /
    `estimate_optimization` / `CostEstimate` (pre-flight, no LLM calls).
- First `CHANGELOG.md` for the project.

Example:

```python
from eval_lib import FewShotExample, CustomEvalMetric
from eval_lib.calibration import (
    CustomEvalConfig, HumanLabel, LabeledCase, CalibrationSet,
    agreement, threshold_sweep, evaluate_config, compare, Budget,
)

config = CustomEvalConfig(
    name="helpfulness",
    evaluation_criteria=["{{actual_output}} directly answers {{input}}"],
    threshold=0.6,
    few_shot_examples=[
        FewShotExample(input="How do I reset my password?",
                       actual_output="Click 'Forgot password'.",
                       verdict="pass", reason="direct and correct"),
    ],
)
report = agreement(labeled_cases)          # judge-vs-human agreement
sweep = threshold_sweep(labeled_cases)     # best pass/fail threshold
run = await evaluate_config(config, labeled_cases, "openai:gpt-4o-mini",
                            budget=Budget(max_llm_calls=500))
```

[Unreleased]: https://github.com/meshkovQA/Eval-ai-library/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.8.0
