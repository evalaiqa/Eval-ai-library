# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries before 0.8.0 were reconstructed from the git history, so older ones are
summarised from their commit messages rather than written at release time.

## [Unreleased]

## [0.8.2] - 2026-10-23

Calibration keeps the user's threshold, learns from every label, and runs several
rounds — fixing cases where AI calibration drifted the threshold to meaningless
values or let conditional criteria drag good answers down. Additions only, but the
CustomEval verdict prompt now offers a sixth verdict (see Changed).

### Added
- **`not_applicable` verdict.** `CustomEvalMetric` (verdict strategy) lets the judge
  mark a conditional criterion ("If {{input}} …") `not_applicable` when its condition
  does not hold; such criteria are dropped from the TCVA aggregation (like skipped
  ones) and listed in `evaluation_log.not_applicable_criteria`. A row where every
  criterion is `not_applicable` gets no verdict (`not_applicable_only`, excluded from
  agreement). `direct` strategy is unchanged.
- **`improve(..., keep_threshold=True)`** (default): candidate A (threshold only) is
  not built, candidates are scored at the config's own threshold, and the analyst is
  told to fit the criteria to that fixed threshold (with the verdict weights and TCVA
  spelled out). `keep_threshold=False` restores 0.8.1 threshold tuning.
- **`improve(..., max_rounds=3)`**: each round starts from the previous winner,
  re-runs the analyst on the rows that still disagree, and keeps the best; it stops
  when every comparable row agrees, a round gains under one row, `max_rounds` is hit,
  or the budget runs out. `ImproveResult` gains `rounds` (a `RoundReport` each) and
  `remaining` (case ids still disagreeing, with the judge's reason).
- **`improve(..., rejudge=True)`** (default): the analyst analyses the "before" run
  (the current config re-judging every labelled row) instead of the `judge_result`
  stored with each label, so labels given on older metric versions are still usable.
- `CaseEvalResult` now carries `reason` and `evaluation_log`, so a run result can
  stand in for `LabeledCase.judge_result`.

### Changed
- The CustomEval **verdict judge prompt** now offers `not_applicable` (with an
  instruction to use it only for an unmet conditional criterion). Prompts with no
  conditional criteria are unaffected in practice; `direct` is unchanged.
- The analyst no longer writes row-specific values (order numbers, codes, SKUs,
  prices, dates) into proposed criteria: such a proposal is rejected with reason
  "row-specific value" and reported in `rejected`.

## [0.8.1] - 2026-10-22

Calibration now picks the best of several candidates and tunes the threshold on
the scores of the config it will actually run with — fixing cases where AI
calibration made a metric *worse* because criteria and threshold were changed
together (the threshold had been fitted to the old scores). Additions only.

### Added
- `threshold_sweep_run(run, current, objective="accuracy")` tunes the threshold
  on an `EvaluationRun`'s own `results[].score` / `human_verdict` pairs. It moves
  the threshold only on a strict gain of at least one agreeing row, and the
  `ThresholdReport` now carries `flipped` rows, `margin` to the nearest score
  (< 0.02 ⇒ `unstable`), per-point `agree` / `n`, and a plain `rationale_text`.
- `improve(calibration_set, learn_cases, measure_cases, model, ...)` — small-set
  mode for the 10–30 labelled-row case. Scores the current config once
  ("before"), runs the analyst on `learn_cases`, then builds up to four
  candidates — **A** threshold only (no LLM), **B** criteria edits, **C** B +
  few-shot, **D** everything else — tuning each candidate's threshold on its own
  scores and comparing them on the rows they all share. Rows used as few-shot
  examples are excluded from that candidate's scoring. The winner needs at least
  one more agreeing row than "before" (ties → fewer edits, A before B), else
  `verdict="no_improvement"` and the config is unchanged. Returns `ImproveResult`
  (`before`, `candidates` with per-candidate agreement / threshold explanation /
  `fixed` / `broken`, `winner`, `verdict`, `proposed_config`, `analysis`, cost).
  `improve` applies nothing.

### Changed
- `ThresholdPoint` gains `agree` / `n`; `ThresholdReport` gains `flipped` /
  `margin` / `rationale_text` (empty for the existing `threshold_sweep`, which is
  unchanged).
- `improve` never bundles a threshold tuned on the old scores with criteria
  edits: such a threshold is applied only to the threshold-only candidate, where
  the criteria do not change.

## [0.8.0] - 2026-10-21

Custom Eval calibration — library-side core. Additions only; existing
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

## [0.7.30] - 2026-10-07

### Added
- `AnswerRelevancyMetric` and `ContextualRelevancyMetric`.

### Changed
- Optimized RAG metrics; improved `utils.py` scoring and penalty calculation.
- Fixed `JUDGE_SEED` applied to judge calls for reproducibility.

### Fixed
- Faithfulness / AnswerRelevancy scoring on equivalent answers: the verdict step
  now classifies statements and excludes input-restatements and non-claims
  (offers / CTA); Faithfulness sees the user input; AnswerRelevancy floors
  on-topic entity details at "minor".

## [0.7.28] - 2026-10-06

### Fixed
- Tool correctness metric evaluation for RAG systems and AI agents.

## [0.7.27] - 2026-09-28

### Fixed
- Role adherence metric calculation.

## [0.7.26] - 2026-09-21

### Changed
- Updated the LiteLLM dependency to pick up new providers and improvements.

## [0.7.25] - 2026-09-08

### Added
- Expanded the restricted-refusal metric and registered the new metrics in the
  metric registry.

## [0.7.24] - 2026-09-06

### Added
- Harmful-content detection security metric (Hugging Face classifier).

### Fixed
- Double counting of errors in `tools_error_metric`; accuracy/performance of the
  existing security metrics.

## [0.7.23] - 2026-09-06

### Added
- Security metrics for harmful content, jailbreak, and prompt-injection detection.

### Fixed
- Local model evaluation.

## [0.7.22] - 2026-09-05

### Fixed
- Tracing module import error (`ImportError`) from incorrect imports in the
  `eval_lib` / `eval_lib.tracing` package `__init__` files.

## [0.7.21] - 2026-09-05

### Fixed
- Tracing callbacks not being registered in some environments; added logging for
  debugging.

## [0.7.20] - 2026-09-02

### Added
- Support for new metrics in the bias metric.

### Fixed
- JSON parsing in the toxicity metric.

## [0.7.19] - 2026-08-27

### Changed
- Documentation update and minor fixes.

## [0.7.17] - 2026-08-26

### Added
- Additional custom metrics.

## [0.7.16] - 2026-08-22

### Fixed
- Dataset parser.

## [0.7.15] - 2026-07-10

### Added
- User id and session on traces.

## [0.7.14] - 2026-07-09

### Changed
- Tracing updates.

## [0.7.13] - 2026-05-22

### Security
- Small security fix.

## [0.7.12] - 2026-05-20

### Changed
- Maintenance and version bump.

## [0.7.11] - 2026-05-20

### Fixed
- Bug in the custom evaluation metric.

## [0.7.10] - 2026-05-20

### Added
- Custom field in custom metrics.

## [0.7.9] - 2026-05-19

### Added
- Multi custom-metric support.

## [0.7.5] - 2026-04-17

### Added
- Reliability metrics, `trace_utils`, and framework tracers (merge of the
  agents_evaluation work).

## [0.7.4] - 2026-04-15

### Changed
- Maintenance and version bump.

## [0.7.3] - 2026-04-15

### Changed
- Maintenance and version bump.

## [0.7.2] - 2026-04-14

### Changed
- Version bump.

## [0.7.1] - 2026-04-14

### Added
- Conversational-flow, failure-rate, goal-achievement, and repetitive-pattern
  metrics; LiteLLM integration.

### Changed
- Updated the task-success and security metrics.

## [0.5.4] - 2026-04-01

### Added
- New agent metrics and tracing capabilities.

## [0.5.3] - 2026-03-23

### Added
- JSON schema validation for evaluation results.

## [0.5.2] - 2026-03-10

### Changed
- Maintenance and version bump.

## [0.5.1] - 2026-03-07

### Changed
- Maintenance and version bump.

## [0.4.40] - 2026-01-22

### Changed
- Maintenance and version bump.

## [0.4.37] - 2025-12-10

### Added
- Tracing module.

## [0.4.11] - 2025-11-19

### Changed
- Maintenance and version bump.

## [0.4.6] - 2025-11-14

### Changed
- Maintenance and version bump.

## [0.4.5] - 2025-11-14

### Changed
- Maintenance and version bump.

## [0.4.2] - 2025-11-03

### Added
- Security metrics; `model_pricing` import.

### Changed
- Updated dependencies.

## [0.3.12] - 2025-10-30

### Fixed
- Evaluation logs and CLI issues.

## [0.3.9] - 2025-10-29

### Added
- CLI to start the dashboard; support for custom LLM clients.

## [0.3.0] - 2025-10-23

### Added
- First stable version; temperature parameter recorded in logs.

## [0.2.0] - 2025-10-22

### Added
- Initial public packaging and threshold normalization. Releases 0.2.1–0.2.32
  were same-day packaging and dependency fixes.

## [0.1.0] - 2025-10-22

### Added
- Initial release of the Eval AI Library with core functionality for evaluating
  RAG systems and AI agents.

[Unreleased]: https://github.com/meshkovQA/Eval-ai-library/compare/v0.8.2...HEAD
[0.8.2]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.8.2
[0.8.1]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.8.1
[0.8.0]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.8.0
[0.7.30]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.7.30
[0.7.28]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.7.28
[0.7.27]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.7.27
[0.7.26]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.7.26
[0.7.25]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.7.25
[0.7.24]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.7.24
[0.7.23]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.7.23
[0.7.22]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.7.22
[0.7.21]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.7.21
[0.7.20]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.7.20
[0.7.19]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.7.19
[0.7.17]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.7.17
[0.7.16]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.7.16
[0.7.15]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.7.15
[0.7.14]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.7.14
[0.7.13]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.7.13
[0.7.12]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.7.12
[0.7.11]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.7.11
[0.7.10]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.7.10
[0.7.9]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.7.9
[0.7.5]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.7.5
[0.7.4]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.7.4
[0.7.3]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.7.3
[0.7.2]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.7.2
[0.7.1]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.7.1
[0.5.4]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.5.4
[0.5.3]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.5.3
[0.5.2]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.5.2
[0.5.1]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.5.1
[0.4.40]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.4.40
[0.4.37]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.4.37
[0.4.11]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.4.11
[0.4.6]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.4.6
[0.4.5]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.4.5
[0.4.2]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.4.2
[0.3.12]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.3.12
[0.3.9]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.3.9
[0.3.0]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.3.0
[0.2.0]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.2.0
[0.1.0]: https://github.com/meshkovQA/Eval-ai-library/releases/tag/v0.1.0
