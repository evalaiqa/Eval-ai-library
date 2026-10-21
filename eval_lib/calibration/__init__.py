# eval_lib/calibration/__init__.py
"""
Custom Eval calibration — core library API (release 0.8.0).

Calibrate a ``CustomEvalMetric`` against human labels: few-shot examples, an
agreement/threshold statistics layer, an AI analyst that diagnoses judge-vs-human
disagreements and proposes criterion edits, config re-evaluation / comparison on
a held-out split, and an optimizer that drafts an improved version. Everything
here is additive and does not change the behaviour of existing metrics.
"""

from eval_lib.calibration.analyst import (
    AnalysisResult,
    CalibrationAnalyst,
    Finding,
)
from eval_lib.calibration.budget import (
    Budget,
    BudgetExceeded,
    InsufficientLabels,
)
from eval_lib.calibration.candidates import (
    CaseEvalResult,
    Comparison,
    ConfigDiff,
    EvaluationRun,
    apply_proposals,
    compare,
    diff_configs,
    evaluate_config,
)
from eval_lib.calibration.estimate import (
    CostEstimate,
    estimate_analysis,
    estimate_evaluation,
    estimate_optimization,
)
from eval_lib.calibration.models import (
    CalibrationSet,
    CustomEvalConfig,
    HumanLabel,
    LabeledCase,
    Proposal,
    Split,
)
from eval_lib.calibration.optimizer import (
    CalibrationOptimizer,
    CandidateResult,
    OptimizationResult,
)
from eval_lib.calibration.stats import (
    AgreementReport,
    Confusion,
    ConfusionCell,
    CriterionStats,
    InterAnnotatorReport,
    ThresholdPoint,
    ThresholdReport,
    agreement,
    criterion_attribution,
    inter_annotator,
    threshold_sweep,
)
from eval_lib.few_shot import FewShotExample

__all__ = [
    # Few-shot
    "FewShotExample",
    # Models
    "CustomEvalConfig",
    "HumanLabel",
    "LabeledCase",
    "CalibrationSet",
    "Split",
    "Proposal",
    # Stats
    "agreement",
    "inter_annotator",
    "threshold_sweep",
    "criterion_attribution",
    "AgreementReport",
    "InterAnnotatorReport",
    "ThresholdReport",
    "ThresholdPoint",
    "CriterionStats",
    "Confusion",
    "ConfusionCell",
    # Candidates
    "apply_proposals",
    "diff_configs",
    "evaluate_config",
    "compare",
    "ConfigDiff",
    "EvaluationRun",
    "CaseEvalResult",
    "Comparison",
    # Analyst
    "CalibrationAnalyst",
    "AnalysisResult",
    "Finding",
    # Optimizer
    "CalibrationOptimizer",
    "OptimizationResult",
    "CandidateResult",
    # Budget
    "Budget",
    "BudgetExceeded",
    "InsufficientLabels",
    # Estimates
    "estimate_evaluation",
    "estimate_analysis",
    "estimate_optimization",
    "CostEstimate",
]
