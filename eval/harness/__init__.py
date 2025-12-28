"""Evaluation harness package."""

from .results import TaskResult, EvalSummary, ResultsStore
from .repo_manager import RepoManager
from .evaluator import Evaluator
from .runner import EvaluationRunner
from .metrics import (
    # Enums
    Phase,
    FailureMode,
    # Metric classes
    TokenMetrics,
    ReasoningMetrics,
    ToolArgumentMetrics,
    PhaseMetrics,
    ExplorationMetrics,
    TrajectoryMetrics,
    ConvergenceMetrics,
    ErrorRecoveryMetrics,
    ToolUsageMetrics,
    PatchQualityMetrics,
    FailureAnalysis,
    DebugMetrics,
    # Functions
    compute_debug_metrics,
    format_debug_report,
)

__all__ = [
    # Results
    "TaskResult",
    "EvalSummary", 
    "ResultsStore",
    # Managers
    "RepoManager",
    "Evaluator",
    "EvaluationRunner",
    # Enums
    "Phase",
    "FailureMode",
    # Metrics
    "TokenMetrics",
    "ReasoningMetrics",
    "ToolArgumentMetrics",
    "PhaseMetrics",
    "ExplorationMetrics",
    "TrajectoryMetrics",
    "ConvergenceMetrics",
    "ErrorRecoveryMetrics",
    "ToolUsageMetrics",
    "PatchQualityMetrics",
    "FailureAnalysis",
    "DebugMetrics",
    # Functions
    "compute_debug_metrics",
    "format_debug_report",
]