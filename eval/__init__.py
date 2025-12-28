"""Evaluation framework for coding agents."""

from .task import Task, estimate_difficulty
from .collector import TaskCollector
from .github_client import GitHubClient
from .harness import (
    TaskResult,
    EvalSummary,
    ResultsStore,
    RepoManager,
    Evaluator,
    EvaluationRunner,
)

__all__ = [
    # Task
    "Task",
    "estimate_difficulty",
    # Collector
    "TaskCollector",
    "GitHubClient",
    # Harness
    "TaskResult",
    "EvalSummary",
    "ResultsStore",
    "RepoManager",
    "Evaluator",
    "EvaluationRunner",
]
