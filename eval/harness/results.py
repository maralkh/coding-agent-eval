"""Results storage and aggregation."""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class TestOutcome:
    """Outcome of a single test."""
    name: str
    passed: bool
    error: str | None = None


@dataclass
class TaskResult:
    """Result of evaluating an agent on a single task."""

    task_id: str
    
    # Core metrics
    resolved: bool = False          # fail_to_pass tests now pass
    no_regression: bool = True      # pass_to_pass tests still pass
    
    # Test details
    fail_to_pass_results: list[dict] = field(default_factory=list)
    pass_to_pass_results: list[dict] = field(default_factory=list)
    
    # Counts
    fail_to_pass_total: int = 0
    fail_to_pass_passed: int = 0
    pass_to_pass_total: int = 0
    pass_to_pass_passed: int = 0
    
    # Quality metrics
    diff_size: int = 0              # lines changed
    files_changed: list[str] = field(default_factory=list)
    
    # Efficiency
    steps: int = 0
    duration: float = 0.0
    
    # Agent output
    agent_patch: str = ""
    explanation: str = ""
    
    # Status
    error: str | None = None
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TaskResult":
        return cls(**data)


@dataclass
class EvalSummary:
    """Aggregated evaluation metrics."""

    total_tasks: int = 0
    resolved_count: int = 0
    resolve_rate: float = 0.0
    
    no_regression_count: int = 0
    no_regression_rate: float = 0.0
    
    # By difficulty
    by_difficulty: dict = field(default_factory=dict)
    
    # Efficiency
    avg_steps: float = 0.0
    avg_duration: float = 0.0
    
    # Quality
    avg_diff_size: float = 0.0
    
    # Errors
    error_count: int = 0


class ResultsStore:
    """Stores and loads evaluation results."""

    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.output_dir / "results.jsonl"

    def save_result(self, result: TaskResult) -> None:
        """Append a result to the JSONL file."""
        with open(self.results_file, "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")

    def load_results(self) -> list[TaskResult]:
        """Load all results from the JSONL file."""
        if not self.results_file.exists():
            return []
        
        results = []
        with open(self.results_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    results.append(TaskResult.from_dict(data))
        return results

    def get_completed_task_ids(self) -> set[str]:
        """Get set of task IDs that have been completed."""
        results = self.load_results()
        return {r.task_id for r in results}

    def get_summary(self) -> EvalSummary:
        """Compute aggregate metrics from results."""
        results = self.load_results()
        
        if not results:
            return EvalSummary()
        
        total = len(results)
        resolved = sum(1 for r in results if r.resolved)
        no_regression = sum(1 for r in results if r.no_regression)
        errors = sum(1 for r in results if r.error)
        
        # By difficulty (if we can infer it from task_id patterns)
        # This is a simplified version - in practice you'd join with task metadata
        
        # Efficiency
        steps = [r.steps for r in results if r.steps > 0]
        durations = [r.duration for r in results if r.duration > 0]
        diff_sizes = [r.diff_size for r in results]
        
        return EvalSummary(
            total_tasks=total,
            resolved_count=resolved,
            resolve_rate=resolved / total if total > 0 else 0.0,
            no_regression_count=no_regression,
            no_regression_rate=no_regression / total if total > 0 else 0.0,
            avg_steps=sum(steps) / len(steps) if steps else 0.0,
            avg_duration=sum(durations) / len(durations) if durations else 0.0,
            avg_diff_size=sum(diff_sizes) / len(diff_sizes) if diff_sizes else 0.0,
            error_count=errors,
        )

    def generate_report(self) -> str:
        """Generate a markdown report."""
        results = self.load_results()
        summary = self.get_summary()
        
        lines = [
            "# Evaluation Report",
            "",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Tasks | {summary.total_tasks} |",
            f"| Resolved | {summary.resolved_count} ({summary.resolve_rate:.1%}) |",
            f"| No Regression | {summary.no_regression_count} ({summary.no_regression_rate:.1%}) |",
            f"| Errors | {summary.error_count} |",
            "",
            "## Efficiency",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Avg Steps | {summary.avg_steps:.1f} |",
            f"| Avg Duration | {summary.avg_duration:.1f}s |",
            f"| Avg Diff Size | {summary.avg_diff_size:.1f} lines |",
            "",
            "## Per-Task Results",
            "",
            "| Task ID | Resolved | Regression | Steps | Duration |",
            "|---------|----------|------------|-------|----------|",
        ]
        
        for r in results:
            resolved = "✓" if r.resolved else "✗"
            regression = "✓" if r.no_regression else "✗"
            error = f" ({r.error})" if r.error else ""
            lines.append(
                f"| {r.task_id} | {resolved} | {regression} | {r.steps} | {r.duration:.1f}s{error} |"
            )
        
        return "\n".join(lines)

    def save_report(self, path: Path | str | None = None) -> Path:
        """Save markdown report to file."""
        if path is None:
            path = self.output_dir / "report.md"
        path = Path(path)
        
        report = self.generate_report()
        path.write_text(report)
        return path


# =============================================================================
# BENCHMARK RESULT LOADING
# =============================================================================

@dataclass 
class BenchmarkTaskResult:
    """Result of running a single task with a model (from benchmark.py)."""
    task_id: str
    model: str
    provider: str
    
    # Outcome
    resolved: bool = False
    submitted: bool = False
    
    # Efficiency
    steps: int = 0
    duration: float = 0.0
    
    # Quality metrics
    similarity_score: float = 0.0
    reasoning_score: float = 0.0
    exploration_efficiency: float = 0.0
    trajectory_efficiency: float = 0.0
    
    # Failure info
    primary_failure_mode: str = ""
    failure_reasons: list = field(default_factory=list)
    
    # Error
    error: str = ""
    
    def to_metrics_dict(self) -> dict:
        """Convert to dict suitable for scoring functions."""
        return {
            "resolved": float(self.resolved),
            "submitted": float(self.submitted),
            "steps": self.steps,
            "duration": self.duration,
            "similarity_score": self.similarity_score,
            "reasoning_score": self.reasoning_score,
            "exploration_efficiency": self.exploration_efficiency,
            "trajectory_efficiency": self.trajectory_efficiency,
        }


@dataclass
class ModelSummary:
    """Aggregated results for a model across all tasks."""
    model: str
    provider: str
    
    # Counts
    total_tasks: int = 0
    resolved_count: int = 0
    submitted_count: int = 0
    error_count: int = 0
    
    # Rates
    resolve_rate: float = 0.0
    submit_rate: float = 0.0
    
    # Averages
    avg_steps: float = 0.0
    avg_duration: float = 0.0
    avg_similarity: float = 0.0
    avg_reasoning_score: float = 0.0
    avg_exploration_efficiency: float = 0.0
    avg_trajectory_efficiency: float = 0.0
    
    # Failure breakdown
    failure_modes: dict = field(default_factory=dict)
    
    # Per-task results
    task_results: list = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""
    timestamp: str
    config: dict
    models: list  # List of ModelSummary
    tasks: list   # List of task IDs
    
    # Cross-model comparisons
    best_model_by_resolve_rate: str = ""
    best_model_by_similarity: str = ""


def load_benchmark_task_results(path: Path | str) -> list[BenchmarkTaskResult]:
    """
    Load task results from JSONL file (all_results.jsonl).
    
    Args:
        path: Path to all_results.jsonl
        
    Returns:
        List of BenchmarkTaskResult objects
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            # Handle both old and new formats
            if "failure_reasons" not in data:
                data["failure_reasons"] = []
            results.append(BenchmarkTaskResult(**data))
    
    return results


def load_benchmark(path: Path | str) -> BenchmarkResult:
    """
    Load complete benchmark result from JSON file.
    
    Args:
        path: Path to benchmark_result.json
        
    Returns:
        BenchmarkResult object
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    
    with open(path) as f:
        data = json.load(f)
    
    # Reconstruct ModelSummary objects
    models = []
    for m in data.get("models", []):
        # Reconstruct task_results
        task_results = []
        for tr in m.get("task_results", []):
            if "failure_reasons" not in tr:
                tr["failure_reasons"] = []
            task_results.append(BenchmarkTaskResult(**tr))
        
        summary = ModelSummary(
            model=m["model"],
            provider=m["provider"],
            total_tasks=m.get("total_tasks", 0),
            resolved_count=m.get("resolved_count", 0),
            submitted_count=m.get("submitted_count", 0),
            error_count=m.get("error_count", 0),
            resolve_rate=m.get("resolve_rate", 0.0),
            submit_rate=m.get("submit_rate", 0.0),
            avg_steps=m.get("avg_steps", 0.0),
            avg_duration=m.get("avg_duration", 0.0),
            avg_similarity=m.get("avg_similarity", 0.0),
            avg_reasoning_score=m.get("avg_reasoning_score", 0.0),
            avg_exploration_efficiency=m.get("avg_exploration_efficiency", 0.0),
            avg_trajectory_efficiency=m.get("avg_trajectory_efficiency", 0.0),
            failure_modes=m.get("failure_modes", {}),
            task_results=task_results,
        )
        models.append(summary)
    
    return BenchmarkResult(
        timestamp=data.get("timestamp", ""),
        config=data.get("config", {}),
        models=models,
        tasks=data.get("tasks", []),
        best_model_by_resolve_rate=data.get("best_model_by_resolve_rate", ""),
        best_model_by_similarity=data.get("best_model_by_similarity", ""),
    )


def load_benchmark_dir(results_dir: Path | str) -> BenchmarkResult:
    """
    Load results from a benchmark output directory.
    
    Args:
        results_dir: Path to results directory (e.g., results/benchmark/)
        
    Returns:
        BenchmarkResult object
    """
    results_dir = Path(results_dir)
    
    # Try to load benchmark_result.json first
    benchmark_file = results_dir / "benchmark_result.json"
    if benchmark_file.exists():
        return load_benchmark(benchmark_file)
    
    # Fall back to reconstructing from all_results.jsonl
    results_file = results_dir / "all_results.jsonl"
    if results_file.exists():
        task_results = load_benchmark_task_results(results_file)
        return _reconstruct_benchmark(task_results)
    
    raise FileNotFoundError(f"No results found in {results_dir}")


def _reconstruct_benchmark(task_results: list[BenchmarkTaskResult]) -> BenchmarkResult:
    """Reconstruct BenchmarkResult from individual task results."""
    from collections import defaultdict
    import numpy as np
    
    # Group by model
    by_model = defaultdict(list)
    for tr in task_results:
        by_model[(tr.provider, tr.model)].append(tr)
    
    models = []
    for (provider, model), results in by_model.items():
        summary = ModelSummary(
            model=model,
            provider=provider,
            total_tasks=len(results),
            resolved_count=sum(1 for r in results if r.resolved),
            submitted_count=sum(1 for r in results if r.submitted),
            error_count=sum(1 for r in results if r.error),
            task_results=results,
        )
        
        # Compute rates
        if summary.total_tasks > 0:
            summary.resolve_rate = summary.resolved_count / summary.total_tasks
            summary.submit_rate = summary.submitted_count / summary.total_tasks
        
        # Compute averages (excluding errors)
        valid_results = [r for r in results if not r.error]
        if valid_results:
            summary.avg_steps = float(np.mean([r.steps for r in valid_results]))
            summary.avg_duration = float(np.mean([r.duration for r in valid_results]))
            summary.avg_similarity = float(np.mean([r.similarity_score for r in valid_results]))
            summary.avg_reasoning_score = float(np.mean([r.reasoning_score for r in valid_results]))
            summary.avg_exploration_efficiency = float(np.mean([r.exploration_efficiency for r in valid_results]))
            summary.avg_trajectory_efficiency = float(np.mean([r.trajectory_efficiency for r in valid_results]))
        
        # Count failure modes
        failure_modes = defaultdict(int)
        for r in results:
            if r.primary_failure_mode:
                failure_modes[r.primary_failure_mode] += 1
        summary.failure_modes = dict(failure_modes)
        
        models.append(summary)
    
    # Find best models
    best_resolve = max(models, key=lambda m: m.resolve_rate) if models else None
    best_similarity = max(models, key=lambda m: m.avg_similarity) if models else None
    
    return BenchmarkResult(
        timestamp=datetime.now().isoformat(),
        config={},
        models=models,
        tasks=list(set(tr.task_id for tr in task_results)),
        best_model_by_resolve_rate=best_resolve.model if best_resolve else "",
        best_model_by_similarity=best_similarity.model if best_similarity else "",
    )


class BenchmarkAnalyzer:
    """
    Analyze and compare benchmark results.
    
    Usage:
        analyzer = BenchmarkAnalyzer("results/benchmark/")
        analyzer.summary()
        analyzer.compare_models()
        analyzer.compute_unified_scores()
    """
    
    def __init__(self, results_path: Path | str):
        """
        Load results from path.
        
        Args:
            results_path: Path to benchmark_result.json, all_results.jsonl, or directory
        """
        path = Path(results_path)
        
        if path.is_dir():
            self.benchmark = load_benchmark_dir(path)
        elif path.suffix == ".json":
            self.benchmark = load_benchmark(path)
        elif path.suffix == ".jsonl":
            task_results = load_benchmark_task_results(path)
            self.benchmark = _reconstruct_benchmark(task_results)
        else:
            raise ValueError(f"Unknown file type: {path}")
    
    def summary(self) -> str:
        """Print and return summary of results."""
        lines = [
            "=" * 70,
            "BENCHMARK SUMMARY",
            "=" * 70,
            f"Timestamp: {self.benchmark.timestamp}",
            f"Tasks: {len(self.benchmark.tasks)}",
            f"Models: {len(self.benchmark.models)}",
            "",
        ]
        
        # Model table
        lines.append(f"{'Model':<35} {'Provider':<12} {'Resolved':<12} {'Similarity':<12} {'Steps':<8}")
        lines.append("-" * 70)
        
        for m in sorted(self.benchmark.models, key=lambda x: -x.resolve_rate):
            resolved = f"{m.resolved_count}/{m.total_tasks} ({m.resolve_rate:.0%})"
            similarity = f"{m.avg_similarity:.1%}"
            lines.append(f"{m.model:<35} {m.provider:<12} {resolved:<12} {similarity:<12} {m.avg_steps:<8.1f}")
        
        lines.append("-" * 70)
        lines.append(f"Best by resolve rate: {self.benchmark.best_model_by_resolve_rate}")
        lines.append(f"Best by similarity: {self.benchmark.best_model_by_similarity}")
        lines.append("")
        
        result = "\n".join(lines)
        print(result)
        return result
    
    def compare_models(self, metric: str = "resolve_rate") -> list[tuple[str, float]]:
        """
        Compare models by a specific metric.
        
        Args:
            metric: Metric name (resolve_rate, avg_similarity, avg_steps, etc.)
            
        Returns:
            List of (model_name, value) sorted by value descending
        """
        results = []
        for m in self.benchmark.models:
            value = getattr(m, metric, 0)
            results.append((m.model, value))
        
        # Sort (higher is better for most metrics, lower for steps)
        reverse = metric not in ["avg_steps", "avg_duration"]
        results.sort(key=lambda x: x[1], reverse=reverse)
        
        return results
    
    def get_task_results(self, model: str | None = None) -> list[BenchmarkTaskResult]:
        """
        Get all task results, optionally filtered by model.
        
        Args:
            model: Model name to filter by (optional)
            
        Returns:
            List of BenchmarkTaskResult objects
        """
        results = []
        for m in self.benchmark.models:
            if model is None or m.model == model:
                results.extend(m.task_results)
        return results
    
    def get_metrics_dicts(self, model: str | None = None) -> list[dict]:
        """
        Get metrics as dicts suitable for scoring functions.
        
        Args:
            model: Model name to filter by (optional)
            
        Returns:
            List of metrics dicts
        """
        return [tr.to_metrics_dict() for tr in self.get_task_results(model)]
    
    def compute_unified_scores(
        self,
        method: str = "hierarchical",
        model: str | None = None,
    ) -> list[tuple[str, float]]:
        """
        Compute unified scores for all task results.
        
        Args:
            method: Scoring method (weighted, geometric, hierarchical, percentile, topsis, pareto)
            model: Model name to filter by (optional)
            
        Returns:
            List of (task_id, score) tuples
        """
        try:
            from eval.harness.scoring import UnifiedScorer
        except ImportError:
            raise ImportError("Scoring module not available")
        
        results = self.get_task_results(model)
        metrics_list = [r.to_metrics_dict() for r in results]
        
        # For methods requiring reference population, use all results
        if method in ["percentile", "topsis", "pareto", "pca"]:
            all_metrics = self.get_metrics_dicts()
            scorer = UnifiedScorer(method=method, reference_population=all_metrics)
        else:
            scorer = UnifiedScorer(method=method)
        
        scores = []
        for result, metrics in zip(results, metrics_list):
            score = scorer.score(metrics)
            scores.append((result.task_id, score))
        
        return scores
    
    def compute_elo_ratings(self) -> dict[str, float]:
        """
        Compute Elo ratings for all models.
        
        Returns:
            Dict mapping model name to Elo rating
        """
        try:
            from eval.harness.scoring import compute_elo_ratings
        except ImportError:
            raise ImportError("Scoring module not available")
        
        # Build task_results structure for Elo
        task_results = {}
        for m in self.benchmark.models:
            task_results[m.model] = {}
            for tr in m.task_results:
                task_results[m.model][tr.task_id] = tr.to_metrics_dict()
        
        return compute_elo_ratings(task_results)
    
    def failure_analysis(self, model: str | None = None) -> dict[str, int]:
        """
        Get failure mode breakdown.
        
        Args:
            model: Model name to filter by (optional)
            
        Returns:
            Dict mapping failure mode to count
        """
        from collections import defaultdict
        
        modes = defaultdict(int)
        for m in self.benchmark.models:
            if model is None or m.model == model:
                for mode, count in m.failure_modes.items():
                    modes[mode] += count
        
        return dict(sorted(modes.items(), key=lambda x: -x[1]))
    
    def to_dataframe(self):
        """
        Convert results to pandas DataFrame.
        
        Returns:
            DataFrame with one row per task result
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required: pip install pandas")
        
        rows = []
        for m in self.benchmark.models:
            for tr in m.task_results:
                rows.append(asdict(tr))
        
        return pd.DataFrame(rows)


# Convenience functions

def quick_summary(results_path: Path | str) -> str:
    """Quick summary of benchmark results."""
    analyzer = BenchmarkAnalyzer(results_path)
    return analyzer.summary()


def load_and_score(
    results_path: Path | str,
    method: str = "hierarchical",
) -> list[tuple[str, str, float]]:
    """
    Load results and compute unified scores.
    
    Returns:
        List of (model, task_id, score) tuples
    """
    analyzer = BenchmarkAnalyzer(results_path)
    
    results = []
    for m in analyzer.benchmark.models:
        scores = analyzer.compute_unified_scores(method=method, model=m.model)
        for task_id, score in scores:
            results.append((m.model, task_id, score))
    
    return results
