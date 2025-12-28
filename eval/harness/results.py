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
