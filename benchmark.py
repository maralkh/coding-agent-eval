#!/usr/bin/env python
"""
Benchmark script for comparing multiple models on evaluation tasks.

Usage:
    # Run benchmark with multiple models
    python benchmark.py --tasks eval/tasks/ --models gpt-4o o4-mini claude-sonnet-4-20250514
    
    # Specify providers explicitly
    python benchmark.py --tasks eval/tasks/ \
        --models openai:gpt-4o openai:o4-mini anthropic:claude-sonnet-4-20250514
    
    # Run specific tasks
    python benchmark.py --tasks eval/tasks/task1.json eval/tasks/task2.json --models gpt-4o
    
    # View existing results
    python benchmark.py --results-only -o results/benchmark_20241228/
    
    # Limit tasks for quick testing
    python benchmark.py --tasks eval/tasks/ --models gpt-4o --max-tasks 5
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import statistics


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for a model to benchmark."""
    provider: str
    model: str
    display_name: str = ""
    
    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.model


@dataclass 
class TaskResult:
    """Result of running a single task with a model."""
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
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "config": self.config,
            "models": [asdict(m) for m in self.models],
            "tasks": self.tasks,
            "best_model_by_resolve_rate": self.best_model_by_resolve_rate,
            "best_model_by_similarity": self.best_model_by_similarity,
        }


# =============================================================================
# MODEL PARSING
# =============================================================================

def parse_model_spec(spec: str) -> ModelConfig:
    """
    Parse a model specification string.
    
    Formats:
        - "gpt-4o" -> infer provider
        - "openai:gpt-4o" -> explicit provider
        - "anthropic:claude-sonnet-4-20250514" -> explicit provider
    """
    if ":" in spec:
        provider, model = spec.split(":", 1)
    else:
        model = spec
        # Infer provider from model name
        if model.startswith(("gpt-", "o1", "o3", "o4")):
            provider = "openai"
        elif model.startswith("claude"):
            provider = "anthropic"
        elif model.startswith("llama") or model.startswith("mixtral"):
            provider = "groq"
        else:
            provider = "openai"  # Default
    
    return ModelConfig(provider=provider, model=model)


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

class BenchmarkRunner:
    """Runs benchmarks across multiple models and tasks."""
    
    def __init__(
        self,
        output_dir: str = "results/benchmark",
        max_steps: int = 20,
        timeout: int = 600,
        verbose: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_steps = max_steps
        self.timeout = timeout
        self.verbose = verbose
        
        # Results storage
        self.results: dict[str, dict[str, TaskResult]] = {}  # model -> task_id -> result
    
    def run(
        self,
        tasks: list[Path],
        models: list[ModelConfig],
        skip_existing: bool = True,
    ) -> BenchmarkResult:
        """
        Run benchmark on all tasks with all models.
        
        Args:
            tasks: List of task JSON file paths
            models: List of model configurations
            skip_existing: Skip task/model combinations that already have results
        """
        timestamp = datetime.now().isoformat()
        task_ids = [self._get_task_id(t) for t in tasks]
        
        print("=" * 70)
        print("BENCHMARK RUN")
        print("=" * 70)
        print(f"Tasks: {len(tasks)}")
        print(f"Models: {[m.display_name for m in models]}")
        print(f"Output: {self.output_dir}")
        print(f"Max steps: {self.max_steps}")
        print("=" * 70)
        print()
        
        # Load existing results
        existing = self._load_existing_results()
        
        # Run each model on each task
        total_runs = len(tasks) * len(models)
        completed = 0
        
        for model in models:
            model_key = f"{model.provider}:{model.model}"
            if model_key not in self.results:
                self.results[model_key] = {}
            
            print(f"\n{'='*70}")
            print(f"MODEL: {model.display_name} ({model.provider})")
            print(f"{'='*70}")
            
            for task_path in tasks:
                task_id = self._get_task_id(task_path)
                completed += 1
                
                # Check if already done
                if skip_existing and self._has_result(existing, model_key, task_id):
                    print(f"  [{completed}/{total_runs}] {task_id}: SKIPPED (exists)")
                    self.results[model_key][task_id] = existing[model_key][task_id]
                    continue
                
                print(f"  [{completed}/{total_runs}] {task_id}...", end=" ", flush=True)
                
                # Run the task
                result = self._run_single(task_path, model)
                self.results[model_key][task_id] = result
                
                # Save incrementally
                self._save_result(result)
                
                # Print status
                status = "✓ RESOLVED" if result.resolved else "✗ FAILED"
                if result.error:
                    status = f"✗ ERROR: {result.error[:30]}"
                print(f"{status} ({result.steps} steps, {result.duration:.1f}s)")
        
        # Compute summaries
        model_summaries = []
        for model in models:
            model_key = f"{model.provider}:{model.model}"
            summary = self._compute_summary(model, self.results.get(model_key, {}))
            model_summaries.append(summary)
        
        # Find best models
        best_resolve = max(model_summaries, key=lambda m: m.resolve_rate)
        best_similarity = max(model_summaries, key=lambda m: m.avg_similarity)
        
        benchmark_result = BenchmarkResult(
            timestamp=timestamp,
            config={
                "max_steps": self.max_steps,
                "timeout": self.timeout,
            },
            models=model_summaries,
            tasks=task_ids,
            best_model_by_resolve_rate=best_resolve.model,
            best_model_by_similarity=best_similarity.model,
        )
        
        # Save final results
        self._save_benchmark_result(benchmark_result)
        
        # Print summary
        self._print_summary(benchmark_result)
        
        return benchmark_result
    
    def _run_single(self, task_path: Path, model: ModelConfig) -> TaskResult:
        """Run a single task with a model using test_e2e.py."""
        task_id = self._get_task_id(task_path)
        
        result = TaskResult(
            task_id=task_id,
            model=model.model,
            provider=model.provider,
        )
        
        start_time = time.time()
        
        try:
            # Run test_e2e.py as subprocess
            cmd = [
                sys.executable, "test_e2e.py",
                "--task", str(task_path),
                "--provider", model.provider,
                "--model", model.model,
                "--max-steps", str(self.max_steps),
                "-o", str(self.output_dir / "runs"),
                "--json-output",  # Output results as JSON
            ]
            
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=Path(__file__).parent,
            )
            
            result.duration = time.time() - start_time
            
            # Parse JSON output from test_e2e.py
            if proc.returncode == 0:
                result = self._parse_run_output(proc.stdout, result)
            else:
                result.error = proc.stderr[:200] if proc.stderr else "Unknown error"
                
        except subprocess.TimeoutExpired:
            result.duration = self.timeout
            result.error = f"Timeout after {self.timeout}s"
        except Exception as e:
            result.duration = time.time() - start_time
            result.error = str(e)[:200]
        
        return result
    
    def _parse_run_output(self, output: str, result: TaskResult) -> TaskResult:
        """Parse JSON output from test_e2e.py."""
        # Look for JSON block in output
        try:
            # Find JSON between markers
            if "===JSON_OUTPUT===" in output:
                json_start = output.index("===JSON_OUTPUT===") + len("===JSON_OUTPUT===")
                json_end = output.index("===END_JSON===")
                json_str = output[json_start:json_end].strip()
                data = json.loads(json_str)
            else:
                # Try to parse the whole output as JSON
                # Or look for the last JSON object
                lines = output.strip().split("\n")
                for line in reversed(lines):
                    if line.startswith("{"):
                        data = json.loads(line)
                        break
                else:
                    return result
            
            # Extract metrics
            result.resolved = data.get("resolved", False)
            result.submitted = data.get("submitted", False)
            result.steps = data.get("steps", 0)
            result.similarity_score = data.get("similarity_score", 0.0)
            result.reasoning_score = data.get("reasoning_score", 0.0)
            result.exploration_efficiency = data.get("exploration_efficiency", 0.0)
            result.trajectory_efficiency = data.get("trajectory_efficiency", 0.0)
            result.primary_failure_mode = data.get("primary_failure_mode", "")
            result.failure_reasons = data.get("failure_reasons", [])
            
        except (json.JSONDecodeError, ValueError, KeyError):
            # If we can't parse JSON, try to extract from text output
            result.resolved = "Resolved: True" in output
            result.submitted = "Submitted: True" in output
        
        return result
    
    def _compute_summary(
        self, 
        model: ModelConfig, 
        results: dict[str, TaskResult]
    ) -> ModelSummary:
        """Compute aggregate statistics for a model."""
        summary = ModelSummary(
            model=model.model,
            provider=model.provider,
            total_tasks=len(results),
        )
        
        if not results:
            return summary
        
        task_results = list(results.values())
        summary.task_results = [asdict(r) for r in task_results]
        
        # Counts
        summary.resolved_count = sum(1 for r in task_results if r.resolved)
        summary.submitted_count = sum(1 for r in task_results if r.submitted)
        summary.error_count = sum(1 for r in task_results if r.error)
        
        # Rates
        summary.resolve_rate = summary.resolved_count / summary.total_tasks
        summary.submit_rate = summary.submitted_count / summary.total_tasks
        
        # Averages (excluding errors)
        valid_results = [r for r in task_results if not r.error]
        if valid_results:
            summary.avg_steps = statistics.mean(r.steps for r in valid_results)
            summary.avg_duration = statistics.mean(r.duration for r in valid_results)
            summary.avg_similarity = statistics.mean(r.similarity_score for r in valid_results)
            summary.avg_reasoning_score = statistics.mean(r.reasoning_score for r in valid_results)
            summary.avg_exploration_efficiency = statistics.mean(r.exploration_efficiency for r in valid_results)
            summary.avg_trajectory_efficiency = statistics.mean(r.trajectory_efficiency for r in valid_results)
        
        # Failure mode breakdown
        for r in task_results:
            if r.primary_failure_mode:
                mode = r.primary_failure_mode
                summary.failure_modes[mode] = summary.failure_modes.get(mode, 0) + 1
        
        return summary
    
    def _get_task_id(self, task_path: Path) -> str:
        """Extract task ID from path."""
        return task_path.stem
    
    def _has_result(
        self, 
        existing: dict, 
        model_key: str, 
        task_id: str
    ) -> bool:
        """Check if we already have a result for this model/task."""
        return model_key in existing and task_id in existing[model_key]
    
    def _load_existing_results(self) -> dict:
        """Load existing results from output directory."""
        existing = {}
        results_file = self.output_dir / "all_results.jsonl"
        
        if results_file.exists():
            with open(results_file) as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        model_key = f"{data['provider']}:{data['model']}"
                        task_id = data['task_id']
                        
                        if model_key not in existing:
                            existing[model_key] = {}
                        
                        existing[model_key][task_id] = TaskResult(**{
                            k: v for k, v in data.items()
                            if k in TaskResult.__dataclass_fields__
                        })
                    except (json.JSONDecodeError, KeyError):
                        continue
        
        return existing
    
    def _save_result(self, result: TaskResult) -> None:
        """Append a single result to the results file."""
        results_file = self.output_dir / "all_results.jsonl"
        with open(results_file, "a") as f:
            f.write(json.dumps(asdict(result)) + "\n")
    
    def _save_benchmark_result(self, result: BenchmarkResult) -> None:
        """Save the complete benchmark result."""
        # Save as JSON
        result_file = self.output_dir / "benchmark_result.json"
        with open(result_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        # Save comparison table as markdown
        report_file = self.output_dir / "REPORT.md"
        with open(report_file, "w") as f:
            f.write(self._generate_report(result))
        
        print(f"\nResults saved to: {self.output_dir}")
    
    def _generate_report(self, result: BenchmarkResult) -> str:
        """Generate a markdown report."""
        lines = [
            "# Benchmark Report",
            "",
            f"**Generated:** {result.timestamp}",
            f"**Tasks:** {len(result.tasks)}",
            f"**Models:** {len(result.models)}",
            "",
            "## Summary",
            "",
            "| Model | Provider | Resolved | Submit | Avg Steps | Avg Similarity | Avg Reasoning |",
            "|-------|----------|----------|--------|-----------|----------------|---------------|",
        ]
        
        for m in sorted(result.models, key=lambda x: -x.resolve_rate):
            resolved_pct = f"{m.resolve_rate:.1%}"
            submit_pct = f"{m.submit_rate:.1%}"
            lines.append(
                f"| {m.model} | {m.provider} | {resolved_pct} ({m.resolved_count}/{m.total_tasks}) | "
                f"{submit_pct} | {m.avg_steps:.1f} | {m.avg_similarity:.1%} | {m.avg_reasoning_score:.1%} |"
            )
        
        lines.extend([
            "",
            "## Best Models",
            "",
            f"- **Highest Resolve Rate:** {result.best_model_by_resolve_rate}",
            f"- **Highest Similarity:** {result.best_model_by_similarity}",
            "",
            "## Failure Mode Breakdown",
            "",
        ])
        
        for m in result.models:
            if m.failure_modes:
                lines.append(f"### {m.model}")
                lines.append("")
                for mode, count in sorted(m.failure_modes.items(), key=lambda x: -x[1]):
                    pct = count / m.total_tasks * 100
                    lines.append(f"- {mode}: {count} ({pct:.1f}%)")
                lines.append("")
        
        lines.extend([
            "## Per-Task Results",
            "",
            "| Task | " + " | ".join(m.model for m in result.models) + " |",
            "|------|" + "|".join("---" for _ in result.models) + "|",
        ])
        
        for task_id in result.tasks:
            row = [task_id]
            for m in result.models:
                task_result = next(
                    (r for r in m.task_results if r["task_id"] == task_id),
                    None
                )
                if task_result:
                    status = "✓" if task_result["resolved"] else "✗"
                    row.append(f"{status} {task_result['similarity_score']:.0%}")
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")
        
        return "\n".join(lines)
    
    def _print_summary(self, result: BenchmarkResult) -> None:
        """Print a summary table to console."""
        print()
        print("=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        print()
        
        # Header
        print(f"{'Model':<35} {'Provider':<12} {'Resolved':<12} {'Similarity':<12} {'Steps':<8}")
        print("-" * 80)
        
        # Rows sorted by resolve rate
        for m in sorted(result.models, key=lambda x: -x.resolve_rate):
            resolved = f"{m.resolved_count}/{m.total_tasks} ({m.resolve_rate:.0%})"
            similarity = f"{m.avg_similarity:.1%}"
            steps = f"{m.avg_steps:.1f}"
            print(f"{m.model:<35} {m.provider:<12} {resolved:<12} {similarity:<12} {steps:<8}")
        
        print("-" * 80)
        print()
        print(f"Best resolve rate: {result.best_model_by_resolve_rate}")
        print(f"Best similarity: {result.best_model_by_similarity}")
        print()


# =============================================================================
# RESULTS VIEWER
# =============================================================================

def view_results(output_dir: str) -> None:
    """View existing benchmark results."""
    output_path = Path(output_dir)
    
    result_file = output_path / "benchmark_result.json"
    if not result_file.exists():
        print(f"No benchmark results found in {output_dir}")
        return
    
    with open(result_file) as f:
        data = json.load(f)
    
    # Reconstruct BenchmarkResult
    models = [ModelSummary(**m) for m in data["models"]]
    result = BenchmarkResult(
        timestamp=data["timestamp"],
        config=data["config"],
        models=models,
        tasks=data["tasks"],
        best_model_by_resolve_rate=data["best_model_by_resolve_rate"],
        best_model_by_similarity=data["best_model_by_similarity"],
    )
    
    # Print using existing method
    runner = BenchmarkRunner(output_dir=output_dir)
    runner._print_summary(result)
    
    print(f"Full report: {output_path / 'REPORT.md'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark multiple models on evaluation tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmark with multiple models
  python benchmark.py --tasks eval/tasks/ --models gpt-4o o4-mini claude-sonnet-4-20250514
  
  # Specify providers explicitly  
  python benchmark.py --tasks eval/tasks/ --models openai:gpt-4o anthropic:claude-sonnet-4-20250514
  
  # Quick test with limited tasks
  python benchmark.py --tasks eval/tasks/ --models gpt-4o --max-tasks 3
  
  # View existing results
  python benchmark.py --results-only -o results/benchmark/
        """
    )
    
    # Tasks
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Task JSON files or directories containing tasks",
    )
    
    # Models
    parser.add_argument(
        "--models",
        nargs="+",
        help="Models to benchmark (e.g., gpt-4o, openai:o4-mini, anthropic:claude-sonnet-4-20250514)",
    )
    
    # Output
    parser.add_argument(
        "-o", "--output",
        default="results/benchmark",
        help="Output directory for results (default: results/benchmark)",
    )
    
    # Options
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum agent steps per task (default: 20)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per task in seconds (default: 600)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        help="Limit number of tasks (for testing)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Don't skip existing results (re-run everything)",
    )
    
    # Results only
    parser.add_argument(
        "--results-only",
        action="store_true",
        help="Only view existing results, don't run new benchmarks",
    )
    
    args = parser.parse_args()
    
    # Results only mode
    if args.results_only:
        view_results(args.output)
        return
    
    # Validate inputs
    if not args.tasks:
        parser.error("--tasks is required unless using --results-only")
    if not args.models:
        parser.error("--models is required unless using --results-only")
    
    # Collect task files
    task_files = []
    for task_spec in args.tasks:
        path = Path(task_spec)
        if path.is_file():
            task_files.append(path)
        elif path.is_dir():
            task_files.extend(sorted(path.glob("*.json")))
        else:
            print(f"Warning: {task_spec} not found, skipping")
    
    if not task_files:
        print("Error: No task files found")
        return
    
    # Limit tasks if requested
    if args.max_tasks:
        task_files = task_files[:args.max_tasks]
    
    # Parse model specs
    models = [parse_model_spec(m) for m in args.models]
    
    # Run benchmark
    runner = BenchmarkRunner(
        output_dir=args.output,
        max_steps=args.max_steps,
        timeout=args.timeout,
    )
    
    runner.run(
        tasks=task_files,
        models=models,
        skip_existing=not args.no_skip,
    )


if __name__ == "__main__":
    main()