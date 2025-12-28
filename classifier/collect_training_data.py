#!/usr/bin/env python
"""
Collect training data for the success classifier.

Runs the agent on tasks and saves detailed metrics that can be used
to train a classifier predicting success/failure.

Usage:
    # Collect data from a directory of tasks
    python collect_training_data.py --tasks eval/tasks/ --provider openai --model gpt-4o
    
    # Collect with multiple models (more diverse training data)
    python collect_training_data.py --tasks eval/tasks/ \
        --provider openai --model gpt-4o \
        --provider openai --model o4-mini
    
    # Append to existing dataset
    python collect_training_data.py --tasks eval/tasks/ --append
    
    # View dataset statistics
    python collect_training_data.py --stats -o training_data/
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent to path for imports
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval import Task
from eval.harness.metrics import compute_debug_metrics


@dataclass
class TrainingExample:
    """A single training example with features and label."""
    
    # Identifiers
    task_id: str
    model: str
    provider: str
    timestamp: str
    
    # Label (what we're predicting)
    resolved: bool
    
    # Features - Reasoning
    reasoning_quality_score: float = 0.0
    has_explicit_reasoning: bool = False
    mentions_issue_keywords: bool = False
    mentions_relevant_files: bool = False
    hypothesizes_before_acting: bool = False
    explains_changes: bool = False
    verifies_after_change: bool = False
    
    # Features - Phases
    exploration_steps: int = 0
    implementation_steps: int = 0
    verification_steps: int = 0
    exploration_pct: float = 0.0
    implementation_pct: float = 0.0
    verification_pct: float = 0.0
    phase_transitions: int = 0
    followed_read_before_write: bool = False
    followed_test_after_change: bool = False
    
    # Features - Exploration
    exploration_strategy: str = ""
    files_explored: int = 0
    directories_explored: int = 0
    relevant_file_discovery_step: int = -1
    exploration_efficiency: float = 0.0
    wasted_explorations: int = 0
    search_to_read_ratio: float = 0.0
    
    # Features - Trajectory
    trajectory_length: int = 0
    optimal_length: int = 0
    trajectory_efficiency: float = 0.0
    unnecessary_steps: int = 0
    
    # Features - Convergence
    final_similarity: float = 0.0
    max_progress: float = 0.0
    converged: bool = False
    monotonic_progress: bool = False
    had_regression: bool = False
    progress_volatility: float = 0.0
    
    # Features - Error Recovery
    total_errors: int = 0
    recovered_errors: int = 0
    recovery_rate: float = 0.0
    max_repetition: int = 0
    stuck_episodes: int = 0
    max_stuck_duration: int = 0
    
    # Features - Tool Usage
    total_tool_calls: int = 0
    read_relevant_files: bool = False
    used_str_replace: bool = False
    used_write_file: bool = False
    ran_tests: bool = False
    submitted: bool = False
    tool_errors_count: int = 0
    
    # Features - Patch Quality
    correct_files_touched: bool = False
    patch_similarity: float = 0.0
    line_level_similarity: float = 0.0
    lines_added: int = 0
    lines_removed: int = 0
    patch_too_large: bool = False
    
    # Features - Derived
    steps_per_file: float = 0.0
    edit_to_explore_ratio: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_metrics(
        cls,
        task_id: str,
        model: str,
        provider: str,
        resolved: bool,
        metrics,  # DebugMetrics
    ) -> "TrainingExample":
        """Create a training example from computed metrics."""
        
        example = cls(
            task_id=task_id,
            model=model,
            provider=provider,
            timestamp=datetime.now().isoformat(),
            resolved=resolved,
        )
        
        # Reasoning features
        r = metrics.reasoning_metrics
        example.reasoning_quality_score = r.reasoning_quality_score
        example.has_explicit_reasoning = r.has_explicit_reasoning
        example.mentions_issue_keywords = r.mentions_issue_keywords
        example.mentions_relevant_files = r.mentions_relevant_files
        example.hypothesizes_before_acting = r.hypothesizes_before_acting
        example.explains_changes = r.explains_changes
        example.verifies_after_change = r.verifies_after_change
        
        # Phase features
        p = metrics.phase_metrics
        example.exploration_steps = p.exploration_steps
        example.implementation_steps = p.implementation_steps
        example.verification_steps = p.verification_steps
        example.exploration_pct = p.exploration_pct
        example.implementation_pct = p.implementation_pct
        example.verification_pct = p.verification_pct
        example.phase_transitions = p.phase_transitions
        example.followed_read_before_write = p.followed_read_before_write
        example.followed_test_after_change = p.followed_test_after_change
        
        # Exploration features
        e = metrics.exploration_metrics
        example.exploration_strategy = e.exploration_strategy
        example.files_explored = e.files_explored
        example.directories_explored = e.directories_explored
        example.relevant_file_discovery_step = e.relevant_file_discovery_step
        example.exploration_efficiency = e.exploration_efficiency
        example.wasted_explorations = e.wasted_explorations
        example.search_to_read_ratio = e.search_to_read_ratio
        
        # Trajectory features
        t = metrics.trajectory_metrics
        example.trajectory_length = t.trajectory_length
        example.optimal_length = t.optimal_length
        example.trajectory_efficiency = t.trajectory_efficiency
        example.unnecessary_steps = len(t.unnecessary_steps)
        
        # Convergence features
        c = metrics.convergence_metrics
        example.final_similarity = c.final_similarity
        example.max_progress = c.max_progress
        example.converged = c.converged
        example.monotonic_progress = c.monotonic_progress
        example.had_regression = c.had_regression
        example.progress_volatility = c.progress_volatility
        
        # Error recovery features
        er = metrics.error_recovery_metrics
        example.total_errors = er.total_errors
        example.recovered_errors = er.recovered_errors
        example.recovery_rate = er.recovery_rate
        example.max_repetition = er.max_repetition
        example.stuck_episodes = len(er.stuck_episodes)
        example.max_stuck_duration = er.max_stuck_duration
        
        # Tool usage features
        tu = metrics.tool_usage
        example.total_tool_calls = tu.total_calls
        example.read_relevant_files = tu.read_relevant_files
        example.used_str_replace = tu.used_str_replace
        example.used_write_file = tu.used_write_file
        example.ran_tests = tu.ran_tests
        example.submitted = tu.submitted
        example.tool_errors_count = len(tu.tool_errors)
        
        # Patch quality features
        pq = metrics.patch_quality
        example.correct_files_touched = pq.correct_files_touched
        example.patch_similarity = pq.similarity_score
        example.line_level_similarity = pq.line_level_similarity
        example.lines_added = pq.lines_added
        example.lines_removed = pq.lines_removed
        example.patch_too_large = pq.patch_too_large
        
        # Derived features
        if example.files_explored > 0:
            example.steps_per_file = example.trajectory_length / example.files_explored
        if example.exploration_steps > 0:
            example.edit_to_explore_ratio = example.implementation_steps / example.exploration_steps
        
        return example


class TrainingDataCollector:
    """Collects training data by running agent on tasks."""
    
    def __init__(
        self,
        output_dir: str = "training_data",
        max_steps: int = 20,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_steps = max_steps
        self.data_file = self.output_dir / "training_data.jsonl"
    
    def collect(
        self,
        tasks: list[Path],
        provider: str,
        model: str,
        skip_existing: bool = True,
    ) -> list[TrainingExample]:
        """Run agent on tasks and collect training examples."""
        
        from agent import RepoAgent
        from eval.harness import RepoManager, Evaluator
        
        repo_manager = RepoManager()
        evaluator = Evaluator()
        
        # Load existing examples
        existing_keys = set()
        if skip_existing:
            existing_keys = self._load_existing_keys()
        
        examples = []
        
        print(f"Collecting training data: {len(tasks)} tasks with {model}")
        print("-" * 60)
        
        for i, task_path in enumerate(tasks, 1):
            task = Task.load(task_path)
            key = f"{task.id}:{provider}:{model}"
            
            if key in existing_keys:
                print(f"[{i}/{len(tasks)}] {task.id}: SKIPPED (exists)")
                continue
            
            print(f"[{i}/{len(tasks)}] {task.id}...", end=" ", flush=True)
            
            try:
                # Setup repo
                repo_path = repo_manager.clone(task.repo)
                repo_manager.reset(repo_path)
                repo_manager.checkout(repo_path, task.base_commit)
                
                # Run agent
                agent = RepoAgent(model=model, max_steps=self.max_steps)
                agent.llm.provider = provider
                
                start_time = time.time()
                agent_result = agent.solve(
                    task=task,
                    repo_path=repo_path,
                    include_hints=True,
                )
                duration = time.time() - start_time
                
                # Evaluate
                eval_result = evaluator.evaluate(
                    task=task,
                    repo_path=repo_path,
                    agent_patch=agent_result.patch,
                    steps=agent_result.steps,
                    duration=duration,
                )
                
                # Compute metrics
                metrics = compute_debug_metrics(
                    task_id=task.id,
                    messages=agent_result.messages,
                    agent_patch=agent_result.patch,
                    gold_patch=task.gold_patch,
                    relevant_files=task.relevant_files,
                    resolved=eval_result.resolved,
                    max_steps=self.max_steps,
                    actual_steps=agent_result.steps,
                    issue_body=task.issue_body,
                    fail_to_pass=task.fail_to_pass,
                )
                
                # Create training example
                example = TrainingExample.from_metrics(
                    task_id=task.id,
                    model=model,
                    provider=provider,
                    resolved=eval_result.resolved,
                    metrics=metrics,
                )
                
                # Save immediately
                self._save_example(example)
                examples.append(example)
                
                status = "✓" if example.resolved else "✗"
                print(f"{status} ({agent_result.steps} steps, {duration:.1f}s)")
                
                # Reset repo
                repo_manager.reset(repo_path)
                
            except Exception as e:
                print(f"ERROR: {e}")
                continue
        
        print("-" * 60)
        print(f"Collected {len(examples)} new examples")
        print(f"Total in dataset: {len(self._load_all_examples())}")
        
        return examples
    
    def collect_from_results(self, results_dir: str) -> list[TrainingExample]:
        """
        Create training examples from existing run results.
        
        This is useful if you've already run the agent and saved results,
        and want to use them for training without re-running.
        """
        results_path = Path(results_dir)
        examples = []
        
        # Look for result JSON files
        for result_file in results_path.glob("*.json"):
            if result_file.name in ("benchmark_result.json", "runs.jsonl"):
                continue
            
            try:
                with open(result_file) as f:
                    data = json.load(f)
                
                # Check if it has the metrics we need
                if "metrics" not in data or not data["metrics"]:
                    continue
                
                metrics_data = data["metrics"]
                
                # Create example from saved metrics
                example = TrainingExample(
                    task_id=data["task"]["id"],
                    model=data["config"]["model"],
                    provider=data["config"]["provider"],
                    timestamp=data["timestamp"],
                    resolved=data["evaluation"]["resolved"],
                )
                
                # Fill in features from saved metrics
                if "reasoning" in metrics_data:
                    r = metrics_data["reasoning"]
                    example.reasoning_quality_score = r.get("quality_score", 0)
                    example.has_explicit_reasoning = r.get("has_explicit_reasoning", False)
                    example.mentions_issue_keywords = r.get("mentions_issue_keywords", False)
                    example.mentions_relevant_files = r.get("mentions_relevant_files", False)
                    example.hypothesizes_before_acting = r.get("hypothesizes_before_acting", False)
                    example.explains_changes = r.get("explains_changes", False)
                    example.verifies_after_change = r.get("verifies_after_change", False)
                
                if "phases" in metrics_data:
                    p = metrics_data["phases"]
                    example.exploration_steps = p.get("exploration_steps", 0)
                    example.implementation_steps = p.get("implementation_steps", 0)
                    example.verification_steps = p.get("verification_steps", 0)
                    example.exploration_pct = p.get("exploration_pct", 0)
                    example.phase_transitions = p.get("phase_transitions", 0)
                    example.followed_read_before_write = p.get("followed_read_before_write", False)
                    example.followed_test_after_change = p.get("followed_test_after_change", False)
                
                if "exploration" in metrics_data:
                    e = metrics_data["exploration"]
                    example.exploration_strategy = e.get("strategy", "")
                    example.files_explored = e.get("files_explored", 0)
                    example.directories_explored = e.get("directories_explored", 0)
                    example.relevant_file_discovery_step = e.get("relevant_file_discovery_step", -1)
                    example.exploration_efficiency = e.get("exploration_efficiency", 0)
                    example.wasted_explorations = e.get("wasted_explorations", 0)
                
                if "trajectory" in metrics_data:
                    t = metrics_data["trajectory"]
                    example.trajectory_length = t.get("length", 0)
                    example.optimal_length = t.get("optimal_length", 0)
                    example.trajectory_efficiency = t.get("efficiency", 0)
                    example.unnecessary_steps = t.get("unnecessary_steps", 0)
                
                if "convergence" in metrics_data:
                    c = metrics_data["convergence"]
                    example.final_similarity = c.get("final_similarity", 0)
                    example.max_progress = c.get("max_progress", 0)
                    example.converged = c.get("converged", False)
                    example.monotonic_progress = c.get("monotonic_progress", False)
                    example.had_regression = c.get("had_regression", False)
                    example.progress_volatility = c.get("progress_volatility", 0)
                
                if "error_recovery" in metrics_data:
                    er = metrics_data["error_recovery"]
                    example.total_errors = er.get("total_errors", 0)
                    example.recovered_errors = er.get("recovered_errors", 0)
                    example.recovery_rate = er.get("recovery_rate", 0)
                    example.max_repetition = er.get("max_repetition", 0)
                    example.stuck_episodes = er.get("stuck_episodes", 0)
                    example.max_stuck_duration = er.get("max_stuck_duration", 0)
                
                if "tool_usage" in metrics_data:
                    tu = metrics_data["tool_usage"]
                    example.total_tool_calls = tu.get("total_calls", 0)
                    example.read_relevant_files = tu.get("read_relevant_files", False)
                    example.used_str_replace = tu.get("used_str_replace", False)
                    example.used_write_file = tu.get("used_write_file", False)
                    example.ran_tests = tu.get("ran_tests", False)
                    example.submitted = tu.get("submitted", False)
                    example.tool_errors_count = len(tu.get("tool_errors", []))
                
                if "patch_quality" in metrics_data:
                    pq = metrics_data["patch_quality"]
                    example.correct_files_touched = pq.get("correct_files_touched", False)
                    example.patch_similarity = pq.get("similarity_score", 0)
                    example.line_level_similarity = pq.get("line_level_similarity", 0)
                    example.lines_added = pq.get("lines_added", 0)
                    example.lines_removed = pq.get("lines_removed", 0)
                    example.patch_too_large = pq.get("patch_too_large", False)
                
                # Derived features
                if example.files_explored > 0:
                    example.steps_per_file = example.trajectory_length / example.files_explored
                if example.exploration_steps > 0:
                    example.edit_to_explore_ratio = example.implementation_steps / example.exploration_steps
                
                self._save_example(example)
                examples.append(example)
                
                print(f"Loaded: {result_file.name} -> {'✓' if example.resolved else '✗'}")
                
            except Exception as e:
                print(f"Error loading {result_file}: {e}")
                continue
        
        print(f"\nLoaded {len(examples)} examples from results")
        return examples
    
    def _save_example(self, example: TrainingExample) -> None:
        """Append an example to the data file."""
        with open(self.data_file, "a") as f:
            f.write(json.dumps(example.to_dict()) + "\n")
    
    def _load_existing_keys(self) -> set:
        """Load keys of existing examples."""
        keys = set()
        if self.data_file.exists():
            with open(self.data_file) as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        key = f"{data['task_id']}:{data['provider']}:{data['model']}"
                        keys.add(key)
                    except:
                        continue
        return keys
    
    def _load_all_examples(self) -> list[TrainingExample]:
        """Load all examples from the data file."""
        examples = []
        if self.data_file.exists():
            with open(self.data_file) as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        examples.append(TrainingExample(**data))
                    except:
                        continue
        return examples
    
    def show_stats(self) -> None:
        """Show statistics about the collected data."""
        examples = self._load_all_examples()
        
        if not examples:
            print("No training data found.")
            return
        
        print("=" * 60)
        print("TRAINING DATA STATISTICS")
        print("=" * 60)
        print()
        
        # Basic counts
        total = len(examples)
        resolved = sum(1 for e in examples if e.resolved)
        print(f"Total examples: {total}")
        print(f"Resolved: {resolved} ({resolved/total:.1%})")
        print(f"Failed: {total - resolved} ({(total-resolved)/total:.1%})")
        print()
        
        # By model
        by_model = {}
        for e in examples:
            key = f"{e.provider}:{e.model}"
            if key not in by_model:
                by_model[key] = {"total": 0, "resolved": 0}
            by_model[key]["total"] += 1
            if e.resolved:
                by_model[key]["resolved"] += 1
        
        print("By Model:")
        for model, stats in sorted(by_model.items()):
            rate = stats["resolved"] / stats["total"]
            print(f"  {model}: {stats['resolved']}/{stats['total']} ({rate:.1%})")
        print()
        
        # Feature statistics
        print("Feature Statistics (resolved vs failed):")
        print()
        
        resolved_examples = [e for e in examples if e.resolved]
        failed_examples = [e for e in examples if not e.resolved]
        
        def avg(lst, attr):
            values = [getattr(e, attr) for e in lst]
            return sum(values) / len(values) if values else 0
        
        features = [
            ("reasoning_quality_score", "Reasoning Quality"),
            ("exploration_efficiency", "Exploration Efficiency"),
            ("trajectory_efficiency", "Trajectory Efficiency"),
            ("final_similarity", "Final Similarity"),
            ("total_tool_calls", "Total Tool Calls"),
            ("exploration_pct", "Exploration %"),
            ("implementation_pct", "Implementation %"),
            ("verification_pct", "Verification %"),
        ]
        
        print(f"{'Feature':<25} {'Resolved':<12} {'Failed':<12} {'Diff':<10}")
        print("-" * 60)
        
        for attr, name in features:
            r_avg = avg(resolved_examples, attr)
            f_avg = avg(failed_examples, attr)
            diff = r_avg - f_avg
            diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
            print(f"{name:<25} {r_avg:<12.2f} {f_avg:<12.2f} {diff_str:<10}")
        
        print()
        print(f"Data file: {self.data_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect training data for success classifier",
    )
    
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Task files or directories",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model name (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "-o", "--output",
        default="training_data",
        help="Output directory (default: training_data)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Max agent steps (default: 20)",
    )
    parser.add_argument(
        "--from-results",
        help="Load training data from existing results directory",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show dataset statistics",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Don't skip existing examples",
    )
    
    args = parser.parse_args()
    
    collector = TrainingDataCollector(
        output_dir=args.output,
        max_steps=args.max_steps,
    )
    
    # Show stats
    if args.stats:
        collector.show_stats()
        return
    
    # Load from existing results
    if args.from_results:
        collector.collect_from_results(args.from_results)
        collector.show_stats()
        return
    
    # Collect new data
    if not args.tasks:
        parser.error("--tasks is required unless using --stats or --from-results")
    
    # Collect task files
    task_files = []
    for task_spec in args.tasks:
        path = Path(task_spec)
        if path.is_file():
            task_files.append(path)
        elif path.is_dir():
            task_files.extend(sorted(path.glob("*.json")))
    
    if not task_files:
        print("No task files found")
        return
    
    collector.collect(
        tasks=task_files,
        provider=args.provider,
        model=args.model,
        skip_existing=not args.no_skip,
    )
    
    collector.show_stats()


if __name__ == "__main__":
    main()
