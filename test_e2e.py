#!/usr/bin/env python
"""
End-to-end test of the evaluation pipeline.

This script:
1. Loads a sample task
2. Clones/sets up the repository
3. Runs the agent on the task
4. Evaluates the result
5. Generates a report

Usage:
    python test_e2e.py [--task TASK_FILE] [--skip-agent] [--verbose]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def test_imports():
    """Test that all imports work."""
    print("1. Testing imports...")
    
    try:
        from eval import Task, EvaluationRunner, ResultsStore
        from eval.harness import RepoManager, Evaluator
        from agent import RepoAgent
        print("   ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"   ✗ Import error: {e}")
        return False


def test_load_task(task_file: str):
    """Test loading a task."""
    print(f"\n2. Loading task from {task_file}...")
    
    from eval import Task
    
    try:
        task = Task.load(task_file)
        print(f"   ✓ Loaded task: {task.id}")
        print(f"     Repo: {task.repo}")
        print(f"     Issue: {task.issue_title[:60]}...")
        print(f"     Difficulty: {task.difficulty}")
        print(f"     Files: {task.relevant_files}")
        print(f"     Tests to pass: {task.fail_to_pass}")
        return task
    except Exception as e:
        print(f"   ✗ Error loading task: {e}")
        return None


def test_repo_setup(task):
    """Test cloning and checking out the repo."""
    print(f"\n3. Setting up repository...")
    
    from eval.harness import RepoManager
    from pathlib import Path
    
    try:
        is_local = task.repo.startswith("local/")
        
        if is_local:
            # Handle local repos directly
            repo_name = task.repo.split("/", 1)[1]  # e.g., "test_repo"
            
            # Search for the repo in common locations
            repo_path = None
            for base in [Path.cwd(), Path(__file__).parent, Path.cwd().parent]:
                candidate = base / repo_name
                if candidate.exists() and (candidate / ".git").exists():
                    repo_path = candidate
                    break
            
            if not repo_path:
                print(f"   ✗ Local repo not found: {repo_name}")
                print(f"     Searched in: {Path.cwd()}, {Path(__file__).parent}")
                return None
            
            print(f"   ✓ Using local repo: {repo_path}")
            
            # Reset to clean state
            import subprocess
            subprocess.run(["git", "reset", "--hard"], cwd=repo_path, capture_output=True)
            subprocess.run(["git", "clean", "-fd"], cwd=repo_path, capture_output=True)
            print(f"   ✓ Reset to clean state")
            
            return repo_path
        
        else:
            # Remote repo - use RepoManager
            manager = RepoManager()
            
            print(f"   Cloning {task.repo}...")
            repo_path = manager.clone(task.repo)
            print(f"   ✓ Cloned to: {repo_path}")
            
            print(f"   Checking out {task.base_commit[:8]}...")
            manager.reset(repo_path)
            manager.checkout(repo_path, task.base_commit)
            print(f"   ✓ Checked out base commit")
            
            return repo_path
            
    except Exception as e:
        print(f"   ✗ Error setting up repo: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_agent(task, repo_path, max_steps: int = 10, provider: str = None, model: str = None,
               n_samples: int = 1, sampling_strategy: str = "first"):
    """Test running the agent."""
    from agent import RepoAgent
    from agent.repo_agent import RepoAgentResult
    
    if n_samples > 1:
        print(f"\n4. Running agent with sampling (n={n_samples}, strategy={sampling_strategy})...")
    else:
        print(f"\n4. Running agent (max_steps={max_steps}, provider={provider or 'auto'}, model={model or 'default'})...")
    
    try:
        agent = RepoAgent(max_steps=max_steps, provider=provider, model=model)
        
        if n_samples > 1:
            # Use sampling
            sampling_result = agent.solve_with_sampling(
                task=task,
                repo_path=repo_path,
                n_samples=n_samples,
                strategy=sampling_strategy,
                include_hints=True,
            )
            result = sampling_result.selected_result
            
            print(f"   ✓ Agent completed ({n_samples} samples)")
            print(f"     Strategy: {sampling_strategy}")
            print(f"     Selected sample: {sampling_result.selected_index + 1}/{n_samples}")
            print(f"     Samples with patches: {sampling_result.n_with_patch}/{n_samples}")
            print(f"     Samples submitted: {sampling_result.n_submitted}/{n_samples}")
            if sampling_result.scores:
                print(f"     Scores: {[f'{s:.1f}' for s in sampling_result.scores]}")
        else:
            # Single run (original behavior)
            result = agent.solve(
                task=task,
                repo_path=repo_path,
                include_hints=True,
            )
            sampling_result = None
            print(f"   ✓ Agent completed")
        
        print(f"     Success: {result.success}")
        print(f"     Steps: {result.steps}")
        print(f"     Patch size: {len(result.patch)} chars")
        if result.explanation:
            print(f"     Explanation: {result.explanation[:100]}...")
        if result.error:
            print(f"     Error: {result.error}")
        
        # Show the actual patch
        if result.patch:
            print(f"\n   Agent's patch:")
            print("-" * 60)
            print(result.patch[:2000])
            if len(result.patch) > 2000:
                print(f"... ({len(result.patch) - 2000} more chars)")
            print("-" * 60)
        
        return result
    except Exception as e:
        print(f"   ✗ Error running agent: {e}")
        import traceback
        traceback.print_exc()
        # Return an error result instead of None
        return RepoAgentResult(
            success=False,
            patch="",
            explanation="",
            steps=0,
            messages=[],
            error=str(e),
        )


def test_evaluation(task, repo_path, agent_result):
    """Test evaluating the agent's patch."""
    print(f"\n5. Evaluating result...")
    
    from eval.harness import Evaluator
    
    try:
        evaluator = Evaluator(timeout=120)
        
        result = evaluator.evaluate(
            task=task,
            repo_path=repo_path,
            agent_patch=agent_result.patch if agent_result else "",
            steps=agent_result.steps if agent_result else 0,
            duration=0.0,
            explanation=agent_result.explanation if agent_result else "",
        )
        
        print(f"   ✓ Evaluation completed")
        print(f"     Resolved: {result.resolved}")
        print(f"     No regression: {result.no_regression}")
        print(f"     Diff size: {result.diff_size} lines")
        print(f"     Files changed: {result.files_changed}")
        
        if result.fail_to_pass_results:
            print(f"     Fail-to-pass tests: {result.fail_to_pass_passed}/{result.fail_to_pass_total} passed")
        
        return result
    except Exception as e:
        print(f"   ✗ Error evaluating: {e}")
        import traceback
        traceback.print_exc()
        return None


def show_debug_metrics(task, agent_result, eval_result, max_steps):
    """Show detailed debug metrics for the run."""
    print(f"\n7. Debug Metrics...")
    
    try:
        from eval.harness.metrics import compute_debug_metrics, format_debug_report
        
        agent_patch = agent_result.patch if agent_result else ""
        
        metrics = compute_debug_metrics(
            task_id=task.id,
            messages=agent_result.messages if agent_result else [],
            agent_patch=agent_patch,
            gold_patch=task.gold_patch,
            relevant_files=task.relevant_files,
            resolved=eval_result.resolved if eval_result else False,
            max_steps=max_steps,
            actual_steps=agent_result.steps if agent_result else 0,
            tests_passed=eval_result.fail_to_pass_passed if eval_result else 0,
            tests_failed=(eval_result.fail_to_pass_total - eval_result.fail_to_pass_passed) if eval_result else 0,
            issue_body=task.issue_body,
            fail_to_pass=task.fail_to_pass,
        )
        
        report = format_debug_report(metrics, agent_patch)
        print(report)
        
        return metrics
    except Exception as e:
        print(f"   ✗ Error computing metrics: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_run_results(
    task,
    agent_result,
    eval_result,
    metrics,
    provider: str,
    model: str,
    max_steps: int,
    output_dir: str = "results",
    detailed_metrics: bool = False,
):
    """Save run results to a JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build result record
    result = {
        "timestamp": datetime.now().isoformat(),
        "task": {
            "id": task.id,
            "repo": task.repo,
            "issue_title": task.issue_title,
            "difficulty": task.difficulty,
            "relevant_files": task.relevant_files,
            "fail_to_pass": task.fail_to_pass,
        },
        "config": {
            "provider": provider,
            "model": model,
            "max_steps": max_steps,
            "detailed_metrics": detailed_metrics,
        },
        "agent": {
            "success": agent_result.success if agent_result else False,
            "steps": agent_result.steps if agent_result else 0,
            "patch": agent_result.patch if agent_result else "",
            "explanation": agent_result.explanation if agent_result else "",
            "error": agent_result.error if agent_result else None,
        },
        "evaluation": {
            "resolved": eval_result.resolved if eval_result else False,
            "no_regression": eval_result.no_regression if eval_result else True,
            "diff_size": eval_result.diff_size if eval_result else 0,
            "files_changed": eval_result.files_changed if eval_result else [],
            "fail_to_pass_passed": eval_result.fail_to_pass_passed if eval_result else 0,
            "fail_to_pass_total": eval_result.fail_to_pass_total if eval_result else 0,
        },
        "metrics": {},
    }
    
    # Add core metrics (always saved)
    if metrics:
        result["metrics"]["core"] = {
            "similarity_score": metrics.patch_quality.similarity_score,
            "reasoning_score": metrics.reasoning_metrics.reasoning_quality_score,
            "exploration_efficiency": metrics.exploration_metrics.exploration_efficiency,
            "trajectory_efficiency": metrics.trajectory_metrics.trajectory_efficiency,
            "primary_failure_mode": metrics.failure_analysis.primary_failure_mode,
            "failure_reasons": metrics.failure_analysis.failure_reasons,
        }
        
        # Add detailed metrics only if requested
        if detailed_metrics:
            result["metrics"]["detailed"] = {
                "tool_usage": {
                    "total_calls": metrics.tool_usage.total_calls,
                    "calls_by_tool": metrics.tool_usage.calls_by_tool,
                    "tool_sequence": metrics.tool_usage.tool_sequence,
                    "read_relevant_files": metrics.tool_usage.read_relevant_files,
                    "used_str_replace": metrics.tool_usage.used_str_replace,
                    "used_write_file": metrics.tool_usage.used_write_file,
                    "ran_tests": metrics.tool_usage.ran_tests,
                    "submitted": metrics.tool_usage.submitted,
                    "tool_errors": metrics.tool_usage.tool_errors,
                },
                "patch_quality": {
                    "files_changed": metrics.patch_quality.files_changed,
                    "gold_files_touched": metrics.patch_quality.gold_files_touched,
                    "correct_files_touched": metrics.patch_quality.correct_files_touched,
                    "extra_files_touched": metrics.patch_quality.extra_files_touched,
                    "missing_files": metrics.patch_quality.missing_files,
                    "lines_added": metrics.patch_quality.lines_added,
                    "lines_removed": metrics.patch_quality.lines_removed,
                    "line_level_similarity": metrics.patch_quality.line_level_similarity,
                    "patch_too_large": metrics.patch_quality.patch_too_large,
                },
                "reasoning": {
                    "has_explicit_reasoning": metrics.reasoning_metrics.has_explicit_reasoning,
                    "mentions_issue_keywords": metrics.reasoning_metrics.mentions_issue_keywords,
                    "mentions_relevant_files": metrics.reasoning_metrics.mentions_relevant_files,
                    "hypothesizes_before_acting": metrics.reasoning_metrics.hypothesizes_before_acting,
                    "explains_changes": metrics.reasoning_metrics.explains_changes,
                    "verifies_after_change": metrics.reasoning_metrics.verifies_after_change,
                    "issue_keyword_matches": metrics.reasoning_metrics.issue_keyword_matches,
                },
                "phases": {
                    "exploration_steps": metrics.phase_metrics.exploration_steps,
                    "implementation_steps": metrics.phase_metrics.implementation_steps,
                    "verification_steps": metrics.phase_metrics.verification_steps,
                    "exploration_pct": metrics.phase_metrics.exploration_pct,
                    "phase_transitions": metrics.phase_metrics.phase_transitions,
                    "followed_read_before_write": metrics.phase_metrics.followed_read_before_write,
                    "followed_test_after_change": metrics.phase_metrics.followed_test_after_change,
                },
                "exploration": {
                    "strategy": metrics.exploration_metrics.exploration_strategy,
                    "files_explored": metrics.exploration_metrics.files_explored,
                    "directories_explored": metrics.exploration_metrics.directories_explored,
                    "relevant_file_discovery_step": metrics.exploration_metrics.relevant_file_discovery_step,
                    "wasted_explorations": metrics.exploration_metrics.wasted_explorations,
                },
                "trajectory": {
                    "length": metrics.trajectory_metrics.trajectory_length,
                    "optimal_length": metrics.trajectory_metrics.optimal_length,
                    "unnecessary_steps": len(metrics.trajectory_metrics.unnecessary_steps),
                    "agent_trajectory": metrics.trajectory_metrics.agent_trajectory,
                },
                "convergence": {
                    "final_similarity": metrics.convergence_metrics.final_similarity,
                    "max_progress": metrics.convergence_metrics.max_progress,
                    "converged": metrics.convergence_metrics.converged,
                    "monotonic_progress": metrics.convergence_metrics.monotonic_progress,
                    "had_regression": metrics.convergence_metrics.had_regression,
                    "progress_volatility": metrics.convergence_metrics.progress_volatility,
                    "progress_curve": metrics.convergence_metrics.progress_curve,
                },
                "error_recovery": {
                    "total_errors": metrics.error_recovery_metrics.total_errors,
                    "recovered_errors": metrics.error_recovery_metrics.recovered_errors,
                    "recovery_rate": metrics.error_recovery_metrics.recovery_rate,
                    "max_repetition": metrics.error_recovery_metrics.max_repetition,
                    "stuck_episodes": len(metrics.error_recovery_metrics.stuck_episodes),
                    "max_stuck_duration": metrics.error_recovery_metrics.max_stuck_duration,
                },
                "failure_analysis": {
                    "hit_max_steps": metrics.failure_analysis.hit_max_steps,
                    "agent_submitted": metrics.failure_analysis.agent_submitted,
                    "failure_modes": metrics.failure_analysis.failure_modes,
                    "no_changes_made": metrics.failure_analysis.no_changes_made,
                    "wrong_files_modified": metrics.failure_analysis.wrong_files_modified,
                    "patch_too_large": metrics.failure_analysis.patch_too_large,
                    "tool_errors_occurred": metrics.failure_analysis.tool_errors_occurred,
                    "model_got_stuck": metrics.failure_analysis.model_got_stuck,
                },
            }
    
    # Save to file
    model_safe = (model or "default").replace("/", "-")
    filename = f"{task.id}_{model_safe}_{timestamp}.json"
    filepath = output_path / filename
    
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\n8. Results saved to: {filepath}")
    
    # Also append to a summary JSONL file
    summary_file = output_path / "runs.jsonl"
    summary_record = {
        "timestamp": result["timestamp"],
        "task_id": task.id,
        "provider": provider,
        "model": model,
        "resolved": result["evaluation"]["resolved"],
        "steps": result["agent"]["steps"],
        "similarity": metrics.patch_quality.similarity_score if metrics else 0,
        "reasoning_score": metrics.reasoning_metrics.reasoning_quality_score if metrics else 0,
        "exploration_efficiency": metrics.exploration_metrics.exploration_efficiency if metrics else 0,
        "trajectory_efficiency": metrics.trajectory_metrics.trajectory_efficiency if metrics else 0,
        "primary_failure_mode": metrics.failure_analysis.primary_failure_mode if metrics else "",
        "result_file": str(filepath),
    }
    
    with open(summary_file, "a") as f:
        f.write(json.dumps(summary_record) + "\n")
    
    return filepath


def show_runs_summary(output_dir: str = "results"):
    """Show a summary of all previous runs."""
    output_path = Path(output_dir)
    summary_file = output_path / "runs.jsonl"
    
    if not summary_file.exists():
        print(f"No runs found in {output_dir}/")
        return
    
    runs = []
    with open(summary_file) as f:
        for line in f:
            if line.strip():
                runs.append(json.loads(line))
    
    if not runs:
        print("No runs found.")
        return
    
    print(f"\n{'='*80}")
    print(f"RUNS SUMMARY ({len(runs)} total)")
    print(f"{'='*80}")
    print(f"{'Timestamp':<20} {'Task ID':<35} {'Model':<20} {'Resolved':<10} {'Steps':<6} {'Similarity':<10}")
    print("-"*80)
    
    # Group by task
    by_task = {}
    for run in runs:
        task_id = run["task_id"]
        if task_id not in by_task:
            by_task[task_id] = []
        by_task[task_id].append(run)
    
    for task_id, task_runs in by_task.items():
        for run in task_runs:
            ts = run["timestamp"][:19].replace("T", " ")
            model = (run.get("model") or "default")[:18]
            resolved = "✓" if run["resolved"] else "✗"
            steps = run.get("steps", "?")
            similarity = f"{run.get('similarity', 0):.1%}"
            print(f"{ts:<20} {task_id[:33]:<35} {model:<20} {resolved:<10} {steps:<6} {similarity:<10}")
    
    print("-"*80)
    
    # Summary stats
    resolved_count = sum(1 for r in runs if r["resolved"])
    print(f"\nResolved: {resolved_count}/{len(runs)} ({resolved_count/len(runs):.1%})")
    
    # By model
    by_model = {}
    for run in runs:
        model = run.get("model") or "default"
        if model not in by_model:
            by_model[model] = {"total": 0, "resolved": 0}
        by_model[model]["total"] += 1
        if run["resolved"]:
            by_model[model]["resolved"] += 1
    
    if len(by_model) > 1:
        print("\nBy model:")
        for model, stats in sorted(by_model.items()):
            rate = stats["resolved"] / stats["total"]
            print(f"  {model}: {stats['resolved']}/{stats['total']} ({rate:.1%})")
    
    print()


def test_results_store():
    """Test saving and loading results."""
    print(f"\n6. Testing results storage...")
    
    from eval import ResultsStore, TaskResult
    import tempfile
    
    try:
        # Create temp directory
        tmp_dir = tempfile.mkdtemp()
        store = ResultsStore(tmp_dir)
        
        # Save a mock result
        result = TaskResult(
            task_id="test-task",
            resolved=True,
            no_regression=True,
            steps=5,
            duration=30.0,
        )
        store.save_result(result)
        
        # Load it back
        loaded = store.load_results()
        assert len(loaded) == 1
        assert loaded[0].task_id == "test-task"
        
        # Generate report
        report = store.generate_report()
        assert "test-task" in report
        
        print(f"   ✓ Results storage works")
        return True
    except Exception as e:
        print(f"   ✗ Error with results storage: {e}")
        return False


def run_full_pipeline(task_file: str, max_steps: int = 10, provider: str = None, model: str = None):
    """Run the full evaluation pipeline."""
    print("\n" + "="*60)
    print("FULL PIPELINE TEST")
    print("="*60)
    
    from eval import Task, EvaluationRunner
    from pathlib import Path
    import tempfile
    
    try:
        task = Task.load(task_file)
        print(f"Task: {task.id}")
        print(f"Issue: {task.issue_title}")
        print(f"Provider: {provider or 'auto'}")
        print(f"Model: {model or 'default'}")
        print()
        
        # Determine local_repos_dir
        local_repos_dir = None
        for base in [Path.cwd(), Path(__file__).parent]:
            if (base / "test_repo").exists():
                local_repos_dir = base
                break
        
        # Create runner with temp output
        tmp_dir = tempfile.mkdtemp()
        runner = EvaluationRunner(
            output_dir=tmp_dir,
            local_repos_dir=local_repos_dir,
            max_steps=max_steps,
            provider=provider,
            model=model,
            timeout=300,
        )
        
        # Run single task
        print("Running evaluation...")
        result = runner.run_single(task, include_hints=True)
        
        print("\n" + "-"*40)
        print("RESULT:")
        print(f"  Resolved: {result.resolved}")
        print(f"  No regression: {result.no_regression}")
        print(f"  Steps: {result.steps}")
        print(f"  Duration: {result.duration:.1f}s")
        print(f"  Diff size: {result.diff_size} lines")
        
        if result.error:
            print(f"  Error: {result.error}")
        
        # Show report
        from eval import ResultsStore
        store = ResultsStore(tmp_dir)
        report_path = store.save_report()
        print(f"\nReport saved to: {report_path}")
        
        return result
        
    except Exception as e:
        print(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="End-to-end test")
    parser.add_argument(
        "--task",
        default="eval/tasks/test__mathlib-001.json",
        help="Task file to test with (default: simple local test)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Max agent steps (default: 10)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["anthropic", "groq", "openai", "ollama"],
        help="LLM provider (default: auto from model)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (provider auto-detected if not specified)",
    )
    parser.add_argument(
        "--skip-agent",
        action="store_true",
        help="Skip agent execution (just test setup)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full pipeline test",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results",
        help="Output directory for results (default: results)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary of all previous runs and exit",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Output results as JSON (for benchmark script)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="Number of samples per task (default: 1)",
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=["first", "best_of_n", "majority_vote", "pass_at_k"],
        default="first",
        help="Sampling strategy (default: first)",
    )
    parser.add_argument(
        "--detailed-metrics",
        action="store_true",
        help="Compute and save additional detailed metrics",
    )
    
    args = parser.parse_args()
    
    # Show summary if requested
    if args.summary:
        show_runs_summary(args.output)
        sys.exit(0)
    
    print("="*60)
    print("END-TO-END TEST")
    print("="*60)
    
    # Run full pipeline if requested
    if args.full:
        result = run_full_pipeline(args.task, args.max_steps, args.provider, args.model)
        sys.exit(0 if result and result.resolved else 1)
    
    # Otherwise run step-by-step tests
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
        print("\n❌ Import test failed. Fix imports before continuing.")
        sys.exit(1)
    
    # Test task loading
    task = test_load_task(args.task)
    if not task:
        all_passed = False
        print("\n❌ Task loading failed.")
        sys.exit(1)
    
    # Test results storage
    if not test_results_store():
        all_passed = False
    
    # Test repo setup
    repo_path = test_repo_setup(task)
    if not repo_path:
        all_passed = False
        print("\n❌ Repo setup failed.")
        sys.exit(1)
    
    # Test agent (optional)
    agent_result = None
    if not args.skip_agent:
        agent_result = test_agent(
            task, repo_path, args.max_steps, args.provider, args.model,
            n_samples=args.n_samples,
            sampling_strategy=args.sampling_strategy,
        )
        if not agent_result:
            all_passed = False
    else:
        print("\n4. Skipping agent (--skip-agent)")
    
    # Test evaluation
    eval_result = test_evaluation(task, repo_path, agent_result)
    if not eval_result:
        all_passed = False
    
    # Show debug metrics (always, even if evaluation failed)
    metrics = None
    if agent_result:
        metrics = show_debug_metrics(task, agent_result, eval_result, args.max_steps)
    
    # Save results to file
    if agent_result:
        save_run_results(
            task=task,
            agent_result=agent_result,
            eval_result=eval_result,
            metrics=metrics,
            provider=args.provider or "auto",
            model=args.model or "default",
            max_steps=args.max_steps,
            output_dir=args.output,
            detailed_metrics=args.detailed_metrics,
        )
    
    # Output JSON for benchmark script
    if args.json_output:
        json_result = {
            "task_id": task.id,
            "resolved": eval_result.resolved if eval_result else False,
            "submitted": metrics.tool_usage.submitted if metrics else False,
            "steps": agent_result.steps if agent_result else 0,
            "similarity_score": metrics.patch_quality.similarity_score if metrics else 0.0,
            "reasoning_score": metrics.reasoning_metrics.reasoning_quality_score if metrics else 0.0,
            "exploration_efficiency": metrics.exploration_metrics.exploration_efficiency if metrics else 0.0,
            "trajectory_efficiency": metrics.trajectory_metrics.trajectory_efficiency if metrics else 0.0,
            "primary_failure_mode": metrics.failure_analysis.primary_failure_mode if metrics else "",
            "failure_reasons": metrics.failure_analysis.failure_reasons if metrics else [],
            "error": agent_result.error if agent_result and agent_result.error else "",
        }
        print("===JSON_OUTPUT===")
        print(json.dumps(json_result))
        print("===END_JSON===")
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*60)
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
