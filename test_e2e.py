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
import sys
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


def test_agent(task, repo_path, max_steps: int = 10, provider: str = None, model: str = None):
    """Test running the agent."""
    print(f"\n4. Running agent (max_steps={max_steps}, provider={provider or 'auto'}, model={model or 'default'})...")
    
    from agent import RepoAgent
    
    try:
        agent = RepoAgent(max_steps=max_steps, provider=provider, model=model)
        
        result = agent.solve(
            task=task,
            repo_path=repo_path,
            include_hints=True,
        )
        
        print(f"   ✓ Agent completed")
        print(f"     Success: {result.success}")
        print(f"     Steps: {result.steps}")
        print(f"     Patch size: {len(result.patch)} chars")
        if result.explanation:
            print(f"     Explanation: {result.explanation[:100]}...")
        if result.error:
            print(f"     Error: {result.error}")
        
        return result
    except Exception as e:
        print(f"   ✗ Error running agent: {e}")
        import traceback
        traceback.print_exc()
        return None


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
        
        metrics = compute_debug_metrics(
            task_id=task.id,
            messages=agent_result.messages if agent_result else [],
            agent_patch=agent_result.patch if agent_result else "",
            gold_patch=task.gold_patch,
            relevant_files=task.relevant_files,
            resolved=eval_result.resolved if eval_result else False,
            max_steps=max_steps,
            actual_steps=agent_result.steps if agent_result else 0,
            tests_passed=eval_result.fail_to_pass_passed if eval_result else 0,
            tests_failed=(eval_result.fail_to_pass_total - eval_result.fail_to_pass_passed) if eval_result else 0,
        )
        
        report = format_debug_report(metrics)
        print(report)
        
        return metrics
    except Exception as e:
        print(f"   ✗ Error computing metrics: {e}")
        import traceback
        traceback.print_exc()
        return None


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
    
    args = parser.parse_args()
    
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
        agent_result = test_agent(task, repo_path, args.max_steps, args.provider, args.model)
        if not agent_result:
            all_passed = False
    else:
        print("\n4. Skipping agent (--skip-agent)")
    
    # Test evaluation
    eval_result = test_evaluation(task, repo_path, agent_result)
    if not eval_result:
        all_passed = False
    
    # Show debug metrics (always, even if evaluation failed)
    if agent_result:
        show_debug_metrics(task, agent_result, eval_result, args.max_steps)
    
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