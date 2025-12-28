"""Main evaluation runner."""

import time
from datetime import datetime
from pathlib import Path

from ..task import Task
from .repo_manager import RepoManager
from .evaluator import Evaluator
from .results import ResultsStore, TaskResult, EvalSummary

# Import agent (handle potential import issues)
try:
    from agent import RepoAgent
except ImportError:
    RepoAgent = None


class EvaluationRunner:
    """
    Main orchestrator for running evaluations.
    
    Handles the full pipeline: repo setup -> agent run -> evaluation -> results.
    """

    def __init__(
        self,
        output_dir: Path | str = "results",
        cache_dir: Path | str | None = None,
        local_repos_dir: Path | str | None = None,
        model: str | None = None,
        provider: str | None = None,
        max_steps: int = 30,
        timeout: int = 600,
    ):
        """
        Initialize the evaluation runner.
        
        Args:
            output_dir: Directory to save results
            cache_dir: Directory to cache cloned repos
            local_repos_dir: Directory containing local test repos
            model: Model to use for the agent (uses provider default if not specified)
            provider: LLM provider (anthropic, groq, openai, ollama)
            max_steps: Maximum agent steps per task
            timeout: Timeout per task in seconds
        """
        self.output_dir = Path(output_dir)
        self.repo_manager = RepoManager(cache_dir, local_repos_dir)
        self.evaluator = Evaluator(timeout=timeout)
        self.results_store = ResultsStore(self.output_dir)
        
        self.model = model
        self.provider = provider
        self.max_steps = max_steps
        self.timeout = timeout

    def run(
        self,
        tasks: list[Task],
        include_hints: bool = True,
        skip_completed: bool = True,
    ) -> EvalSummary:
        """
        Run evaluation on multiple tasks.
        
        Args:
            tasks: List of tasks to evaluate
            include_hints: Whether to provide relevant_files hints to agent
            skip_completed: Skip tasks that already have results (resume support)
            
        Returns:
            EvalSummary with aggregate metrics
        """
        # Get already completed tasks for resume support
        completed_ids = set()
        if skip_completed:
            completed_ids = self.results_store.get_completed_task_ids()
        
        print(f"=== Evaluation Run ===")
        print(f"Tasks: {len(tasks)}")
        print(f"Already completed: {len(completed_ids)}")
        print(f"Model: {self.model}")
        print(f"Provider: {self.provider or 'auto'}")
        print(f"Max steps: {self.max_steps}")
        print(f"Timeout: {self.timeout}s")
        print()
        
        for i, task in enumerate(tasks, 1):
            if task.id in completed_ids:
                print(f"[{i}/{len(tasks)}] Skipping {task.id} (already completed)")
                continue
            
            print(f"[{i}/{len(tasks)}] Running {task.id}...")
            
            try:
                result = self.run_single(task, include_hints=include_hints)
                
                status = "✓ Resolved" if result.resolved else "✗ Not resolved"
                if result.error:
                    status = f"✗ Error: {result.error}"
                
                print(f"  {status} (steps={result.steps}, time={result.duration:.1f}s)")
                
            except Exception as e:
                print(f"  ✗ Exception: {e}")
                # Save error result
                result = TaskResult(
                    task_id=task.id,
                    error=str(e),
                    timestamp=datetime.now().isoformat(),
                )
                self.results_store.save_result(result)
            
            print()
        
        # Generate summary and report
        summary = self.results_store.get_summary()
        report_path = self.results_store.save_report()
        
        print("=== Summary ===")
        print(f"Resolved: {summary.resolved_count}/{summary.total_tasks} ({summary.resolve_rate:.1%})")
        print(f"No regression: {summary.no_regression_count}/{summary.total_tasks} ({summary.no_regression_rate:.1%})")
        print(f"Avg steps: {summary.avg_steps:.1f}")
        print(f"Avg duration: {summary.avg_duration:.1f}s")
        print()
        print(f"Results saved to: {self.results_store.results_file}")
        print(f"Report saved to: {report_path}")
        
        return summary

    def run_single(
        self,
        task: Task,
        include_hints: bool = True,
    ) -> TaskResult:
        """
        Run evaluation on a single task.
        
        Args:
            task: Task to evaluate
            include_hints: Whether to provide relevant_files hints
            
        Returns:
            TaskResult with all metrics
        """
        # 1. Setup repository
        repo_path = self.repo_manager.clone(task.repo)
        self.repo_manager.reset(repo_path)  # Clean state
        
        # Only checkout specific commit for remote repos
        is_local = task.repo.startswith("local/")
        if not is_local:
            self.repo_manager.checkout(repo_path, task.base_commit)
        
        # 2. Run agent
        start_time = time.time()
        
        if RepoAgent is None:
            raise ImportError("Could not import RepoAgent from agent package")
        
        agent = RepoAgent(model=self.model, provider=self.provider, max_steps=self.max_steps)
        
        try:
            agent_result = agent.solve(
                task=task,
                repo_path=repo_path,
                include_hints=include_hints,
            )
            
            duration = time.time() - start_time
            
            # 3. Evaluate
            result = self.evaluator.evaluate(
                task=task,
                repo_path=repo_path,
                agent_patch=agent_result.patch,
                steps=agent_result.steps,
                duration=duration,
                explanation=agent_result.explanation,
            )
            
            if not agent_result.success:
                result.error = agent_result.error or "Agent did not submit"
            
        except Exception as e:
            duration = time.time() - start_time
            result = TaskResult(
                task_id=task.id,
                duration=duration,
                error=str(e),
                timestamp=datetime.now().isoformat(),
            )
        
        # 4. Reset repo for next task
        self.repo_manager.reset(repo_path)
        
        # 5. Save result
        self.results_store.save_result(result)
        
        return result

    def run_from_directory(
        self,
        tasks_dir: Path | str,
        include_hints: bool = True,
        skip_completed: bool = True,
    ) -> EvalSummary:
        """
        Run evaluation on all tasks in a directory.
        
        Args:
            tasks_dir: Directory containing task JSON files
            include_hints: Whether to provide hints to agent
            skip_completed: Skip already completed tasks
            
        Returns:
            EvalSummary with aggregate metrics
        """
        tasks = Task.load_all(tasks_dir)
        return self.run(tasks, include_hints=include_hints, skip_completed=skip_completed)