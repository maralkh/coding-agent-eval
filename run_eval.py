#!/usr/bin/env python
"""CLI script to run evaluations."""

import argparse
from pathlib import Path

from eval import Task, EvaluationRunner, ResultsStore


def main():
    parser = argparse.ArgumentParser(
        description="Run coding agent evaluation on tasks"
    )
    
    # Task input
    parser.add_argument(
        "--tasks",
        type=str,
        required=True,
        help="Path to task JSON file or directory of tasks",
    )
    
    # Output
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results",
        help="Output directory for results (default: results)",
    )
    
    # Agent configuration
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Model to use (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Maximum agent steps per task (default: 30)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per task in seconds (default: 600)",
    )
    
    # Evaluation options
    parser.add_argument(
        "--no-hints",
        action="store_true",
        help="Don't provide relevant_files hints to agent",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't skip already completed tasks",
    )
    
    # Cache
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Directory to cache cloned repositories",
    )
    
    # Report only mode
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only generate report from existing results",
    )

    args = parser.parse_args()
    
    # Report only mode
    if args.report_only:
        store = ResultsStore(args.output)
        results = store.load_results()
        if not results:
            print(f"No results found in {args.output}")
            return
        
        summary = store.get_summary()
        report_path = store.save_report()
        
        print("=== Summary ===")
        print(f"Total tasks: {summary.total_tasks}")
        print(f"Resolved: {summary.resolved_count} ({summary.resolve_rate:.1%})")
        print(f"No regression: {summary.no_regression_count} ({summary.no_regression_rate:.1%})")
        print()
        print(f"Report saved to: {report_path}")
        return
    
    # Load tasks
    tasks_path = Path(args.tasks)
    if tasks_path.is_file():
        tasks = [Task.load(tasks_path)]
    elif tasks_path.is_dir():
        tasks = Task.load_all(tasks_path)
    else:
        print(f"Error: {tasks_path} does not exist")
        return
    
    if not tasks:
        print("No tasks found")
        return
    
    print(f"Loaded {len(tasks)} tasks")
    
    # Create runner
    runner = EvaluationRunner(
        output_dir=args.output,
        cache_dir=args.cache_dir,
        model=args.model,
        max_steps=args.max_steps,
        timeout=args.timeout,
    )
    
    # Run evaluation
    runner.run(
        tasks=tasks,
        include_hints=not args.no_hints,
        skip_completed=not args.no_resume,
    )


if __name__ == "__main__":
    main()