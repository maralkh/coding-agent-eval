#!/usr/bin/env python
"""CLI script to collect evaluation tasks from GitHub PRs."""

import argparse
from pathlib import Path

from eval import TaskCollector


def main():
    parser = argparse.ArgumentParser(
        description="Collect evaluation tasks from GitHub PRs"
    )
    parser.add_argument(
        "--owner",
        default="scikit-learn",
        help="Repository owner (default: scikit-learn)",
    )
    parser.add_argument(
        "--repo",
        default="scikit-learn",
        help="Repository name (default: scikit-learn)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="eval/tasks",
        help="Output directory for task JSON files",
    )
    parser.add_argument(
        "--max-prs",
        type=int,
        default=20,
        help="Maximum number of tasks to collect (default: 20)",
    )
    parser.add_argument(
        "--min-files",
        type=int,
        default=1,
        help="Minimum files changed (default: 1)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=5,
        help="Maximum files changed (default: 5)",
    )
    parser.add_argument(
        "--min-lines",
        type=int,
        default=10,
        help="Minimum lines changed (default: 10)",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=500,
        help="Maximum lines changed (default: 500)",
    )
    parser.add_argument(
        "--no-require-issue",
        action="store_true",
        help="Don't require PRs to have linked issues",
    )
    parser.add_argument(
        "--no-require-tests",
        action="store_true",
        help="Don't require PRs to include test changes",
    )
    parser.add_argument(
        "--pr",
        type=int,
        help="Collect a single specific PR by number",
    )

    args = parser.parse_args()

    collector = TaskCollector(owner=args.owner, repo=args.repo)

    if args.pr:
        # Collect single PR
        print(f"Collecting PR #{args.pr}...")
        task = collector.collect_single_pr(args.pr)
        if task:
            output_dir = Path(args.output)
            collector.save_tasks([task], output_dir)
            print(f"\nTask saved to {output_dir / f'{task.id}.json'}")
        else:
            print("Failed to collect task from PR")
    else:
        # Collect multiple PRs
        tasks = collector.collect(
            max_prs=args.max_prs,
            min_files=args.min_files,
            max_files=args.max_files,
            min_lines=args.min_lines,
            max_lines=args.max_lines,
            require_issue=not args.no_require_issue,
            require_tests=not args.no_require_tests,
        )

        if tasks:
            collector.save_tasks(tasks, Path(args.output))
            print(f"\n{len(tasks)} tasks saved to {args.output}/")

            # Print summary
            print("\nDifficulty breakdown:")
            for diff in ["easy", "medium", "hard"]:
                count = sum(1 for t in tasks if t.difficulty == diff)
                print(f"  {diff}: {count}")
        else:
            print("No tasks collected")


if __name__ == "__main__":
    main()
