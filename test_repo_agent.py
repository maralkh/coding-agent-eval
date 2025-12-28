#!/usr/bin/env python
"""Test the RepoAgent with a collected task."""

import argparse
import subprocess
import tempfile
from pathlib import Path

from agent import RepoAgent
from eval import Task


def setup_repo(task: Task, target_dir: Path) -> Path:
    """Clone the repo and checkout the base commit."""
    repo_url = f"https://github.com/{task.repo}.git"
    repo_dir = target_dir / task.repo.split("/")[-1]

    print(f"Cloning {task.repo}...")
    subprocess.run(
        ["git", "clone", "--depth", "100", repo_url, str(repo_dir)],
        check=True,
        capture_output=True,
    )

    print(f"Checking out {task.base_commit[:8]}...")
    subprocess.run(
        ["git", "checkout", task.base_commit],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )

    return repo_dir


def main():
    parser = argparse.ArgumentParser(description="Test RepoAgent on a task")
    parser.add_argument(
        "task_file",
        help="Path to task JSON file",
    )
    parser.add_argument(
        "--repo-dir",
        help="Path to existing repo (skip clone)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum agent steps (default: 20)",
    )
    parser.add_argument(
        "--no-hints",
        action="store_true",
        help="Don't provide relevant_files hints",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Model to use",
    )

    args = parser.parse_args()

    # Load task
    task = Task.load(args.task_file)
    print(f"Loaded task: {task.id}")
    print(f"Issue: {task.issue_title}")
    print(f"Difficulty: {task.difficulty}")
    print()

    # Setup repo
    if args.repo_dir:
        repo_dir = Path(args.repo_dir)
    else:
        # Clone to temp directory
        tmp_dir = Path(tempfile.mkdtemp())
        repo_dir = setup_repo(task, tmp_dir)

    print(f"Repository: {repo_dir}")
    print()

    # Run agent
    agent = RepoAgent(model=args.model, max_steps=args.max_steps)

    print("Running agent...")
    print("-" * 50)

    result = agent.solve(
        task=task,
        repo_path=repo_dir,
        include_hints=not args.no_hints,
    )

    print("-" * 50)
    print()
    print(f"Success: {result.success}")
    print(f"Steps: {result.steps}")
    print(f"Explanation: {result.explanation}")
    print()

    if result.patch:
        print("Generated patch:")
        print("-" * 50)
        print(result.patch[:2000])
        if len(result.patch) > 2000:
            print(f"... ({len(result.patch)} chars total)")
    else:
        print("No changes made")

    if result.error:
        print(f"\nError: {result.error}")


if __name__ == "__main__":
    main()
