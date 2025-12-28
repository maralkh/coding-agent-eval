"""Task collector: Extract evaluation tasks from GitHub PRs."""

import re
from pathlib import Path

from .github_client import GitHubClient, PullRequest
from .task import Task, estimate_difficulty


class TaskCollector:
    """Collects and filters PRs to create evaluation tasks."""

    def __init__(
        self,
        owner: str = "scikit-learn",
        repo: str = "scikit-learn",
        token: str | None = None,
    ):
        self.owner = owner
        self.repo = repo
        self.client = GitHubClient(token=token)

    def collect(
        self,
        max_prs: int = 100,
        min_files: int = 1,
        max_files: int = 5,
        min_lines: int = 10,
        max_lines: int = 500,
        require_issue: bool = True,
        require_tests: bool = True,
    ) -> list[Task]:
        """
        Collect tasks from merged PRs.

        Args:
            max_prs: Maximum number of PRs to fetch
            min_files: Minimum files changed
            max_files: Maximum files changed
            min_lines: Minimum lines changed
            max_lines: Maximum lines changed
            require_issue: Only include PRs linked to an issue
            require_tests: Only include PRs that touch test files

        Returns:
            List of Task objects
        """
        print(f"Fetching PRs from {self.owner}/{self.repo}...")

        # Calculate pages needed
        per_page = 30
        max_pages = (max_prs // per_page) + 1

        prs = self.client.get_pull_requests(
            self.owner, self.repo, state="closed", per_page=per_page, max_pages=max_pages
        )

        print(f"Fetched {len(prs)} merged PRs")

        tasks = []
        for pr in prs:
            task = self._process_pr(
                pr,
                min_files=min_files,
                max_files=max_files,
                min_lines=min_lines,
                max_lines=max_lines,
                require_issue=require_issue,
                require_tests=require_tests,
            )
            if task:
                tasks.append(task)
                print(f"  ✓ PR #{pr.number}: {pr.title[:50]}...")

            if len(tasks) >= max_prs:
                break

        print(f"\nCollected {len(tasks)} tasks")
        return tasks

    def _process_pr(
        self,
        pr: PullRequest,
        min_files: int,
        max_files: int,
        min_lines: int,
        max_lines: int,
        require_issue: bool,
        require_tests: bool,
    ) -> Task | None:
        """Process a single PR and return a Task if it meets criteria."""

        # Filter by size
        total_lines = pr.additions + pr.deletions
        if not (min_files <= pr.changed_files <= max_files):
            return None
        if not (min_lines <= total_lines <= max_lines):
            return None

        # Get files changed
        files = self.client.get_pr_files(self.owner, self.repo, pr.number)

        # Filter: must touch Python files
        py_files = [f for f in files if f.endswith(".py")]
        if not py_files:
            return None

        # Filter: must touch test files (if required)
        test_files = [f for f in py_files if "test" in f.lower()]
        if require_tests and not test_files:
            return None

        # Get linked issue
        issue_number = pr.linked_issue
        if require_issue and not issue_number:
            return None

        # Fetch issue details
        if issue_number:
            try:
                issue = self.client.get_issue(self.owner, self.repo, issue_number)
                issue_title = issue.title
                issue_body = issue.body
                issue_url = issue.html_url
            except Exception:
                if require_issue:
                    return None
                issue_title = pr.title
                issue_body = pr.body
                issue_url = ""
        else:
            issue_number = pr.number  # Use PR as issue
            issue_title = pr.title
            issue_body = pr.body
            issue_url = pr.html_url

        # Get the diff
        gold_patch = self.client.get_pr_diff(self.owner, self.repo, pr.number)

        # Extract test names from diff
        fail_to_pass = self._extract_test_names(gold_patch, test_files)

        # Estimate difficulty
        difficulty = estimate_difficulty(pr.changed_files, total_lines)

        # Build task ID
        task_id = f"{self.repo}__{self.repo}-{pr.number}"

        return Task(
            id=task_id,
            repo=f"{self.owner}/{self.repo}",
            base_commit=pr.base_sha,
            issue_number=issue_number,
            issue_title=issue_title,
            issue_body=issue_body,
            pr_number=pr.number,
            pr_title=pr.title,
            gold_patch=gold_patch,
            fail_to_pass=fail_to_pass,
            pass_to_pass=[],  # Would need to run tests to determine
            relevant_files=py_files,
            difficulty=difficulty,
            created_at=pr.created_at,
            pr_url=pr.html_url,
            issue_url=issue_url,
        )

    def _extract_test_names(self, diff: str, test_files: list[str]) -> list[str]:
        """Extract test function names from diff."""
        test_names = []

        # Pattern for test function definitions
        pattern = r"^\+\s*def\s+(test_\w+)"

        for line in diff.split("\n"):
            match = re.match(pattern, line)
            if match:
                test_names.append(match.group(1))

        return test_names

    def save_tasks(self, tasks: list[Task], output_dir: Path | str) -> None:
        """Save tasks to JSON files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for task in tasks:
            path = output_dir / f"{task.id}.json"
            task.save(path)
            print(f"Saved: {path}")

    def collect_single_pr(self, pr_number: int) -> Task | None:
        """Collect a task from a specific PR number."""
        pr = self.client.get_pull_request(self.owner, self.repo, pr_number)

        return self._process_pr(
            pr,
            min_files=0,
            max_files=100,
            min_lines=0,
            max_lines=10000,
            require_issue=False,
            require_tests=False,
        )
