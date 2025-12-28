"""GitHub API client for fetching PR and issue data."""

import os
import time
from dataclasses import dataclass

import requests


@dataclass
class PullRequest:
    """Simplified PR data."""

    number: int
    title: str
    body: str
    base_sha: str  # commit the PR is based on
    merge_commit_sha: str | None
    html_url: str
    created_at: str
    merged_at: str | None
    changed_files: int
    additions: int
    deletions: int
    linked_issue: int | None = None


@dataclass
class Issue:
    """Simplified issue data."""

    number: int
    title: str
    body: str
    html_url: str
    created_at: str


class GitHubClient:
    """Simple GitHub API client."""

    BASE_URL = "https://api.github.com"

    def __init__(self, token: str | None = None):
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.session = requests.Session()
        if self.token:
            self.session.headers["Authorization"] = f"token {self.token}"
        self.session.headers["Accept"] = "application/vnd.github.v3+json"

    def _get(self, endpoint: str, params: dict | None = None) -> dict | list:
        """Make a GET request with rate limit handling."""
        url = f"{self.BASE_URL}/{endpoint}"
        while True:
            response = self.session.get(url, params=params)

            # Handle rate limiting
            if response.status_code == 403:
                reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
                wait = max(reset_time - time.time(), 60)
                print(f"Rate limited. Waiting {wait:.0f}s...")
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response.json()

    def get_pull_request(self, owner: str, repo: str, pr_number: int) -> PullRequest:
        """Fetch a single PR with details."""
        data = self._get(f"repos/{owner}/{repo}/pulls/{pr_number}")

        # Try to find linked issue from body
        linked_issue = self._extract_linked_issue(data.get("body", "") or "")

        return PullRequest(
            number=data["number"],
            title=data["title"],
            body=data.get("body") or "",
            base_sha=data["base"]["sha"],
            merge_commit_sha=data.get("merge_commit_sha"),
            html_url=data["html_url"],
            created_at=data["created_at"],
            merged_at=data.get("merged_at"),
            changed_files=data.get("changed_files", 0),
            additions=data.get("additions", 0),
            deletions=data.get("deletions", 0),
            linked_issue=linked_issue,
        )

    def get_pull_requests(
        self,
        owner: str,
        repo: str,
        state: str = "closed",
        per_page: int = 30,
        max_pages: int = 10,
    ) -> list[PullRequest]:
        """Fetch multiple PRs (paginated)."""
        prs = []
        for page in range(1, max_pages + 1):
            data = self._get(
                f"repos/{owner}/{repo}/pulls",
                params={"state": state, "per_page": per_page, "page": page},
            )
            if not data:
                break

            for item in data:
                # Skip unmerged PRs
                if state == "closed" and not item.get("merged_at"):
                    continue

                # Need to fetch full PR for changed_files count
                pr = self.get_pull_request(owner, repo, item["number"])
                prs.append(pr)

            print(f"Fetched page {page}, total PRs: {len(prs)}")

        return prs

    def get_issue(self, owner: str, repo: str, issue_number: int) -> Issue:
        """Fetch a single issue."""
        data = self._get(f"repos/{owner}/{repo}/issues/{issue_number}")
        return Issue(
            number=data["number"],
            title=data["title"],
            body=data.get("body") or "",
            html_url=data["html_url"],
            created_at=data["created_at"],
        )

    def get_pr_diff(self, owner: str, repo: str, pr_number: int) -> str:
        """Fetch the diff/patch for a PR."""
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/pulls/{pr_number}"
        response = self.session.get(
            url, headers={"Accept": "application/vnd.github.v3.diff"}
        )
        response.raise_for_status()
        return response.text

    def get_pr_files(self, owner: str, repo: str, pr_number: int) -> list[str]:
        """Get list of files changed in a PR."""
        data = self._get(f"repos/{owner}/{repo}/pulls/{pr_number}/files")
        return [f["filename"] for f in data]

    def _extract_linked_issue(self, body: str) -> int | None:
        """Extract linked issue number from PR body."""
        import re

        # Common patterns: "Fixes #123", "Closes #123", "Resolves #123"
        patterns = [
            r"(?:fix|fixes|fixed|close|closes|closed|resolve|resolves|resolved)\s*#(\d+)",
            r"#(\d+)",  # fallback: any issue reference
        ]
        for pattern in patterns:
            match = re.search(pattern, body.lower())
            if match:
                return int(match.group(1))
        return None
