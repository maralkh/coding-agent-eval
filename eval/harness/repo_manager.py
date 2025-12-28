"""Repository management for evaluation."""

import subprocess
import shutil
from pathlib import Path


class RepoManager:
    """Manages repository cloning and state."""

    def __init__(self, cache_dir: Path | str | None = None, local_repos_dir: Path | str | None = None):
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "coding-agent-eval" / "repos"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _run_git(
        self, args: list[str], cwd: Path | None = None, check: bool = True
    ) -> subprocess.CompletedProcess:
        """Run a git command."""
        return subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=check,
        )

    def get_repo_dir(self, repo: str) -> Path:
        """Get the cache directory for a repo."""
        # repo is like "scikit-learn/scikit-learn"
        safe_name = repo.replace("/", "__")
        return self.cache_dir / safe_name

    def clone(self, repo: str, force_fresh: bool = False) -> Path:
        """
        Clone a repository (uses cache if exists).
        
        Args:
            repo: Repository in "owner/name" format
            force_fresh: If True, delete and re-clone
            
        Returns:
            Path to the cloned repository
        """
        repo_dir = self.get_repo_dir(repo)
        
        if force_fresh and repo_dir.exists():
            shutil.rmtree(repo_dir)
        
        if repo_dir.exists():
            print(f"Using cached repo: {repo_dir}")
            # Fetch latest
            try:
                self._run_git(["fetch", "--all"], cwd=repo_dir)
            except subprocess.CalledProcessError:
                pass  # OK if fetch fails (offline, etc.)
            return repo_dir
        
        # Clone fresh
        repo_url = f"https://github.com/{repo}.git"
        print(f"Cloning {repo}...")
        
        self._run_git([
            "clone",
            "--depth", "200",  # Shallow clone for speed
            repo_url,
            str(repo_dir),
        ])
        
        return repo_dir

    def checkout(self, repo_path: Path, commit: str) -> None:
        """
        Checkout a specific commit.
        
        Args:
            repo_path: Path to the repository
            commit: Commit SHA to checkout
        """
        # First, try to checkout directly
        result = self._run_git(
            ["checkout", commit],
            cwd=repo_path,
            check=False,
        )
        
        if result.returncode != 0:
            # Commit might not be in shallow clone, fetch it
            print(f"Fetching commit {commit[:8]}...")
            self._run_git(
                ["fetch", "--depth", "100", "origin", commit],
                cwd=repo_path,
                check=False,
            )
            # Try checkout again
            self._run_git(["checkout", commit], cwd=repo_path)

    def reset(self, repo_path: Path) -> None:
        """
        Hard reset repository to clean state.
        
        Discards all changes made by the agent.
        """
        self._run_git(["reset", "--hard"], cwd=repo_path)
        self._run_git(["clean", "-fd"], cwd=repo_path)

    def get_diff(self, repo_path: Path) -> str:
        """Get git diff of current changes."""
        result = self._run_git(["diff"], cwd=repo_path, check=False)
        return result.stdout

    def get_changed_files(self, repo_path: Path) -> list[str]:
        """Get list of changed files."""
        result = self._run_git(
            ["diff", "--name-only"],
            cwd=repo_path,
            check=False,
        )
        if result.stdout:
            return result.stdout.strip().split("\n")
        return []

    def get_diff_stats(self, repo_path: Path) -> dict:
        """Get diff statistics (lines added/removed)."""
        result = self._run_git(
            ["diff", "--shortstat"],
            cwd=repo_path,
            check=False,
        )
        
        stats = {"files": 0, "insertions": 0, "deletions": 0}
        
        if result.stdout:
            # Parse output like: "2 files changed, 10 insertions(+), 3 deletions(-)"
            import re
            
            files_match = re.search(r"(\d+) files? changed", result.stdout)
            if files_match:
                stats["files"] = int(files_match.group(1))
            
            ins_match = re.search(r"(\d+) insertions?", result.stdout)
            if ins_match:
                stats["insertions"] = int(ins_match.group(1))
            
            del_match = re.search(r"(\d+) deletions?", result.stdout)
            if del_match:
                stats["deletions"] = int(del_match.group(1))
        
        return stats
