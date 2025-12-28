"""Task schema for repository-level evaluation."""

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path


@dataclass
class Task:
    """A single evaluation task derived from a GitHub PR."""

    # Identity
    id: str
    repo: str

    # Git state
    base_commit: str  # commit before the PR

    # Problem definition
    issue_number: int
    issue_title: str
    issue_body: str
    pr_number: int
    pr_title: str

    # For evaluation (hidden from agent)
    gold_patch: str  # the actual solution diff
    fail_to_pass: list[str] = field(default_factory=list)  # tests that should start failing
    pass_to_pass: list[str] = field(default_factory=list)  # tests that must stay passing

    # Hints (can be withheld from agent)
    relevant_files: list[str] = field(default_factory=list)

    # Metadata
    difficulty: str = "medium"
    created_at: str = ""
    pr_url: str = ""
    issue_url: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def save(self, path: Path | str) -> None:
        """Save task to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> "Task":
        """Load task from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def load_all(cls, directory: Path | str) -> list["Task"]:
        """Load all tasks from a directory."""
        directory = Path(directory)
        tasks = []
        for path in sorted(directory.glob("*.json")):
            tasks.append(cls.load(path))
        return tasks


def estimate_difficulty(files_changed: int, lines_changed: int) -> str:
    """Estimate task difficulty based on PR size."""
    if files_changed <= 1 and lines_changed < 50:
        return "easy"
    elif files_changed <= 3 and lines_changed < 200:
        return "medium"
    else:
        return "hard"
