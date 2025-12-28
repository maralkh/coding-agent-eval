"""Repository-level coding agent."""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .llm import LLMClient
from .repo_tools import REPO_TOOLS, run_repo_tool
from .prompts import REPO_SYSTEM_PROMPT, format_repo_task_prompt

# Import Task type for type hints (avoid circular import at runtime)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from eval.task import Task


@dataclass
class RepoAgentResult:
    """Result of a repository agent run."""
    success: bool
    patch: str  # git diff of changes made
    explanation: str
    steps: int
    messages: list[dict] = field(default_factory=list)
    error: str | None = None


class RepoAgent:
    """An agent that solves issues in code repositories."""

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        max_steps: int = 30,
    ):
        self.llm = LLMClient(model=model, provider=provider)
        self.max_steps = max_steps

    def solve(
        self,
        task: "Task",
        repo_path: str | Path,
        include_hints: bool = True,
    ) -> RepoAgentResult:
        """
        Solve a repository task.

        Args:
            task: Task object containing issue details
            repo_path: Path to the cloned repository
            include_hints: Whether to include relevant_files hints

        Returns:
            RepoAgentResult with patch and metadata
        """
        repo_path = Path(repo_path).resolve()

        # Format the prompt
        user_prompt = format_repo_task_prompt(
            issue_title=task.issue_title,
            issue_body=task.issue_body,
            relevant_files=task.relevant_files if include_hints else None,
        )

        messages = [{"role": "user", "content": user_prompt}]
        explanation = ""
        steps = 0

        while steps < self.max_steps:
            steps += 1

            # Get LLM response
            response = self.llm.chat(
                messages=messages,
                system=REPO_SYSTEM_PROMPT,
                tools=REPO_TOOLS,
            )

            # Process response content
            assistant_content = []
            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

            messages.append({"role": "assistant", "content": assistant_content})

            # Check for tool use
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    # Check if this is the final submission
                    if block.name == "submit_patch":
                        explanation = block.input.get("explanation", "")
                        patch = self._get_git_diff(repo_path)
                        return RepoAgentResult(
                            success=True,
                            patch=patch,
                            explanation=explanation,
                            steps=steps,
                            messages=messages,
                        )

                    # Execute tool
                    result = run_repo_tool(block.name, block.input, str(repo_path))
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            # Add tool results and continue
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            else:
                # No tool calls - might be stuck
                break

            if response.stop_reason == "end_turn" and not tool_results:
                break

        # Reached max steps without submitting
        patch = self._get_git_diff(repo_path)
        return RepoAgentResult(
            success=False,
            patch=patch,
            explanation="",
            steps=steps,
            messages=messages,
            error="Max steps reached without submission",
        )

    def solve_from_dict(
        self,
        task_dict: dict,
        repo_path: str | Path,
        include_hints: bool = True,
    ) -> RepoAgentResult:
        """
        Solve a task from a dictionary (convenience method).

        Args:
            task_dict: Dictionary with issue_title, issue_body, relevant_files
            repo_path: Path to the cloned repository
            include_hints: Whether to include relevant_files hints
        """
        # Create a minimal task-like object
        class TaskLike:
            def __init__(self, d):
                self.issue_title = d.get("issue_title", "")
                self.issue_body = d.get("issue_body", "")
                self.relevant_files = d.get("relevant_files", [])

        return self.solve(TaskLike(task_dict), repo_path, include_hints)

    def _get_git_diff(self, repo_path: Path) -> str:
        """Get the git diff of changes made to the repository."""
        try:
            result = subprocess.run(
                ["git", "diff"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout
        except Exception as e:
            return f"Error getting diff: {e}"