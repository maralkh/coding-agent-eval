"""Repository-level tools for the coding agent."""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CommandResult:
    """Result of a command execution."""
    stdout: str
    stderr: str
    returncode: int

    @property
    def success(self) -> bool:
        return self.returncode == 0

    def __str__(self) -> str:
        output = ""
        if self.stdout:
            output += self.stdout
        if self.stderr:
            output += f"\n[stderr]\n{self.stderr}"
        return output.strip() or "(no output)"


def run_command(cmd: str, cwd: str | None = None, timeout: int = 60) -> CommandResult:
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return CommandResult(
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
        )
    except subprocess.TimeoutExpired:
        return CommandResult(
            stdout="",
            stderr=f"Command timed out after {timeout} seconds",
            returncode=-1,
        )


def read_file(path: str, repo_root: str) -> str:
    """Read a file from the repository."""
    full_path = Path(repo_root) / path
    if not full_path.exists():
        return f"Error: File not found: {path}"
    if not full_path.is_file():
        return f"Error: Not a file: {path}"
    try:
        return full_path.read_text()
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(path: str, content: str, repo_root: str) -> str:
    """Write content to a file in the repository."""
    full_path = Path(repo_root) / path
    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def list_directory(path: str, repo_root: str) -> str:
    """List contents of a directory."""
    full_path = Path(repo_root) / path if path else Path(repo_root)
    if not full_path.exists():
        return f"Error: Directory not found: {path}"
    if not full_path.is_dir():
        return f"Error: Not a directory: {path}"

    try:
        entries = sorted(full_path.iterdir())
        lines = []
        for entry in entries[:100]:  # Limit to 100 entries
            rel_path = entry.relative_to(repo_root)
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{rel_path}{suffix}")
        if len(entries) > 100:
            lines.append(f"... and {len(entries) - 100} more")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing directory: {e}"


def search_code(pattern: str, repo_root: str, file_pattern: str = "*.py") -> str:
    """Search for a pattern in the codebase using grep."""
    cmd = f"grep -rn --include='{file_pattern}' '{pattern}' ."
    result = run_command(cmd, cwd=repo_root, timeout=30)
    if result.returncode == 0:
        lines = result.stdout.strip().split("\n")
        if len(lines) > 50:
            return "\n".join(lines[:50]) + f"\n... and {len(lines) - 50} more matches"
        return result.stdout
    elif result.returncode == 1:
        return "No matches found"
    else:
        return f"Search error: {result.stderr}"


def run_tests(test_path: str, repo_root: str, timeout: int = 120) -> str:
    """Run pytest on a specific test file or directory."""
    cmd = f"python -m pytest {test_path} -v --tb=short"
    result = run_command(cmd, cwd=repo_root, timeout=timeout)
    return str(result)


# Tool definitions for Claude API
REPO_TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file from the repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file relative to repository root",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file in the repository. Creates parent directories if needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file relative to repository root",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_directory",
        "description": "List files and subdirectories in a directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to directory relative to repository root (empty for root)",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_code",
        "description": "Search for a pattern in the codebase using grep. Returns matching lines with file paths.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (text or regex)",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File glob pattern to search in (default: *.py)",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "run_tests",
        "description": "Run pytest on a specific test file or test function.",
        "input_schema": {
            "type": "object",
            "properties": {
                "test_path": {
                    "type": "string",
                    "description": "Path to test file or specific test (e.g., 'tests/test_foo.py' or 'tests/test_foo.py::test_bar')",
                }
            },
            "required": ["test_path"],
        },
    },
    {
        "name": "run_command",
        "description": "Run a shell command in the repository root. Use for git, pip, or other commands.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to run",
                }
            },
            "required": ["command"],
        },
    },
    {
        "name": "submit_patch",
        "description": "Submit your final solution. Call this when you have fixed the issue.",
        "input_schema": {
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "string",
                    "description": "Brief explanation of what you changed and why",
                }
            },
            "required": ["explanation"],
        },
    },
]


def run_repo_tool(name: str, input_data: dict, repo_root: str) -> str:
    """Execute a repository tool and return the result."""
    if name == "read_file":
        return read_file(input_data["path"], repo_root)

    elif name == "write_file":
        return write_file(input_data["path"], input_data["content"], repo_root)

    elif name == "list_directory":
        return list_directory(input_data.get("path", ""), repo_root)

    elif name == "search_code":
        return search_code(
            input_data["pattern"],
            repo_root,
            input_data.get("file_pattern", "*.py"),
        )

    elif name == "run_tests":
        return run_tests(input_data["test_path"], repo_root)

    elif name == "run_command":
        result = run_command(input_data["command"], cwd=repo_root)
        return str(result)

    elif name == "submit_patch":
        return "Patch submitted."

    else:
        return f"Unknown tool: {name}"
