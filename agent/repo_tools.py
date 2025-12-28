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
    # Normalize path - handle absolute paths by making them relative
    path = path.lstrip("/")
    repo_root_stripped = repo_root.lstrip("/")
    if path.startswith(repo_root_stripped):
        path = path[len(repo_root_stripped):].lstrip("/")
    
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
    original_path = path
    
    # Normalize path - handle absolute paths by making them relative
    path = path.lstrip("/")
    repo_root_stripped = repo_root.lstrip("/")
    if path.startswith(repo_root_stripped):
        path = path[len(repo_root_stripped):].lstrip("/")
    
    full_path = Path(repo_root) / path
    
    # Debug output
    print(f"      [write_file debug] original={original_path}")
    print(f"      [write_file debug] normalized={path}")
    print(f"      [write_file debug] full_path={full_path}")
    
    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        return f"Successfully wrote to {path} ({len(content)} chars)"
    except Exception as e:
        return f"Error writing file: {e}"


def str_replace_in_file(path: str, old_str: str, new_str: str, repo_root: str) -> str:
    """Replace a specific string in a file. The old_str must appear exactly once."""
    # Normalize path
    path = path.lstrip("/")
    repo_root_stripped = repo_root.lstrip("/")
    if path.startswith(repo_root_stripped):
        path = path[len(repo_root_stripped):].lstrip("/")
    
    full_path = Path(repo_root) / path
    
    if not full_path.exists():
        return f"Error: File not found: {path}"
    
    try:
        content = full_path.read_text()
        
        # Count occurrences
        count = content.count(old_str)
        
        if count == 0:
            # Show a preview of the file to help debug
            lines = content.split('\n')
            preview = '\n'.join(lines[:20])
            return f"Error: String not found in {path}. File starts with:\n{preview}\n..."
        elif count > 1:
            return f"Error: String appears {count} times in {path}. It must appear exactly once. Make the search string more specific."
        
        # Replace
        new_content = content.replace(old_str, new_str)
        full_path.write_text(new_content)
        
        # Count lines changed
        old_lines = len(old_str.split('\n'))
        new_lines = len(new_str.split('\n'))
        
        return f"Successfully replaced {old_lines} line(s) with {new_lines} line(s) in {path}"
    except Exception as e:
        return f"Error: {e}"


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
    
    # Check if file_pattern is a specific file path (contains /) or a glob pattern
    if '/' in file_pattern and '*' not in file_pattern:
        # It's a specific file path - search only that file
        file_path = Path(repo_root) / file_pattern
        if not file_path.exists():
            return f"Error: File not found: {file_pattern}"
        cmd = f"grep -n '{pattern}' '{file_pattern}'"
        result = run_command(cmd, cwd=repo_root, timeout=30)
        if result.returncode == 0:
            # Prepend filename to results for consistency
            lines = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    lines.append(f"{file_pattern}:{line}")
            if len(lines) > 50:
                return "\n".join(lines[:50]) + f"\n... and {len(lines) - 50} more matches"
            return "\n".join(lines)
        elif result.returncode == 1:
            return "No matches found"
        else:
            return f"Search error: {result.stderr}"
    else:
        # It's a glob pattern - use --include
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
        "description": "Write content to a file in the repository. Creates parent directories if needed. WARNING: This overwrites the entire file. For small edits, use str_replace_in_file instead.",
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
        "name": "str_replace_in_file",
        "description": "Replace a specific string in a file with new content. The old_str must appear exactly once in the file. This is the preferred way to make targeted edits.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file relative to repository root",
                },
                "old_str": {
                    "type": "string",
                    "description": "The exact string to replace (must appear exactly once in the file)",
                },
                "new_str": {
                    "type": "string",
                    "description": "The new string to replace it with",
                },
            },
            "required": ["path", "old_str", "new_str"],
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
        "description": "Search for a pattern in the codebase using grep. Returns matching lines with file paths and line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (text or regex)",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Either a glob pattern (e.g., '*.py') or a specific file path (e.g., 'src/module.py'). Default: *.py",
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
    # Debug: print tool name
    print(f"    [Tool: {name}]", end="")
    
    if name == "read_file":
        path = input_data.get("path") or input_data.get("file_path") or input_data.get("file")
        if not path:
            return "Error: Missing 'path' parameter"
        print(f" path={path}")
        result = read_file(path, repo_root)
        # Debug: show preview of result
        preview = result[:100].replace('\n', '\\n') if len(result) > 100 else result.replace('\n', '\\n')
        print(f"      [Result preview: {preview}...]")
        return result

    elif name == "write_file":
        path = input_data.get("path") or input_data.get("file_path") or input_data.get("file")
        content = input_data.get("content") or input_data.get("contents") or input_data.get("text") or ""
        
        if not path:
            return "Error: Missing 'path' parameter"
        
        print(f" path={path} ({len(content)} chars)")
        result = write_file(path, content, repo_root)
        print(f"      [Result: {result}]")
        return result

    elif name == "str_replace_in_file":
        # Debug: show what parameters we received
        print(f" keys={list(input_data.keys())}")
        
        # Handle different parameter names the model might use
        path = input_data.get("path") or input_data.get("file_path") or input_data.get("file")
        old_str = input_data.get("old_str") or input_data.get("old") or input_data.get("search") or input_data.get("find") or input_data.get("original")
        new_str = input_data.get("new_str") or input_data.get("new") or input_data.get("replace") or input_data.get("replacement")
        
        print(f"      path={path}, old_str={len(old_str) if old_str else 0} chars, new_str={len(new_str) if new_str else 0} chars")
        
        if not path:
            return "Error: Missing 'path' parameter"
        if old_str is None:
            return f"Error: Missing 'old_str' parameter. Got keys: {list(input_data.keys())}"
        if new_str is None:
            return f"Error: Missing 'new_str' parameter. Got keys: {list(input_data.keys())}"
        
        result = str_replace_in_file(path, old_str, new_str, repo_root)
        print(f"      [Result: {result}]")
        return result

    elif name == "list_directory":
        path = input_data.get("path") or input_data.get("dir") or input_data.get("directory") or ""
        print(f" path={path}")
        result = list_directory(path, repo_root)
        print(f"      [Result: {result[:200]}...]" if len(result) > 200 else f"      [Result: {result}]")
        return result

    elif name == "search_code":
        pattern = input_data.get("pattern") or input_data.get("query") or input_data.get("search")
        file_pattern = input_data.get("file_pattern") or input_data.get("glob") or "*.py"
        print(f" pattern='{pattern}' file_pattern='{file_pattern}'")
        if not pattern:
            return "Error: Missing 'pattern' parameter"
        result = search_code(pattern, repo_root, file_pattern)
        preview = result[:200] if len(result) > 200 else result
        print(f"      [Result: {preview}...]")
        return result

    elif name == "run_tests":
        test_path = input_data.get("test_path") or input_data.get("path") or input_data.get("tests")
        print(f" test_path={test_path}")
        if not test_path:
            return "Error: Missing 'test_path' parameter"
        return run_tests(test_path, repo_root)

    elif name == "run_command":
        command = input_data.get("command") or input_data.get("cmd")
        print(f" command={command}")
        if not command:
            return "Error: Missing 'command' parameter"
        result = run_command(command, cwd=repo_root)
        return str(result)

    elif name == "submit_patch":
        return "Patch submitted."

    else:
        return f"Unknown tool: {name}"