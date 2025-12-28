"""Tools available to the coding agent."""

import subprocess
import tempfile
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Result of code execution."""
    stdout: str
    stderr: str
    returncode: int

    @property
    def success(self) -> bool:
        return self.returncode == 0


def execute_python(code: str, timeout: int = 10) -> ExecutionResult:
    """Execute Python code in a subprocess and return the result."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                ["python", f.name],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return ExecutionResult(
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                stdout="",
                stderr=f"Execution timed out after {timeout} seconds",
                returncode=-1,
            )


# Tool definition for Claude API
TOOLS = [
    {
        "name": "execute_python",
        "description": "Execute Python code and return the output. Use this to test your solution.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute",
                }
            },
            "required": ["code"],
        },
    },
    {
        "name": "submit_solution",
        "description": "Submit your final solution code. Call this when you are confident your solution is correct.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The final Python code solution",
                }
            },
            "required": ["code"],
        },
    },
]


def run_tool(name: str, input_data: dict) -> str:
    """Execute a tool by name and return the result as a string."""
    if name == "execute_python":
        result = execute_python(input_data["code"])
        output = ""
        if result.stdout:
            output += f"stdout:\n{result.stdout}\n"
        if result.stderr:
            output += f"stderr:\n{result.stderr}\n"
        if not output:
            output = "(no output)"
        return output.strip()

    elif name == "submit_solution":
        # This is handled specially by the agent loop
        return "Solution submitted."

    else:
        return f"Unknown tool: {name}"
