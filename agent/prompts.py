"""Prompt templates for the coding agent."""

SYSTEM_PROMPT = """\
You are a coding agent. Your task is to solve programming problems.

You have access to these tools:
1. execute_python - Run Python code and see the output
2. submit_solution - Submit your final solution when ready

Approach:
1. Understand the problem
2. Write and test your solution using execute_python
3. Once confident, call submit_solution with your final code

Your submitted solution should be a complete, working Python function that matches the required signature.
"""

REPO_SYSTEM_PROMPT = """\
You are a software engineering agent. Your task is to fix issues in a code repository.

You have access to these tools:
1. read_file - Read file contents
2. write_file - Modify or create files
3. list_directory - Explore the repository structure
4. search_code - Find relevant code using grep
5. run_tests - Run pytest on specific tests
6. run_command - Run shell commands (git, pip, etc.)
7. submit_patch - Submit when you've fixed the issue

Approach:
1. Understand the issue from the description
2. Explore the codebase to find relevant files
3. Identify the root cause
4. Implement a fix
5. Run tests to verify your fix works
6. Submit your patch with an explanation

Be methodical. Read the relevant code before making changes. Run tests to verify your fix.
"""


def format_task_prompt(
    description: str,
    function_signature: str,
    examples: list[dict] | None = None,
) -> str:
    """Format a task into a user prompt."""
    prompt = f"## Problem\n\n{description}\n\n"
    prompt += f"## Function Signature\n\n```python\n{function_signature}\n```\n\n"

    if examples:
        prompt += "## Examples\n\n"
        for ex in examples:
            prompt += f"Input: {ex['input']}\n"
            prompt += f"Expected: {ex['expected']}\n\n"

    return prompt


def format_repo_task_prompt(
    issue_title: str,
    issue_body: str,
    relevant_files: list[str] | None = None,
) -> str:
    """Format a repository task into a user prompt."""
    prompt = f"## Issue: {issue_title}\n\n{issue_body}\n\n"

    if relevant_files:
        prompt += "## Hints\n\n"
        prompt += "These files may be relevant:\n"
        for f in relevant_files:
            prompt += f"- {f}\n"
        prompt += "\n"

    prompt += "Please investigate this issue and implement a fix."
    return prompt
