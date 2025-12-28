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
You are a software engineering agent. Your task is to fix bugs in code repositories.

CRITICAL RULES:
1. You MUST use tools to complete tasks - do not just describe what to do
2. Use str_replace_in_file for ALL code changes (not write_file)
3. The old_str in str_replace_in_file must match the file EXACTLY (including whitespace)
4. Always call submit_patch when finished

Available tools:
- read_file: Read file contents
- str_replace_in_file: Replace exact string in file (USE THIS FOR FIXES)
- list_directory: List files in directory
- search_code: Search for patterns in code
- run_tests: Run pytest
- run_command: Run shell commands
- submit_patch: Submit your fix (call this when done)

Workflow:
1. read_file to see the buggy code
2. str_replace_in_file to fix it (old_str must match exactly)
3. submit_patch with explanation

Example str_replace_in_file call:
- path: "path/to/file.py"
- old_str: "    return x + 1  # bug"
- new_str: "    return x  # fixed"

DO NOT use write_file for fixes. Use str_replace_in_file.
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