# Coding Agent Eval

A minimal framework for building and evaluating coding agents on real-world repository tasks.

## Overview

This project provides:

1. **A minimal coding agent** that uses Claude to solve programming tasks
2. **A task collector** that extracts evaluation tasks from GitHub PRs
3. **An evaluation framework** for measuring agent performance on repository-level coding tasks

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/coding-agent-eval.git
cd coding-agent-eval
pip install -r requirements.txt
```

Set your API keys:

```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
export GITHUB_TOKEN="your-github-token"  # optional, for higher rate limits
```

## Quick Start

### 1. Simple Coding Agent

```python
from agent import CodingAgent

agent = CodingAgent(max_steps=10)
result = agent.solve(
    description="Write a function that returns the sum of two numbers.",
    function_signature="def add(a: int, b: int) -> int:",
    examples=[{"input": {"a": 2, "b": 3}, "expected": 5}],
)

print(f"Solution:\n{result.solution}")
```

### 2. Repository-Level Agent

```python
from agent import RepoAgent
from eval import Task

# Load a task
task = Task.load("eval/tasks/scikit-learn__scikit-learn-28280.json")

# Run agent on cloned repo
agent = RepoAgent(max_steps=30)
result = agent.solve(task, repo_path="/path/to/scikit-learn")

print(f"Success: {result.success}")
print(f"Patch:\n{result.patch}")
```

### 3. Collect Tasks from GitHub

```bash
# Collect 20 tasks from scikit-learn
python collect_tasks.py --max-prs 20

# Collect a specific PR
python collect_tasks.py --pr 28280

# Customize filters
python collect_tasks.py --max-files 3 --max-lines 200
```

## Project Structure

```
coding-agent-eval/
├── agent/                    # Agent implementations
│   ├── agent.py              # Simple coding agent
│   ├── repo_agent.py         # Repository-level agent
│   ├── repo_tools.py         # File/search/test tools
│   ├── prompts.py            # System prompts
│   ├── llm.py                # Claude API wrapper
│   └── tools.py              # Code execution tools
├── eval/                     # Evaluation framework
│   ├── task.py               # Task schema
│   ├── collector.py          # GitHub PR collector
│   ├── github_client.py      # GitHub API client
│   └── tasks/                # Collected task files
├── collect_tasks.py          # CLI for collecting tasks
├── test_agent.py             # Test simple agent
├── test_repo_agent.py        # Test repo agent
└── requirements.txt
```

## Agent Tools

### Simple Agent (CodingAgent)

| Tool | Description |
|------|-------------|
| `execute_python` | Run Python code and see output |
| `submit_solution` | Submit final solution |

### Repository Agent (RepoAgent)

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents |
| `write_file` | Create or modify files |
| `list_directory` | Explore repo structure |
| `search_code` | Grep for patterns in codebase |
| `run_tests` | Run pytest on specific tests |
| `run_command` | Run shell commands |
| `submit_patch` | Submit final solution |

## Task Schema

Tasks are stored as JSON files with this structure:

```json
{
  "id": "scikit-learn__scikit-learn-28280",
  "repo": "scikit-learn/scikit-learn",
  "base_commit": "abc123...",
  "issue_number": 28248,
  "issue_title": "Bug in cosine_similarity",
  "issue_body": "Description of the issue...",
  "pr_number": 28280,
  "gold_patch": "diff --git...",
  "fail_to_pass": ["test_cosine_similarity_float32"],
  "relevant_files": ["sklearn/metrics/pairwise.py"],
  "difficulty": "easy"
}
```

## Evaluation Metrics

When evaluating agents, we measure:

| Metric | Description |
|--------|-------------|
| `resolve_rate` | % of tasks where fail→pass tests now pass |
| `regression_rate` | % of tasks with no test regressions |
| `test_pass_rate` | % of individual tests passing |
| `steps` | Number of agent iterations |
| `diff_size` | Lines changed in solution |

## Supported Repositories

The framework is designed to work with Python repositories that use pytest. It has been tested with:

- scikit-learn/scikit-learn
- (more coming soon)

## Contributing

Contributions are welcome! Some ideas:

- Add support for more repositories
- Implement additional evaluation metrics
- Add more agent tools
- Improve task collection heuristics

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Inspired by [SWE-bench](https://www.swebench.com/) and the broader work on evaluating coding agents.
