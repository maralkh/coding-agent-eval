# Coding Agent Evaluation Framework

A framework for building, testing, and evaluating coding agents on real-world repository tasks derived from GitHub pull requests.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Data Collection](#data-collection)
5. [Task Schema](#task-schema)
6. [Agent Implementation](#agent-implementation)
7. [Evaluation Harness](#evaluation-harness)
8. [Metrics & Debugging](#metrics--debugging)
9. [Running Evaluations](#running-evaluations)
10. [Results & Benchmarking](#results--benchmarking)
11. [Known Issues & Solutions](#known-issues--solutions)
12. [Future Work](#future-work)

---

## Overview

### Goal

Build an evaluation framework to measure how well coding agents can solve real-world software engineering tasks. Tasks are derived from merged GitHub pull requests, providing:
- Real bug reports and feature requests (issues)
- Ground truth solutions (the merged PR)
- Test cases that verify correctness

### Inspiration

This project is inspired by [SWE-bench](https://www.swebench.com/), which evaluates AI systems on their ability to resolve GitHub issues.

### Key Features

- **Multi-provider LLM support**: Anthropic, OpenAI (including o-series), Groq, Ollama
- **Automated task collection**: Extract evaluation tasks from GitHub PRs
- **Repository-level agent**: Tools for reading, writing, searching, and testing code
- **Comprehensive metrics**: Tool usage, patch quality, failure analysis
- **Results tracking**: JSON output with run history and benchmarking

---

## Architecture

```
coding-agent-eval/
├── agent/                      # Agent implementations
│   ├── __init__.py
│   ├── agent.py                # Simple coding agent (standalone tasks)
│   ├── repo_agent.py           # Repository-level agent
│   ├── repo_tools.py           # File/search/test tools
│   ├── prompts.py              # System prompts
│   ├── llm.py                  # Multi-provider LLM client
│   └── tools.py                # Code execution tools
│
├── eval/                       # Evaluation framework
│   ├── __init__.py
│   ├── task.py                 # Task schema
│   ├── collector.py            # GitHub PR collector
│   ├── github_client.py        # GitHub API client
│   ├── tasks/                  # Collected task JSON files
│   └── harness/                # Evaluation harness
│       ├── __init__.py
│       ├── evaluator.py        # Test runner & evaluation
│       ├── repo_manager.py     # Git repository management
│       ├── results.py          # Results storage
│       ├── runner.py           # Main evaluation orchestrator
│       └── metrics.py          # Debug metrics & analysis
│
├── results/                    # Output directory
│   ├── runs.jsonl              # Summary of all runs
│   └── *.json                  # Detailed per-run results
│
├── collect_tasks.py            # CLI: Collect tasks from GitHub
├── run_eval.py                 # CLI: Run batch evaluations
├── test_e2e.py                 # CLI: End-to-end testing
└── requirements.txt
```

---

## Installation

### Requirements

- Python 3.10+
- Git

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/coding-agent-eval.git
cd coding-agent-eval
pip install -r requirements.txt
```

### API Keys

```bash
# Required for agent (at least one)
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"
export GROQ_API_KEY="your-groq-key"

# Optional: for higher GitHub rate limits
export GITHUB_TOKEN="your-github-token"
```

---

## Data Collection

### Overview

Tasks are collected from merged GitHub pull requests. Each PR becomes an evaluation task where:
- The **issue** describes the problem to solve
- The **base commit** is the starting state
- The **gold patch** (merged PR diff) is the ground truth solution
- **Test cases** verify correctness

### Collector Pipeline

```
GitHub PR → Filter → Extract Issue → Extract Diff → Identify Tests → Save Task
```

### Usage

```bash
# Collect 20 tasks from scikit-learn
python collect_tasks.py --max-prs 20

# Collect from a specific PR
python collect_tasks.py --pr 28280

# Customize filters
python collect_tasks.py \
    --owner scikit-learn \
    --repo scikit-learn \
    --max-files 3 \
    --max-lines 200 \
    --output eval/tasks
```

### Filtering Criteria

| Filter | Default | Description |
|--------|---------|-------------|
| `--min-files` | 1 | Minimum files changed |
| `--max-files` | 5 | Maximum files changed |
| `--min-lines` | 10 | Minimum lines changed |
| `--max-lines` | 500 | Maximum lines changed |
| `--require-issue` | Yes | PR must link to an issue |
| `--require-tests` | Yes | PR must include test changes |

### Implementation

**`eval/collector.py`**: Main collector class
- Fetches merged PRs via GitHub API
- Filters by size and content
- Extracts linked issues
- Parses test names from diff
- Estimates difficulty

**`eval/github_client.py`**: GitHub API wrapper
- Rate limit handling
- Pagination support
- Diff fetching

---

## Task Schema

Tasks are stored as JSON files with this structure:

```json
{
  "id": "scikit-learn__scikit-learn-32932",
  "repo": "scikit-learn/scikit-learn",
  "base_commit": "b0ba8b029c298e0cc545206d2df4757be0ec2ac2",
  
  "issue_number": 32927,
  "issue_title": "LogisticRegression: Conflicting warnings...",
  "issue_body": "When I use LogisticRegression(penalty=None)...",
  
  "pr_number": 32932,
  "pr_title": "FIX Avoid LogisticRegression spurious warning",
  "gold_patch": "diff --git a/sklearn/linear_model/_logistic.py...",
  
  "fail_to_pass": ["test_c_inf_no_warning"],
  "pass_to_pass": [],
  
  "relevant_files": ["sklearn/linear_model/_logistic.py"],
  "difficulty": "easy",
  
  "created_at": "2025-12-23T09:05:22Z",
  "pr_url": "https://github.com/scikit-learn/scikit-learn/pull/32932",
  "issue_url": "https://github.com/scikit-learn/scikit-learn/issues/32927"
}
```

### Fields

| Field | Description |
|-------|-------------|
| `id` | Unique identifier |
| `repo` | GitHub repository (owner/name) |
| `base_commit` | Commit SHA before the PR (starting state) |
| `issue_*` | Problem description from the linked issue |
| `pr_*` | Pull request metadata |
| `gold_patch` | The merged diff (ground truth solution) |
| `fail_to_pass` | Tests that should fail before fix, pass after |
| `pass_to_pass` | Tests that should always pass (regression check) |
| `relevant_files` | Hint: files likely relevant to the fix |
| `difficulty` | Estimated difficulty (easy/medium/hard) |

### Difficulty Estimation

```python
def estimate_difficulty(files_changed: int, lines_changed: int) -> str:
    if files_changed <= 1 and lines_changed < 50:
        return "easy"
    elif files_changed <= 3 and lines_changed < 200:
        return "medium"
    else:
        return "hard"
```

---

## Agent Implementation

### Overview

The agent uses an LLM with tool use to explore a repository, understand the issue, and implement a fix.

### Components

#### LLM Client (`agent/llm.py`)

Multi-provider support:

| Provider | Models | Notes |
|----------|--------|-------|
| Anthropic | claude-sonnet-4-20250514, etc. | Native format |
| OpenAI | gpt-4o, gpt-4o-mini, o4-mini, etc. | o-series uses `max_completion_tokens` |
| Groq | llama-3.1-70b-versatile, etc. | OpenAI-compatible |
| Ollama | Any local model | OpenAI-compatible |

Auto-detection from model name:
```python
client = LLMClient(model="gpt-4o")  # Detects OpenAI
client = LLMClient(model="claude-sonnet-4-20250514")  # Detects Anthropic
client = LLMClient(provider="ollama", model="llama3")  # Explicit
```

#### Repository Agent (`agent/repo_agent.py`)

Main agent loop:
1. Receive task (issue description, hints)
2. Call LLM with available tools
3. Execute tool calls
4. Repeat until submission or max steps

```python
agent = RepoAgent(model="gpt-4o", max_steps=30)
result = agent.solve(task, repo_path, include_hints=True)
```

#### Tools (`agent/repo_tools.py`)

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents |
| `write_file` | Create or overwrite a file |
| `str_replace_in_file` | **Preferred**: Replace specific text (must appear exactly once) |
| `list_directory` | List files and subdirectories |
| `search_code` | Grep for patterns (supports file paths or glob patterns) |
| `run_tests` | Run pytest on specific tests |
| `run_command` | Run arbitrary shell commands |
| `submit_patch` | Submit final solution |

##### str_replace_in_file (Critical)

This tool is essential for making targeted edits without rewriting entire files:

```python
# Tool call
{
    "name": "str_replace_in_file",
    "input": {
        "path": "sklearn/linear_model/_logistic.py",
        "old_str": "        if penalty is None:",
        "new_str": "        if self.penalty is None:"
    }
}
```

Validation:
- `old_str` must appear **exactly once** in the file
- Shows helpful error if not found or appears multiple times
- Preserves exact whitespace/indentation

#### System Prompt (`agent/prompts.py`)

Key instructions for the agent:
```
CRITICAL RULES:
1. You MUST use tools to complete tasks
2. Use str_replace_in_file for ALL code changes (not write_file)
3. The old_str must match the file EXACTLY (including whitespace)
4. Always call submit_patch when finished
```

---

## Evaluation Harness

### Overview

The harness manages the evaluation pipeline:
1. Clone/setup repository
2. Checkout base commit
3. Run agent
4. Apply patch
5. Run tests
6. Compute metrics
7. Save results

### Components

#### Repository Manager (`eval/harness/repo_manager.py`)

```python
manager = RepoManager(cache_dir="~/.cache/coding-agent-eval/repos")

# Clone (uses cache if exists)
repo_path = manager.clone("scikit-learn/scikit-learn")

# Reset to clean state
manager.reset(repo_path)

# Checkout specific commit
manager.checkout(repo_path, "abc123...")

# Get changes
diff = manager.get_diff(repo_path)
files = manager.get_changed_files(repo_path)
```

#### Evaluator (`eval/harness/evaluator.py`)

Runs tests and computes metrics:

```python
evaluator = Evaluator(timeout=300)

# Run specific tests
result = evaluator.run_pytest(repo_path, ["test_file.py::test_func"])

# Full evaluation
task_result = evaluator.evaluate(
    task=task,
    repo_path=repo_path,
    agent_patch=patch,
    steps=10,
    duration=45.0,
)
```

#### Runner (`eval/harness/runner.py`)

Main orchestrator:

```python
runner = EvaluationRunner(
    output_dir="results",
    cache_dir="~/.cache/repos",
    model="gpt-4o",
    max_steps=30,
    timeout=600,
)

# Run on multiple tasks
summary = runner.run(tasks, include_hints=True, skip_completed=True)

# Run single task
result = runner.run_single(task, include_hints=True)
```

---

## Metrics & Debugging

### Overview

The metrics module (`eval/harness/metrics.py`) provides detailed analysis of agent behavior.

### Tool Usage Metrics

```python
@dataclass
class ToolUsageMetrics:
    total_calls: int
    calls_by_tool: dict          # {"read_file": 3, "str_replace": 1, ...}
    tool_sequence: list          # ["read_file", "search_code", ...]
    
    read_relevant_files: bool    # Did it read the hint files?
    used_str_replace: bool       # Used preferred edit tool?
    used_write_file: bool        # Used full file rewrite?
    ran_tests: bool              # Ran any tests?
    submitted: bool              # Called submit_patch?
    
    tool_errors: list            # Error messages from failed tool calls
```

### Patch Quality Metrics

```python
@dataclass
class PatchQualityMetrics:
    files_changed: list
    gold_files_touched: list
    correct_files_touched: bool
    extra_files_touched: list
    missing_files: list
    
    lines_added: int
    lines_removed: int
    
    similarity_score: float      # 0-1 similarity to gold patch
    patch_too_large: bool        # Rewrote whole file?
```

### Failure Analysis

```python
@dataclass
class FailureAnalysis:
    hit_max_steps: bool
    agent_submitted: bool
    
    no_changes_made: bool
    wrong_files_modified: bool
    patch_too_large: bool
    tool_errors_occurred: bool
    model_got_stuck: bool
    
    failure_reasons: list        # Human-readable explanations
```

### Debug Report

```
=== Debug Report: scikit-learn__scikit-learn-32932 ===

## Result
  Resolved: False
  Tests: 0 passed, 0 failed

## Tool Usage
  Total calls: 12
  Calls by tool: {'read_file': 4, 'search_code': 4, 'str_replace_in_file': 1, ...}
  Sequence: list_directory -> read_file -> search_code -> ...
  Read relevant files: True
  Used str_replace: True
  Submitted: True
  Tool errors: 0

## Patch Quality
  Files changed: ['sklearn/linear_model/_logistic.py']
  Correct files touched: True
  Lines: +1 -1
  Similarity to gold: 24.0%

## Agent's Patch
  diff --git a/sklearn/linear_model/_logistic.py ...
  -        if penalty is None:
  +        if self.penalty is None:

## Failure Analysis
  Hit max steps: False
  Agent submitted: True
  No obvious issues found
```

---

## Running Evaluations

### End-to-End Test

```bash
# Basic run
python test_e2e.py --task eval/tasks/scikit-learn__scikit-learn-32932.json

# With specific provider/model
python test_e2e.py \
    --task eval/tasks/scikit-learn__scikit-learn-32932.json \
    --provider openai \
    --model o4-mini \
    --max-steps 15

# Skip agent (just test setup)
python test_e2e.py --task ... --skip-agent

# View run history
python test_e2e.py --summary
```

### Batch Evaluation

```bash
# Run all tasks in a directory
python run_eval.py --tasks eval/tasks/ --output results/

# With custom settings
python run_eval.py \
    --tasks eval/tasks/ \
    --model gpt-4o \
    --max-steps 30 \
    --timeout 600 \
    --no-hints

# Resume interrupted run
python run_eval.py --tasks eval/tasks/ --output results/  # Skips completed

# Generate report only
python run_eval.py --tasks eval/tasks/ --output results/ --report-only
```

---

## Results & Benchmarking

### Output Structure

```
results/
├── runs.jsonl                                              # Summary log
├── scikit-learn__scikit-learn-32932_o4-mini_20241228_143022.json
├── scikit-learn__scikit-learn-32932_gpt-4o_20241228_144512.json
└── report.md                                               # Markdown report
```

### Run Summary (runs.jsonl)

```json
{"timestamp": "2024-12-28T14:30:22", "task_id": "scikit-learn__scikit-learn-32932", "provider": "openai", "model": "o4-mini", "resolved": false, "steps": 12, "similarity": 0.24}
{"timestamp": "2024-12-28T14:45:12", "task_id": "scikit-learn__scikit-learn-32932", "provider": "openai", "model": "gpt-4o", "resolved": true, "steps": 8, "similarity": 0.95}
```

### Detailed Result (per-run JSON)

```json
{
  "timestamp": "2024-12-28T14:30:22.123456",
  "task": {
    "id": "scikit-learn__scikit-learn-32932",
    "repo": "scikit-learn/scikit-learn",
    "difficulty": "easy"
  },
  "config": {
    "provider": "openai",
    "model": "o4-mini",
    "max_steps": 15
  },
  "agent": {
    "success": true,
    "steps": 12,
    "patch": "diff --git ...",
    "explanation": "Changed penalty to self.penalty..."
  },
  "evaluation": {
    "resolved": false,
    "no_regression": true,
    "diff_size": 2,
    "files_changed": ["sklearn/linear_model/_logistic.py"]
  },
  "metrics": {
    "tool_usage": { ... },
    "patch_quality": { ... },
    "failure_analysis": { ... }
  }
}
```

### View Summary

```bash
python test_e2e.py --summary
```

Output:
```
================================================================================
RUNS SUMMARY (5 total)
================================================================================
Timestamp            Task ID                              Model               Resolved  Steps  Similarity
--------------------------------------------------------------------------------
2024-12-28 14:30:22 scikit-learn__scikit-learn-32932    o4-mini              ✗         12     24.0%
2024-12-28 14:45:12 scikit-learn__scikit-learn-32932    gpt-4o               ✓         8      95.0%
--------------------------------------------------------------------------------

Resolved: 1/2 (50.0%)

By model:
  gpt-4o: 1/1 (100.0%)
  o4-mini: 0/1 (0.0%)
```

---

## Known Issues & Solutions

### 1. search_code with File Paths

**Problem**: Agent passes file paths like `sklearn/linear_model/_logistic.py` but grep's `--include` expects glob patterns.

**Solution**: Detect if pattern is a path or glob:
```python
if '/' in file_pattern and '*' not in file_pattern:
    # Specific file - search directly
    cmd = f"grep -n '{pattern}' '{file_pattern}'"
else:
    # Glob pattern
    cmd = f"grep -rn --include='{file_pattern}' '{pattern}' ."
```

### 2. OpenAI Null Content Error

**Problem**: OpenAI requires `content` to be a string, not null.

**Solution**: Use empty string instead of None:
```python
result = {"role": "assistant", "content": "\n".join(text_parts) if text_parts else ""}
```

### 3. OpenAI o-series max_tokens

**Problem**: o1/o3/o4 models use `max_completion_tokens` instead of `max_tokens`.

**Solution**: Detect model family:
```python
is_o_series = self.model.startswith(("o1", "o3", "o4"))
if is_o_series:
    kwargs["max_completion_tokens"] = max_tokens
else:
    kwargs["max_tokens"] = max_tokens
```

### 4. Agent Rewrites Entire Files

**Problem**: Agent uses `write_file` to rewrite 3000+ line files instead of making targeted edits.

**Solution**: Added `str_replace_in_file` tool and updated prompt to prefer it.

### 5. Tool Parameter KeyErrors

**Problem**: Different models use different parameter names (e.g., `path` vs `file_path`).

**Solution**: Accept multiple parameter aliases:
```python
path = input_data.get("path") or input_data.get("file_path") or input_data.get("file")
```

### 6. Tests Not Found

**Problem**: `fail_to_pass` tests may not exist in base commit (added by PR).

**Solution**: This is expected for some SWE-bench style tasks. The test is added as part of the solution.

---

## Future Work

### Short Term
- [ ] Add token usage tracking
- [ ] Implement parallel execution for batch evaluations
- [ ] Add more targeted edit tools (insert_line, delete_line)
- [ ] Improve test result parsing
- [ ] Add support for more repositories

### Medium Term
- [ ] Web UI for viewing results
- [ ] Automatic task difficulty calibration
- [ ] Agent trajectory visualization
- [ ] Cost tracking per run

### Long Term
- [ ] Multi-file task support
- [ ] Interactive debugging mode
- [ ] Agent self-improvement loop
- [ ] Comparison with SWE-bench leaderboard

---

## Contributing

Contributions welcome! Areas of interest:
- Additional repository support
- New evaluation metrics
- Agent improvements
- Documentation

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Inspired by [SWE-bench](https://www.swebench.com/)
- Built with Claude (Anthropic)
- Tested on scikit-learn