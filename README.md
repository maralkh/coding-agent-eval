# Coding Agent Eval

A comprehensive framework for building and evaluating coding agents on real-world software engineering tasks.

## Overview

This project provides:

1. **A minimal coding agent** that uses LLMs to solve programming tasks
2. **A task collector** that extracts evaluation tasks from GitHub PRs
3. **An evaluation framework** with 59 behavioral metrics
4. **A success prediction classifier** trained on agent behavior
5. **Sampling strategies** (best-of-N, majority voting) for improved performance

## Quick Start

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/coding-agent-eval.git
cd coding-agent-eval

# Option A: Use setup script (recommended)
chmod +x setup.sh
./setup.sh --venv --all

# Option B: Manual installation
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# Option C: Install specific provider only
pip install -e ".[anthropic]"  # For Claude
pip install -e ".[openai]"     # For GPT models
```

### 2. Set API Keys

```bash
# Required: At least one LLM provider
export ANTHROPIC_API_KEY="sk-ant-..."      # For Claude models
export OPENAI_API_KEY="sk-..."             # For GPT models
export GROQ_API_KEY="gsk_..."              # For Groq (free tier available)

# Optional: For higher GitHub rate limits when collecting tasks
export GITHUB_TOKEN="ghp_..."
```

### 3. Verify Installation

```bash
# Check environment
make check-env

# Or manually
python -c "from agent import RepoAgent; from eval import Task; print('✓ Setup complete')"
```

### 4. Run Your First Benchmark

```bash
# Run on sample tasks with Claude
python benchmark.py --tasks eval/tasks/ --models anthropic:claude-sonnet-4-20250514 --max-tasks 3

# Or use make
make benchmark-quick MODELS=anthropic:claude-sonnet-4-20250514
```

### 5. Run Full Pipeline

The `run_pipeline.sh` script runs the complete evaluation pipeline end-to-end:

```bash
./run_pipeline.sh
```

This executes 5 steps:
1. **Collect Tasks** - Gather evaluation tasks from GitHub PRs
2. **E2E Test** - Verify environment with a smoke test
3. **Benchmark** - Run benchmark across models
4. **Visualize** - Generate HTML report and charts
5. **Analyze** - Compute scores, rankings, and statistics

Edit the script to customize models, task limits, or output directory:

```bash
# In run_pipeline.sh
MODELS="gpt-4o"           # Change model
TASKS_DIR="eval/tasks"    # Tasks directory
```

## Installation Options

### Using pip

```bash
# Core only (no LLM SDKs)
pip install -e .

# With specific provider
pip install -e ".[anthropic]"
pip install -e ".[openai]"
pip install -e ".[groq]"

# All providers + classifier
pip install -e ".[all]"

# Development (includes testing tools)
pip install -e ".[dev]"
```

### Using Make

```bash
make install           # Core dependencies
make install-all       # All dependencies
make install-anthropic # Anthropic SDK only
make install-dev       # Development dependencies
```

## Usage

### Running Benchmarks

```bash
# Basic benchmark
python benchmark.py --tasks eval/tasks/ --models gpt-4o

# Multiple models
python benchmark.py --tasks eval/tasks/ \
    --models anthropic:claude-sonnet-4-20250514 openai:gpt-4o

# With sampling (best-of-5)
python benchmark.py --tasks eval/tasks/ --models gpt-4o \
    --n-samples 5 --sampling-strategy best_of_n

# View results
python benchmark.py --results-only -o results/benchmark/
```

### Single Task Testing

```bash
# Run single task
python test_e2e.py --task eval/tasks/scikit-learn__scikit-learn-28280.json \
    --provider anthropic --model claude-sonnet-4-20250514

# Debug mode
python debug_single_task.py --task eval/tasks/task.json --max-steps 5
```

### Programmatic Usage

```python
from agent import RepoAgent
from eval import Task

# Load task
task = Task.load("eval/tasks/scikit-learn__scikit-learn-28280.json")

# Run agent
agent = RepoAgent(max_steps=30, provider="anthropic", model="claude-sonnet-4-20250514")
result = agent.solve(task, repo_path="/path/to/repo")

print(f"Success: {result.success}")
print(f"Steps: {result.steps}")
print(f"Patch:\n{result.patch}")

# With sampling
sampling_result = agent.solve_with_sampling(
    task, repo_path,
    n_samples=5,
    strategy="best_of_n"
)
print(f"Best sample: {sampling_result.selected_index + 1}/{sampling_result.n_samples}")
```

## Project Structure

```
coding-agent-eval/
├── agent/                    # Agent implementations
│   ├── agent.py              # Simple coding agent
│   ├── repo_agent.py         # Repository-level agent
│   ├── repo_tools.py         # File/search/test tools
│   ├── sampling.py           # Best-of-N, majority voting
│   ├── prompts.py            # System prompts
│   └── llm.py                # Multi-provider LLM client
├── eval/                     # Evaluation framework
│   ├── task.py               # Task schema
│   ├── collector.py          # GitHub PR collector
│   ├── harness/              # Evaluation harness
│   │   ├── evaluator.py      # Test runner
│   │   ├── metrics.py        # 59 behavioral metrics
│   │   └── repo_manager.py   # Git operations
│   └── tasks/                # Sample tasks
├── classifier/               # Success prediction
│   ├── train_classifier.py   # Train RF/SVM/etc
│   └── collect_training_data.py
├── benchmark.py              # Multi-model benchmark runner
├── analyze_results.py        # Scoring and statistical analysis
├── test_e2e.py               # End-to-end testing
├── visualize_results.py      # Generate HTML reports
├── collect_tasks.py          # Collect tasks from GitHub
├── run_pipeline.sh           # End-to-end pipeline script
├── setup.sh                  # Setup script
├── Makefile                  # Make commands
├── requirements.txt          # Dependencies
└── pyproject.toml            # Package configuration
```

## Make Commands

```bash
make help              # Show all commands
make install           # Install dependencies
make check-env         # Verify setup
make test              # Run tests
make benchmark         # Run benchmark
make benchmark-quick   # Quick test (3 tasks)
make run-task          # Run single task
make visualize         # Generate reports
make pipeline          # Run full pipeline
make pipeline-quick    # Quick pipeline (3 tasks)
make clean             # Clean generated files
```

## Supported LLM Providers

| Provider | Models | Setup |
|----------|--------|-------|
| Anthropic | Claude Opus 4, Sonnet 4, Haiku 3.5 | `export ANTHROPIC_API_KEY=...` |
| OpenAI | GPT-4o, GPT-4o-mini, o1 | `export OPENAI_API_KEY=...` |
| Groq | Llama 3.1, Mixtral (free tier) | `export GROQ_API_KEY=...` |
| Ollama | Local models | `ollama serve` |

## Agent Tools

### Repository Agent (RepoAgent)

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents |
| `str_replace_in_file` | Make targeted edits (preferred) |
| `write_file` | Create or overwrite files |
| `list_directory` | Explore repo structure |
| `search_code` | Grep for patterns in codebase |
| `run_tests` | Run pytest on specific tests |
| `run_command` | Run shell commands |
| `submit_patch` | Submit final solution |

## Evaluation Metrics

The framework computes 59 behavioral metrics across 9 categories:

| Category | Metrics |
|----------|---------|
| Reasoning | Quality score, keyword coverage, hypothesis formation |
| Phases | Exploration/implementation/verification distribution |
| Exploration | Files explored, discovery step, wasted explorations |
| Trajectory | Length, efficiency, unnecessary steps |
| Convergence | Progress curve, volatility, regressions |
| Error Recovery | Total errors, recovery rate, stuck episodes |
| Tool Usage | Call counts, patterns, errors |
| Patch Quality | Size, correct files, similarity to gold |
| Semantic | Location correctness, change type match |

## Sampling Strategies

Improve resolve rates with multiple samples:

```bash
# Best-of-N: Select highest-scoring sample
python benchmark.py --tasks eval/tasks/ --models gpt-4o \
    --n-samples 5 --sampling-strategy best_of_n

# Majority voting: Select most common patch
python benchmark.py --tasks eval/tasks/ --models gpt-4o \
    --n-samples 5 --sampling-strategy majority_vote
```

## Training Success Classifier

```bash
# Collect training data from benchmark results
python -m classifier.collect_training_data --results results/benchmark/

# Train classifier
python -m classifier.train_classifier --data training_data/

# Model saved to models/classifier.joblib
```

## Visualization & Analysis

```bash
# Generate HTML report with charts
python visualize_results.py --results results/benchmark/ --output results/report.html

# Analyze results with scoring
python analyze_results.py --results results/benchmark/ --compare-methods

# Generate correlation plots
python analyze_results.py --results results/benchmark/ --plot plots/
```

Available scoring methods: `weighted`, `geometric`, `hierarchical`, `percentile`, `topsis`, `pareto`

## Documentation

- [DOCUMENTATION.md](DOCUMENTATION.md) - Comprehensive usage guide
- [docs/paper.tex](docs/paper.tex) - Technical report with methodology

## Contributing

Contributions welcome! Ideas:

- Add support for more repositories and languages
- Implement additional evaluation metrics
- Improve sampling strategies
- Add parallel execution

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Inspired by [SWE-bench](https://www.swebench.com/) and the broader work on evaluating coding agents.