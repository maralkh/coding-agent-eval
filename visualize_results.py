#!/usr/bin/env python
"""
Visualize benchmark results and generate reports.

Creates charts and HTML reports from benchmark results.

Usage:
    # Generate HTML report from results directory
    python visualize_results.py --results results/
    
    # Generate report with specific output name
    python visualize_results.py --results results/ --output report.html
    
    # Generate PNG images only (no HTML)
    python visualize_results.py --results results/ --images-only --output-dir charts/
    
    # Compare specific benchmark files
    python visualize_results.py --benchmark results/benchmark_20241228_120000.json
"""

import argparse
import base64
import io
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# =============================================================================
# DATA LOADING
# =============================================================================

@dataclass
class TaskResult:
    """Result of running a single task with a model."""
    task_id: str
    model: str
    provider: str
    resolved: bool = False
    submitted: bool = False
    steps: int = 0
    duration: float = 0.0
    similarity_score: float = 0.0
    reasoning_score: float = 0.0
    exploration_efficiency: float = 0.0
    trajectory_efficiency: float = 0.0
    primary_failure_mode: str = ""
    failure_reasons: list = field(default_factory=list)
    error: str = ""
    
    # Extended metrics (if available)
    semantic_similarity_score: float = 0.0
    likely_correct: bool = False
    correctness_confidence: float = 0.0
    fixes_same_file: bool = False
    fixes_same_function: bool = False


@dataclass
class ModelSummary:
    """Aggregated results for a model."""
    model: str
    provider: str
    total_tasks: int = 0
    resolved_count: int = 0
    submitted_count: int = 0
    error_count: int = 0
    resolve_rate: float = 0.0
    submit_rate: float = 0.0
    avg_steps: float = 0.0
    avg_duration: float = 0.0
    avg_similarity: float = 0.0
    avg_reasoning_score: float = 0.0
    avg_exploration_efficiency: float = 0.0
    avg_trajectory_efficiency: float = 0.0
    failure_modes: dict = field(default_factory=dict)
    task_results: list = field(default_factory=list)


def load_results(results_dir: str) -> tuple[list[TaskResult], list[ModelSummary], dict]:
    """Load all results from a results directory."""
    results_path = Path(results_dir)
    
    task_results = []
    model_summaries = []
    benchmark_info = {}
    
    # Load individual task results from JSONL
    jsonl_file = results_path / "all_results.jsonl"
    if jsonl_file.exists():
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    task_results.append(TaskResult(**{
                        k: v for k, v in data.items() 
                        if k in TaskResult.__dataclass_fields__
                    }))
    
    # Load benchmark summary files
    for json_file in sorted(results_path.glob("benchmark_*.json")):
        with open(json_file) as f:
            data = json.load(f)
            benchmark_info = data
            
            # Extract model summaries
            for model_data in data.get("models", []):
                summary = ModelSummary(**{
                    k: v for k, v in model_data.items()
                    if k in ModelSummary.__dataclass_fields__
                })
                model_summaries.append(summary)
    
    return task_results, model_summaries, benchmark_info


def load_benchmark_file(benchmark_file: str) -> tuple[list[TaskResult], list[ModelSummary], dict]:
    """Load results from a specific benchmark JSON file."""
    with open(benchmark_file) as f:
        data = json.load(f)
    
    task_results = []
    model_summaries = []
    
    for model_data in data.get("models", []):
        # Create ModelSummary
        summary = ModelSummary(**{
            k: v for k, v in model_data.items()
            if k in ModelSummary.__dataclass_fields__ and k != "task_results"
        })
        
        # Extract task results
        for tr_data in model_data.get("task_results", []):
            tr = TaskResult(**{
                k: v for k, v in tr_data.items()
                if k in TaskResult.__dataclass_fields__
            })
            task_results.append(tr)
            summary.task_results.append(tr)
        
        model_summaries.append(summary)
    
    return task_results, model_summaries, data


# =============================================================================
# CHART GENERATION
# =============================================================================

def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def create_model_comparison_chart(summaries: list[ModelSummary]) -> str:
    """Create bar chart comparing models on key metrics."""
    if not summaries:
        return ""
    
    models = [s.model for s in summaries]
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Metrics to compare
    resolve_rates = [s.resolve_rate * 100 for s in summaries]
    submit_rates = [s.submit_rate * 100 for s in summaries]
    similarities = [s.avg_similarity * 100 for s in summaries]
    reasoning = [s.avg_reasoning_score * 100 for s in summaries]
    
    bars1 = ax.bar(x - 1.5*width, resolve_rates, width, label='Resolve Rate', color='#2ecc71')
    bars2 = ax.bar(x - 0.5*width, submit_rates, width, label='Submit Rate', color='#3498db')
    bars3 = ax.bar(x + 0.5*width, similarities, width, label='Avg Similarity', color='#9b59b6')
    bars4 = ax.bar(x + 1.5*width, reasoning, width, label='Reasoning Score', color='#e74c3c')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 105)  # Extra space for labels
    
    # Add value labels
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.0f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    result = fig_to_base64(fig)
    plt.close(fig)
    return result


def create_efficiency_chart(summaries: list[ModelSummary]) -> str:
    """Create chart comparing efficiency metrics."""
    if not summaries:
        return ""
    
    models = [s.model for s in summaries]
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    exploration_eff = [s.avg_exploration_efficiency * 100 for s in summaries]
    trajectory_eff = [s.avg_trajectory_efficiency * 100 for s in summaries]
    
    bars1 = ax.bar(x - width/2, exploration_eff, width, label='Exploration Efficiency', color='#1abc9c')
    bars2 = ax.bar(x + width/2, trajectory_eff, width, label='Trajectory Efficiency', color='#f39c12')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Efficiency (%)')
    ax.set_title('Agent Efficiency Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    result = fig_to_base64(fig)
    plt.close(fig)
    return result


def create_steps_distribution_chart(task_results: list[TaskResult]) -> str:
    """Create histogram of steps taken per model (normalized to percentage)."""
    if not task_results:
        return ""
    
    # Group by model
    model_steps = defaultdict(list)
    for tr in task_results:
        model_steps[tr.model].append(tr.steps)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_steps)))
    
    for i, (model, steps) in enumerate(model_steps.items()):
        if not steps:
            continue
        # Use weights to normalize to percentage
        steps_arr = np.array(steps)
        weights = np.ones(len(steps)) / len(steps) * 100
        ax.hist(steps_arr, bins=20, alpha=0.6, label=model, color=colors[i], weights=weights)
    
    ax.set_xlabel('Number of Steps')
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(0, 100)
    ax.set_title('Distribution of Steps Taken')
    ax.legend()
    
    plt.tight_layout()
    result = fig_to_base64(fig)
    plt.close(fig)
    return result


def create_failure_mode_chart(summaries: list[ModelSummary]) -> str:
    """Create failure mode breakdown chart (normalized to percentage)."""
    if not summaries:
        return ""
    
    # Aggregate failure modes across all models
    all_failures = defaultdict(int)
    total_failures = 0
    for s in summaries:
        for mode, count in s.failure_modes.items():
            if mode and count > 0:
                all_failures[mode] += count
                total_failures += count
    
    if not all_failures or total_failures == 0:
        return ""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    modes = list(all_failures.keys())
    counts = list(all_failures.values())
    
    # Convert to percentages
    percentages = [c / total_failures * 100 for c in counts]
    
    # Sort by percentage
    sorted_pairs = sorted(zip(percentages, modes, counts), reverse=True)
    percentages, modes, counts = zip(*sorted_pairs) if sorted_pairs else ([], [], [])
    
    # Truncate long mode names
    modes = [m[:30] + '...' if len(m) > 30 else m for m in modes]
    
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(modes)))
    
    bars = ax.barh(modes, percentages, color=colors)
    ax.set_xlabel('Percentage (%)')
    ax.set_xlim(0, 100)
    ax.set_title('Failure Mode Distribution')
    
    # Add value labels
    for bar, pct, count in zip(bars, percentages, counts):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}% (n={count})', va='center', fontsize=9)
    
    plt.tight_layout()
    result = fig_to_base64(fig)
    plt.close(fig)
    return result


def create_per_model_failure_chart(summaries: list[ModelSummary]) -> str:
    """Create stacked bar chart of failure modes per model (normalized to percentage)."""
    if not summaries:
        return ""
    
    # Collect all failure modes
    all_modes = set()
    for s in summaries:
        all_modes.update(s.failure_modes.keys())
    all_modes = sorted([m for m in all_modes if m])
    
    if not all_modes:
        return ""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = [s.model for s in summaries]
    x = np.arange(len(models))
    
    # Calculate total failures per model for normalization
    model_totals = []
    for s in summaries:
        total = sum(s.failure_modes.get(mode, 0) for mode in all_modes)
        model_totals.append(total if total > 0 else 1)  # Avoid division by zero
    
    bottom = np.zeros(len(models))
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_modes)))
    
    for i, mode in enumerate(all_modes):
        counts = [s.failure_modes.get(mode, 0) for s in summaries]
        # Normalize to percentage per model
        percentages = np.array([c / t * 100 for c, t in zip(counts, model_totals)])
        ax.bar(x, percentages, bottom=bottom, label=mode[:20], color=colors[i])
        bottom = bottom + percentages
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(0, 100)
    ax.set_title('Failure Modes by Model (Normalized)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    result = fig_to_base64(fig)
    plt.close(fig)
    return result


def create_radar_chart(summary: ModelSummary) -> str:
    """Create radar chart for a single model's metrics."""
    categories = [
        'Resolve Rate', 'Submit Rate', 'Similarity',
        'Reasoning', 'Exploration Eff.', 'Trajectory Eff.'
    ]
    values = [
        summary.resolve_rate * 100,
        summary.submit_rate * 100,
        summary.avg_similarity * 100,
        summary.avg_reasoning_score * 100,
        summary.avg_exploration_efficiency * 100,
        summary.avg_trajectory_efficiency * 100,
    ]
    
    # Close the polygon
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    ax.plot(angles, values, 'o-', linewidth=2, color='#3498db')
    ax.fill(angles, values, alpha=0.25, color='#3498db')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title(f'{summary.model}\nPerformance Profile', fontsize=12)
    
    plt.tight_layout()
    result = fig_to_base64(fig)
    plt.close(fig)
    return result


def create_duration_chart(task_results: list[TaskResult]) -> str:
    """Create box plot of task durations per model."""
    if not task_results:
        return ""
    
    # Group by model
    model_durations = defaultdict(list)
    for tr in task_results:
        if tr.duration > 0:
            model_durations[tr.model].append(tr.duration)
    
    if not model_durations:
        return ""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(model_durations.keys())
    data = [model_durations[m] for m in models]
    
    bp = ax.boxplot(data, tick_labels=models, patch_artist=True)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Duration (seconds)')
    ax.set_title('Task Duration Distribution')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    result = fig_to_base64(fig)
    plt.close(fig)
    return result


def create_success_by_task_chart(task_results: list[TaskResult]) -> str:
    """Create heatmap of success by task and model."""
    if not task_results:
        return ""
    
    # Build matrix
    tasks = sorted(set(tr.task_id for tr in task_results))
    models = sorted(set(tr.model for tr in task_results))
    
    if len(tasks) > 20:  # Limit for readability
        tasks = tasks[:20]
    
    matrix = np.zeros((len(models), len(tasks)))
    
    for tr in task_results:
        if tr.task_id in tasks and tr.model in models:
            ti = tasks.index(tr.task_id)
            mi = models.index(tr.model)
            matrix[mi, ti] = 1 if tr.resolved else 0.5 if tr.submitted else 0
    
    fig, ax = plt.subplots(figsize=(max(12, len(tasks) * 0.5), max(4, len(models) * 0.8)))
    
    cmap = plt.cm.RdYlGn
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(tasks)))
    ax.set_yticks(range(len(models)))
    
    # Truncate task IDs for display
    task_labels = [t.split('__')[-1][:15] if '__' in t else t[:15] for t in tasks]
    ax.set_xticklabels(task_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(models, fontsize=9)
    
    ax.set_title('Task Success by Model (Green=Resolved, Yellow=Submitted, Red=Failed)')
    
    plt.colorbar(im, ax=ax, label='Status')
    plt.tight_layout()
    result = fig_to_base64(fig)
    plt.close(fig)
    return result


def create_metrics_correlation_chart(task_results: list[TaskResult]) -> str:
    """Create scatter plot of metrics correlations (normalized to percentage)."""
    if not task_results or len(task_results) < 5:
        return ""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Similarity vs Resolved
    ax = axes[0, 0]
    resolved = [tr.similarity_score for tr in task_results if tr.resolved]
    failed = [tr.similarity_score for tr in task_results if not tr.resolved]
    # Normalize with weights
    weights_resolved = np.ones(len(resolved)) / len(resolved) * 100 if resolved else []
    weights_failed = np.ones(len(failed)) / len(failed) * 100 if failed else []
    ax.hist([resolved, failed], bins=20, label=['Resolved', 'Failed'], 
            color=['#2ecc71', '#e74c3c'], weights=[weights_resolved, weights_failed])
    ax.set_xlabel('Similarity Score')
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(0, 100)
    ax.set_title('Similarity Score Distribution')
    ax.legend()
    
    # Reasoning vs Resolved
    ax = axes[0, 1]
    resolved = [tr.reasoning_score for tr in task_results if tr.resolved]
    failed = [tr.reasoning_score for tr in task_results if not tr.resolved]
    weights_resolved = np.ones(len(resolved)) / len(resolved) * 100 if resolved else []
    weights_failed = np.ones(len(failed)) / len(failed) * 100 if failed else []
    ax.hist([resolved, failed], bins=20, label=['Resolved', 'Failed'], 
            color=['#2ecc71', '#e74c3c'], weights=[weights_resolved, weights_failed])
    ax.set_xlabel('Reasoning Score')
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(0, 100)
    ax.set_title('Reasoning Score Distribution')
    ax.legend()
    
    # Steps vs Resolved
    ax = axes[1, 0]
    resolved = [tr.steps for tr in task_results if tr.resolved]
    failed = [tr.steps for tr in task_results if not tr.resolved]
    weights_resolved = np.ones(len(resolved)) / len(resolved) * 100 if resolved else []
    weights_failed = np.ones(len(failed)) / len(failed) * 100 if failed else []
    ax.hist([resolved, failed], bins=20, label=['Resolved', 'Failed'], 
            color=['#2ecc71', '#e74c3c'], alpha=0.7, weights=[weights_resolved, weights_failed])
    ax.set_xlabel('Steps')
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(0, 100)
    ax.set_title('Steps Distribution')
    ax.legend()
    
    # Exploration Efficiency vs Trajectory Efficiency
    ax = axes[1, 1]
    colors = ['#2ecc71' if tr.resolved else '#e74c3c' for tr in task_results]
    ax.scatter(
        [tr.exploration_efficiency for tr in task_results],
        [tr.trajectory_efficiency for tr in task_results],
        c=colors, alpha=0.6
    )
    ax.set_xlabel('Exploration Efficiency')
    ax.set_ylabel('Trajectory Efficiency')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Exploration vs Trajectory Efficiency')
    # Legend
    resolved_patch = mpatches.Patch(color='#2ecc71', label='Resolved')
    failed_patch = mpatches.Patch(color='#e74c3c', label='Failed')
    ax.legend(handles=[resolved_patch, failed_patch])
    
    plt.tight_layout()
    result = fig_to_base64(fig)
    plt.close(fig)
    return result


# =============================================================================
# HTML REPORT GENERATION
# =============================================================================

def generate_summary_cards(summaries: list[ModelSummary]) -> str:
    """Generate HTML for summary cards."""
    if not summaries:
        return ""
    
    # Aggregate stats
    total_tasks = sum(s.total_tasks for s in summaries) // len(summaries) if summaries else 0
    total_resolved = sum(s.resolved_count for s in summaries)
    avg_resolve_rate = np.mean([s.resolve_rate for s in summaries]) * 100
    avg_steps = np.mean([s.avg_steps for s in summaries])
    
    best_model = max(summaries, key=lambda s: s.resolve_rate).model if summaries else "N/A"
    
    cards = [
        ("Total Tasks", str(total_tasks), ""),
        ("Total Resolved", str(total_resolved), "success"),
        ("Avg Resolve Rate", f"{avg_resolve_rate:.1f}%", "success" if avg_resolve_rate > 50 else "warning"),
        ("Avg Steps", f"{avg_steps:.1f}", ""),
        ("Best Model", best_model, "success"),
        ("Models Tested", str(len(summaries)), ""),
    ]
    
    html = ""
    for title, value, card_class in cards:
        html += f'''
        <div class="card {card_class}">
            <h3>{title}</h3>
            <div class="value">{value}</div>
        </div>
        '''
    
    return html


def generate_results_table(summaries: list[ModelSummary]) -> str:
    """Generate HTML for results table."""
    rows = ""
    for s in sorted(summaries, key=lambda x: x.resolve_rate, reverse=True):
        rate_class = "success" if s.resolve_rate > 0.5 else "warning" if s.resolve_rate > 0.2 else "danger"
        rows += f'''
        <tr>
            <td><strong>{s.model}</strong></td>
            <td>{s.total_tasks}</td>
            <td>{s.resolved_count}</td>
            <td><span class="badge badge-{rate_class}">{s.resolve_rate*100:.1f}%</span></td>
            <td>{s.avg_steps:.1f}</td>
            <td>{s.avg_duration:.1f}s</td>
            <td>{s.avg_similarity*100:.1f}%</td>
        </tr>
        '''
    return rows


def generate_radar_charts_html(summaries: list[ModelSummary]) -> str:
    """Generate HTML for radar charts."""
    html = ""
    for s in summaries:
        chart_b64 = create_radar_chart(s)
        if chart_b64:
            html += f'''
            <div class="chart-container">
                <img src="data:image/png;base64,{chart_b64}" alt="{s.model} Radar">
            </div>
            '''
    return html


def generate_html_report(
    task_results: list[TaskResult],
    summaries: list[ModelSummary],
    benchmark_info: dict,
) -> str:
    """Generate complete HTML report."""
    
    # Generate all charts
    model_comparison = create_model_comparison_chart(summaries)
    efficiency = create_efficiency_chart(summaries)
    steps = create_steps_distribution_chart(task_results)
    failure = create_failure_mode_chart(summaries)
    per_model_failure = create_per_model_failure_chart(summaries)
    duration = create_duration_chart(task_results)
    success_heatmap = create_success_by_task_chart(task_results)
    correlation = create_metrics_correlation_chart(task_results)
    
    # Generate HTML components
    summary_cards = generate_summary_cards(summaries)
    results_table = generate_results_table(summaries)
    radar_charts = generate_radar_charts_html(summaries)
    
    # Calculate stats
    total_tasks = len(benchmark_info.get("tasks", [])) or (sum(s.total_tasks for s in summaries) // len(summaries) if summaries else 0)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Build HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark Results Report</title>
    <style>
        :root {{
            --primary: #3498db;
            --success: #2ecc71;
            --warning: #f39c12;
            --danger: #e74c3c;
            --dark: #2c3e50;
            --light: #ecf0f1;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6; color: var(--dark); background: var(--light); padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        header {{
            background: var(--dark); color: white; padding: 30px;
            border-radius: 10px; margin-bottom: 30px;
        }}
        header h1 {{ margin-bottom: 10px; }}
        header .meta {{ opacity: 0.8; font-size: 0.9em; }}
        .summary-cards {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px; margin-bottom: 30px;
        }}
        .card {{
            background: white; border-radius: 10px; padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .card h3 {{ color: var(--primary); margin-bottom: 10px; font-size: 0.9em; text-transform: uppercase; }}
        .card .value {{ font-size: 2em; font-weight: bold; }}
        .card.success .value {{ color: var(--success); }}
        .card.warning .value {{ color: var(--warning); }}
        .card.danger .value {{ color: var(--danger); }}
        section {{
            background: white; border-radius: 10px; padding: 25px;
            margin-bottom: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        section h2 {{
            color: var(--dark); border-bottom: 2px solid var(--primary);
            padding-bottom: 10px; margin-bottom: 20px;
        }}
        .chart-container {{ text-align: center; margin: 20px 0; }}
        .chart-container img {{ max-width: 100%; height: auto; border-radius: 5px; }}
        .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid var(--light); }}
        th {{ background: var(--dark); color: white; }}
        tr:hover {{ background: #f5f6fa; }}
        .badge {{ display: inline-block; padding: 3px 8px; border-radius: 3px; font-size: 0.8em; font-weight: bold; }}
        .badge-success {{ background: var(--success); color: white; }}
        .badge-warning {{ background: var(--warning); color: white; }}
        .badge-danger {{ background: var(--danger); color: white; }}
        footer {{ text-align: center; color: #7f8c8d; padding: 20px; }}
        @media (max-width: 768px) {{ .chart-grid {{ grid-template-columns: 1fr; }} }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Benchmark Results Report</h1>
            <div class="meta">
                <p>Generated: {timestamp}</p>
                <p>Tasks: {total_tasks} | Models: {len(summaries)}</p>
            </div>
        </header>
        
        <div class="summary-cards">
            {summary_cards}
        </div>
        
        <section>
            <h2>📈 Model Comparison</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{model_comparison}" alt="Model Comparison">
            </div>
        </section>
        
        <section>
            <h2>⚡ Efficiency Metrics</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{efficiency}" alt="Efficiency Comparison">
            </div>
        </section>
        
        <section>
            <h2>🎯 Model Performance Profiles</h2>
            <div class="chart-grid">
                {radar_charts}
            </div>
        </section>
        
        <section>
            <h2>📊 Detailed Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Tasks</th>
                        <th>Resolved</th>
                        <th>Resolve Rate</th>
                        <th>Avg Steps</th>
                        <th>Avg Duration</th>
                        <th>Avg Similarity</th>
                    </tr>
                </thead>
                <tbody>
                    {results_table}
                </tbody>
            </table>
        </section>
        
        <section>
            <h2>⏱️ Task Duration Distribution</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{duration}" alt="Duration Distribution">
            </div>
        </section>
        
        <section>
            <h2>📉 Steps Distribution</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{steps}" alt="Steps Distribution">
            </div>
        </section>
        
        <section>
            <h2>🔍 Metrics Analysis</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{correlation}" alt="Metrics Correlation">
            </div>
        </section>
        
        <section>
            <h2>❌ Failure Analysis</h2>
            <div class="chart-grid">
                <div class="chart-container">
                    <img src="data:image/png;base64,{failure}" alt="Failure Modes">
                </div>
                <div class="chart-container">
                    <img src="data:image/png;base64,{per_model_failure}" alt="Failures by Model">
                </div>
            </div>
        </section>
        
        <section>
            <h2>🗺️ Task Success Heatmap</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{success_heatmap}" alt="Success Heatmap">
            </div>
        </section>
        
        <footer>
            <p>Generated by Coding Agent Eval Framework</p>
        </footer>
    </div>
</body>
</html>'''
    
    return html


# =============================================================================
# IMAGE EXPORT
# =============================================================================

def save_charts_as_images(
    task_results: list[TaskResult],
    summaries: list[ModelSummary],
    output_dir: Path,
):
    """Save all charts as PNG files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    charts = [
        ("model_comparison.png", create_model_comparison_chart, summaries),
        ("efficiency.png", create_efficiency_chart, summaries),
        ("steps_distribution.png", create_steps_distribution_chart, task_results),
        ("failure_modes.png", create_failure_mode_chart, summaries),
        ("per_model_failures.png", create_per_model_failure_chart, summaries),
        ("duration.png", create_duration_chart, task_results),
        ("success_heatmap.png", create_success_by_task_chart, task_results),
        ("metrics_correlation.png", create_metrics_correlation_chart, task_results),
    ]
    
    for filename, chart_func, data in charts:
        try:
            b64 = chart_func(data)
            if b64:
                img_data = base64.b64decode(b64)
                (output_dir / filename).write_bytes(img_data)
                print(f"  Saved {filename}")
        except Exception as e:
            print(f"  Warning: Could not create {filename}: {e}")
    
    # Radar charts per model
    for s in summaries:
        try:
            b64 = create_radar_chart(s)
            if b64:
                safe_name = s.model.replace("/", "_").replace(":", "_")
                img_data = base64.b64decode(b64)
                (output_dir / f"radar_{safe_name}.png").write_bytes(img_data)
                print(f"  Saved radar_{safe_name}.png")
        except Exception as e:
            print(f"  Warning: Could not create radar chart for {s.model}: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize benchmark results and generate reports"
    )
    
    parser.add_argument(
        "--results", "-r",
        type=str,
        help="Path to results directory containing benchmark output",
    )
    parser.add_argument(
        "--benchmark", "-b",
        type=str,
        help="Path to specific benchmark JSON file",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="benchmark_report.html",
        help="Output file path (default: benchmark_report.html)",
    )
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Only generate PNG images, no HTML report",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="charts",
        help="Output directory for images (with --images-only)",
    )
    
    args = parser.parse_args()
    
    if not args.results and not args.benchmark:
        parser.error("Either --results or --benchmark must be specified")
    
    # Load data
    if args.benchmark:
        print(f"Loading benchmark file: {args.benchmark}")
        task_results, summaries, benchmark_info = load_benchmark_file(args.benchmark)
    else:
        print(f"Loading results from: {args.results}")
        task_results, summaries, benchmark_info = load_results(args.results)
    
    print(f"  Loaded {len(task_results)} task results")
    print(f"  Loaded {len(summaries)} model summaries")
    
    if not summaries:
        print("Error: No data found to visualize")
        sys.exit(1)
    
    # Generate output
    if args.images_only:
        print(f"\nGenerating charts in: {args.output_dir}")
        save_charts_as_images(task_results, summaries, Path(args.output_dir))
        print(f"\nCharts saved to {args.output_dir}/")
    else:
        print(f"\nGenerating HTML report...")
        html = generate_html_report(task_results, summaries, benchmark_info)
        
        output_path = Path(args.output)
        output_path.write_text(html)
        print(f"\nReport saved to: {output_path}")
        print(f"Open in browser: file://{output_path.absolute()}")


if __name__ == "__main__":
    main()
