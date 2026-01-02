#!/usr/bin/env python3
"""
Curriculum Learning Signal Analysis

Ranks tasks by "informativeness" for agent training based on classifier uncertainty.
Tasks where the classifier is uncertain (P ≈ 0.5) are most informative for learning.

Usage:
------
    python analysis/curriculum.py --data training_data/real_training_data.jsonl --output analysis/outputs/curriculum

Arguments:
----------
    --data        Path to training data JSONL file
    --output      Output directory for results and plots

Outputs:
--------
    task_difficulty.csv         Task rankings with difficulty metrics
    curriculum_ordering.png     Visualization of task difficulty spectrum
    model_agreement.png         Heatmap of model agreement per task
    uncertainty_distribution.png Distribution of classifier uncertainty
    curriculum_report.md        Markdown summary report

Key Concepts:
-------------
    - Tasks where all models succeed (easy) or all fail (hard) teach less
    - Tasks with mixed outcomes or uncertain predictions are most informative
    - Curriculum learning: order tasks from easy → uncertain → hard
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
import warnings
warnings.filterwarnings('ignore')


def load_training_data(data_path: Path) -> pd.DataFrame:
    """Load training data from JSONL file."""
    records = []
    with open(data_path) as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get numeric feature columns."""
    exclude = {'task_id', 'model', 'provider', 'timestamp', 'resolved', 'exploration_strategy'}
    return [c for c in df.columns if c not in exclude and df[c].dtype in ['int64', 'float64', 'bool']]


def compute_task_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute difficulty metrics for each task.
    
    Metrics:
    - success_rate: Fraction of models that solved it
    - model_agreement: How much models agree (0 = split, 1 = unanimous)
    - classifier_uncertainty: Mean |P - 0.5| (low = uncertain = informative)
    - informativeness: Combined score (higher = better for training)
    """
    
    # Get classifier probabilities
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].fillna(0)
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
    y = df['resolved'].astype(int).values
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                 min_samples_leaf=5, random_state=42)
    probabilities = cross_val_predict(clf, X.values, y, cv=5, method='predict_proba')[:, 1]
    df = df.copy()
    df['probability'] = probabilities
    
    # Group by task
    task_metrics = []
    
    for task_id, group in df.groupby('task_id'):
        n_models = len(group)
        n_success = group['resolved'].sum()
        success_rate = n_success / n_models
        
        # Model agreement: 1 if all same, 0 if perfectly split
        agreement = abs(2 * success_rate - 1)  # 0 at 50%, 1 at 0% or 100%
        
        # Classifier uncertainty: distance from 0.5
        mean_prob = group['probability'].mean()
        uncertainty = 1 - abs(2 * mean_prob - 1)  # 1 at 0.5, 0 at 0 or 1
        
        # Variance in predictions across models
        prob_variance = group['probability'].var() if n_models > 1 else 0
        
        # Informativeness: high when uncertain AND models disagree
        informativeness = uncertainty * (1 - agreement + 0.1)  # +0.1 to avoid zero
        
        # Difficulty category
        if success_rate >= 0.75:
            difficulty_cat = 'easy'
        elif success_rate <= 0.25:
            difficulty_cat = 'hard'
        else:
            difficulty_cat = 'medium'
        
        task_metrics.append({
            'task_id': task_id,
            'n_models': n_models,
            'n_success': int(n_success),
            'success_rate': success_rate,
            'model_agreement': agreement,
            'mean_probability': mean_prob,
            'prob_variance': prob_variance,
            'uncertainty': uncertainty,
            'informativeness': informativeness,
            'difficulty_category': difficulty_cat,
            'models_succeeded': ', '.join(group[group['resolved']]['model'].tolist()),
            'models_failed': ', '.join(group[~group['resolved']]['model'].tolist()),
        })
    
    return pd.DataFrame(task_metrics)


def create_curriculum_ordering(task_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create curriculum ordering from easy → medium → hard.
    
    Within each category, order by informativeness (most informative first).
    """
    # Sort within categories
    easy = task_df[task_df['difficulty_category'] == 'easy'].sort_values('informativeness', ascending=False)
    medium = task_df[task_df['difficulty_category'] == 'medium'].sort_values('informativeness', ascending=False)
    hard = task_df[task_df['difficulty_category'] == 'hard'].sort_values('informativeness', ascending=False)
    
    # Concatenate in curriculum order
    curriculum = pd.concat([easy, medium, hard], ignore_index=True)
    curriculum['curriculum_order'] = range(1, len(curriculum) + 1)
    
    return curriculum


def plot_difficulty_spectrum(task_df: pd.DataFrame, output_path: Path) -> Path:
    """Visualize task difficulty spectrum."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Success rate distribution
    ax1 = axes[0, 0]
    colors = {'easy': '#2ecc71', 'medium': '#f39c12', 'hard': '#e74c3c'}
    for cat in ['easy', 'medium', 'hard']:
        data = task_df[task_df['difficulty_category'] == cat]['success_rate']
        ax1.hist(data, bins=10, alpha=0.6, label=f'{cat.capitalize()} ({len(data)})', 
                color=colors[cat], edgecolor='black')
    ax1.set_xlabel('Success Rate', fontsize=12)
    ax1.set_ylabel('Number of Tasks', fontsize=12)
    ax1.set_title('Task Difficulty Distribution', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Informativeness vs Success Rate
    ax2 = axes[0, 1]
    scatter = ax2.scatter(task_df['success_rate'], task_df['informativeness'],
                         c=task_df['uncertainty'], cmap='RdYlGn', 
                         s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Success Rate', fontsize=12)
    ax2.set_ylabel('Informativeness Score', fontsize=12)
    ax2.set_title('Task Informativeness vs Difficulty', fontsize=14)
    plt.colorbar(scatter, ax=ax2, label='Uncertainty')
    ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Uncertainty distribution
    ax3 = axes[1, 0]
    ax3.hist(task_df['uncertainty'], bins=20, color='#3498db', 
            edgecolor='black', alpha=0.7)
    ax3.axvline(x=task_df['uncertainty'].mean(), color='red', 
               linestyle='--', linewidth=2, label=f'Mean: {task_df["uncertainty"].mean():.2f}')
    ax3.set_xlabel('Classifier Uncertainty', fontsize=12)
    ax3.set_ylabel('Number of Tasks', fontsize=12)
    ax3.set_title('Distribution of Classifier Uncertainty', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Curriculum order visualization
    ax4 = axes[1, 1]
    curriculum = create_curriculum_ordering(task_df)
    colors_list = [colors[cat] for cat in curriculum['difficulty_category']]
    ax4.bar(range(len(curriculum)), curriculum['success_rate'], color=colors_list, alpha=0.7)
    ax4.set_xlabel('Curriculum Order', fontsize=12)
    ax4.set_ylabel('Success Rate', fontsize=12)
    ax4.set_title('Curriculum Ordering (Easy → Medium → Hard)', fontsize=14)
    ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[cat], label=cat.capitalize()) 
                      for cat in ['easy', 'medium', 'hard']]
    ax4.legend(handles=legend_elements, loc='upper right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    plot_path = output_path / 'curriculum_ordering.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def plot_model_agreement(df: pd.DataFrame, task_df: pd.DataFrame, output_path: Path) -> Path:
    """Create heatmap showing which models solved which tasks."""
    
    # Pivot to create task x model matrix
    pivot = df.pivot(index='task_id', columns='model', values='resolved')
    pivot = pivot.astype(int)
    
    # Sort tasks by success rate
    task_order = task_df.sort_values('success_rate', ascending=False)['task_id'].tolist()
    pivot = pivot.reindex(task_order)
    
    fig, ax = plt.subplots(figsize=(10, 14))
    
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(0, len(pivot.index), 5))
    ax.set_yticklabels([pivot.index[i][:30] for i in range(0, len(pivot.index), 5)], fontsize=8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Task (sorted by success rate)', fontsize=12)
    ax.set_title('Model Success by Task (Green=Solved, Red=Failed)', fontsize=14)
    
    plt.colorbar(im, ax=ax, label='Solved', shrink=0.5)
    
    plt.tight_layout()
    
    plot_path = output_path / 'model_agreement.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def plot_boundary_tasks(task_df: pd.DataFrame, output_path: Path) -> Path:
    """Highlight the most informative 'boundary' tasks."""
    
    # Top 10 most informative tasks
    top_informative = task_df.nlargest(15, 'informativeness')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {'easy': '#2ecc71', 'medium': '#f39c12', 'hard': '#e74c3c'}
    bar_colors = [colors[cat] for cat in top_informative['difficulty_category']]
    
    bars = ax.barh(range(len(top_informative)), top_informative['informativeness'], 
                   color=bar_colors, alpha=0.7, edgecolor='black')
    
    ax.set_yticks(range(len(top_informative)))
    ax.set_yticklabels([f"{t[:35]}..." if len(t) > 35 else t 
                       for t in top_informative['task_id']], fontsize=9)
    ax.set_xlabel('Informativeness Score', fontsize=12)
    ax.set_title('Top 15 Most Informative Tasks for Training', fontsize=14)
    ax.invert_yaxis()
    
    # Add success rate annotations
    for i, (idx, row) in enumerate(top_informative.iterrows()):
        ax.annotate(f"{row['success_rate']:.0%}", 
                   xy=(row['informativeness'] + 0.01, i),
                   va='center', fontsize=9)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[cat], label=cat.capitalize()) 
                      for cat in ['easy', 'medium', 'hard']]
    ax.legend(handles=legend_elements, loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    plot_path = output_path / 'boundary_tasks.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def generate_report(task_df: pd.DataFrame, curriculum: pd.DataFrame, output_path: Path) -> Path:
    """Generate markdown report."""
    
    lines = [
        "# Curriculum Learning Signal Analysis",
        "",
        "## Overview",
        "",
        "This analysis ranks tasks by their **informativeness** for agent training.",
        "Tasks where the classifier is uncertain (P ≈ 0.5) or models disagree are most valuable",
        "for learning, as they represent the decision boundary the model needs to learn.",
        "",
        "## Task Difficulty Distribution",
        "",
        f"| Category | Count | Success Rate Range |",
        f"|----------|-------|-------------------|",
        f"| Easy | {len(task_df[task_df['difficulty_category'] == 'easy'])} | ≥75% |",
        f"| Medium | {len(task_df[task_df['difficulty_category'] == 'medium'])} | 25-75% |",
        f"| Hard | {len(task_df[task_df['difficulty_category'] == 'hard'])} | ≤25% |",
        "",
        "## Key Metrics",
        "",
        f"- **Total tasks**: {len(task_df)}",
        f"- **Mean success rate**: {task_df['success_rate'].mean():.1%}",
        f"- **Mean classifier uncertainty**: {task_df['uncertainty'].mean():.3f}",
        f"- **Tasks with unanimous model agreement**: {len(task_df[task_df['model_agreement'] == 1])}",
        f"- **Tasks with model disagreement**: {len(task_df[task_df['model_agreement'] < 1])}",
        "",
        "## Top 10 Most Informative Tasks",
        "",
        "These tasks are best for training—they're at the decision boundary where models disagree",
        "or the classifier is uncertain.",
        "",
        "| Rank | Task ID | Success Rate | Uncertainty | Informativeness |",
        "|------|---------|--------------|-------------|-----------------|",
    ]
    
    for i, (_, row) in enumerate(task_df.nlargest(10, 'informativeness').iterrows(), 1):
        task_short = row['task_id'][:40] + "..." if len(row['task_id']) > 40 else row['task_id']
        lines.append(
            f"| {i} | {task_short} | {row['success_rate']:.0%} | {row['uncertainty']:.3f} | {row['informativeness']:.3f} |"
        )
    
    lines.extend([
        "",
        "## Easiest Tasks (All Models Succeed)",
        "",
        "| Task ID | Success Rate | Models Succeeded |",
        "|---------|--------------|------------------|",
    ])
    
    easy_tasks = task_df[task_df['success_rate'] >= 0.75].nlargest(5, 'success_rate')
    for _, row in easy_tasks.iterrows():
        task_short = row['task_id'][:40] + "..." if len(row['task_id']) > 40 else row['task_id']
        lines.append(f"| {task_short} | {row['success_rate']:.0%} | {row['models_succeeded'][:30]} |")
    
    lines.extend([
        "",
        "## Hardest Tasks (All Models Fail)",
        "",
        "| Task ID | Success Rate | Models Failed |",
        "|---------|--------------|---------------|",
    ])
    
    hard_tasks = task_df[task_df['success_rate'] <= 0.25].nsmallest(5, 'success_rate')
    for _, row in hard_tasks.iterrows():
        task_short = row['task_id'][:40] + "..." if len(row['task_id']) > 40 else row['task_id']
        models_failed = row['models_failed'][:40] if len(row['models_failed']) > 40 else row['models_failed']
        lines.append(f"| {task_short} | {row['success_rate']:.0%} | {models_failed} |")
    
    lines.extend([
        "",
        "## Curriculum Ordering Strategy",
        "",
        "For training, order tasks as:",
        "",
        "1. **Easy tasks first**: Build confidence, learn basic patterns",
        "2. **Medium tasks (most informative)**: Learn decision boundaries",
        "3. **Hard tasks last**: Tackle edge cases after mastering basics",
        "",
        "Within each category, prioritize by informativeness (model disagreement + uncertainty).",
        "",
        "## Practical Applications",
        "",
        "1. **Curriculum Learning**: Train agents on easy → hard progression",
        "2. **Active Learning**: Focus data collection on uncertain/boundary tasks",
        "3. **Evaluation Design**: Use informative tasks for discriminative benchmarks",
        "4. **Ensemble Training**: Use tasks with model disagreement to train diverse ensembles",
        "",
        "## Visualizations",
        "",
        "- `curriculum_ordering.png`: Task difficulty spectrum and curriculum order",
        "- `model_agreement.png`: Heatmap of which models solved which tasks",
        "- `boundary_tasks.png`: Top informative tasks for training",
        "",
    ])
    
    report_path = output_path / 'curriculum_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    
    return report_path


def main():
    """Run curriculum analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Curriculum Learning Signal Analysis')
    parser.add_argument('--data', type=Path, default=Path('training_data/real_training_data.jsonl'),
                       help='Path to training data JSONL')
    parser.add_argument('--output', type=Path, default=Path('analysis/outputs/curriculum'),
                       help='Output directory')
    
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CURRICULUM LEARNING SIGNAL ANALYSIS")
    print("=" * 60)
    print()
    
    # Load data
    print(f"Loading data from {args.data}...")
    df = load_training_data(args.data)
    print(f"Loaded {len(df)} samples")
    print(f"Unique tasks: {df['task_id'].nunique()}")
    print(f"Models: {sorted(df['model'].unique())}")
    print()
    
    # Compute task difficulty
    print("Computing task difficulty metrics...")
    task_df = compute_task_difficulty(df)
    
    # Save task metrics
    task_df.to_csv(args.output / 'task_difficulty.csv', index=False)
    print(f"Saved task metrics to {args.output / 'task_difficulty.csv'}")
    
    # Create curriculum ordering
    curriculum = create_curriculum_ordering(task_df)
    curriculum.to_csv(args.output / 'curriculum_ordering.csv', index=False)
    
    # Print summary
    print("\nTask Difficulty Distribution:")
    print("-" * 40)
    for cat in ['easy', 'medium', 'hard']:
        count = len(task_df[task_df['difficulty_category'] == cat])
        print(f"  {cat.capitalize():8}: {count} tasks")
    
    print(f"\nMean classifier uncertainty: {task_df['uncertainty'].mean():.3f}")
    print(f"Tasks with model disagreement: {len(task_df[task_df['model_agreement'] < 1])}")
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    plot1 = plot_difficulty_spectrum(task_df, args.output)
    print(f"  Difficulty spectrum: {plot1}")
    
    plot2 = plot_model_agreement(df, task_df, args.output)
    print(f"  Model agreement: {plot2}")
    
    plot3 = plot_boundary_tasks(task_df, args.output)
    print(f"  Boundary tasks: {plot3}")
    
    # Generate report
    report_path = generate_report(task_df, curriculum, args.output)
    print(f"\nReport saved to {report_path}")
    
    # Print top informative tasks
    print("\n" + "=" * 60)
    print("TOP 10 MOST INFORMATIVE TASKS")
    print("=" * 60)
    print()
    print(f"{'Rank':<5} {'Task':<45} {'Success':>8} {'Info':>8}")
    print("-" * 70)
    
    for i, (_, row) in enumerate(task_df.nlargest(10, 'informativeness').iterrows(), 1):
        task_short = row['task_id'][:42] + "..." if len(row['task_id']) > 42 else row['task_id']
        print(f"{i:<5} {task_short:<45} {row['success_rate']:>7.0%} {row['informativeness']:>8.3f}")
    
    print("\n" + "=" * 60)
    print("✓ Analysis complete!")


if __name__ == '__main__':
    main()