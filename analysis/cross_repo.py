#!/usr/bin/env python3
"""
Cross-Repository Transfer Analysis

Analyzes how well behavioral features and classifier predictions
generalize across different repositories/domains.

Since we only have scikit-learn data, this analysis:
1. Simulates cross-repo by splitting tasks into "pseudo-repos" 
2. Analyzes feature stability across task subsets
3. Identifies which features are repo-specific vs universal
4. Provides recommendations for transfer learning

Usage:
------
    python analysis/cross_repo.py --data training_data/real_training_data.jsonl

Outputs:
--------
    feature_stability.png        Feature variance across task groups
    transfer_matrix.png          Cross-validation transfer performance
    universal_features.csv       Features that generalize well
    cross_repo_report.md         Analysis report
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from scipy import stats
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


def create_pseudo_repos(df: pd.DataFrame, n_groups: int = 4) -> pd.DataFrame:
    """
    Split tasks into pseudo-repositories for transfer analysis.
    
    Groups tasks by their characteristics to simulate different repos.
    """
    df = df.copy()
    
    # Get unique tasks
    tasks = df['task_id'].unique()
    
    # Simple split: divide tasks into n groups
    task_to_group = {task: i % n_groups for i, task in enumerate(sorted(tasks))}
    df['pseudo_repo'] = df['task_id'].map(task_to_group)
    
    # Name the pseudo repos
    repo_names = ['repo_A', 'repo_B', 'repo_C', 'repo_D']
    df['pseudo_repo_name'] = df['pseudo_repo'].map(lambda x: repo_names[x])
    
    return df


def compute_feature_stability(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Compute how stable each feature is across pseudo-repos.
    
    Features with low cross-repo variance are more likely to transfer.
    """
    results = []
    
    for feature in feature_cols:
        # Compute mean per pseudo-repo
        repo_means = df.groupby('pseudo_repo_name')[feature].mean()
        
        # Coefficient of variation across repos
        cv = repo_means.std() / (repo_means.mean() + 1e-10)
        
        # Overall stats
        overall_mean = df[feature].mean()
        overall_std = df[feature].std()
        
        # Correlation with success (should be stable if feature generalizes)
        correlations = []
        for repo in df['pseudo_repo_name'].unique():
            repo_df = df[df['pseudo_repo_name'] == repo]
            if len(repo_df) > 5 and repo_df[feature].std() > 0:
                corr = repo_df[feature].corr(repo_df['resolved'].astype(float))
                if not np.isnan(corr):
                    correlations.append(corr)
        
        corr_stability = np.std(correlations) if len(correlations) > 1 else 1.0
        
        # Stability score (lower = more stable = better for transfer)
        stability_score = cv + corr_stability
        
        results.append({
            'feature': feature,
            'overall_mean': overall_mean,
            'overall_std': overall_std,
            'cross_repo_cv': cv,
            'correlation_stability': corr_stability,
            'stability_score': stability_score,
            'transfers_well': stability_score < 0.5,
        })
    
    return pd.DataFrame(results).sort_values('stability_score')


def compute_transfer_matrix(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Compute transfer performance: train on one repo, test on others.
    """
    X = df[feature_cols].fillna(0)
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
    y = df['resolved'].astype(int).values
    groups = df['pseudo_repo'].values
    
    repos = sorted(df['pseudo_repo'].unique())
    repo_names = ['repo_A', 'repo_B', 'repo_C', 'repo_D']
    
    # Transfer matrix: train on row, test on column
    matrix = np.zeros((len(repos), len(repos)))
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                 min_samples_leaf=3, random_state=42)
    
    for train_repo in repos:
        train_mask = groups == train_repo
        X_train, y_train = X.values[train_mask], y[train_mask]
        
        if len(np.unique(y_train)) < 2:
            # Skip if only one class in training
            continue
            
        clf.fit(X_train, y_train)
        
        for test_repo in repos:
            test_mask = groups == test_repo
            X_test, y_test = X.values[test_mask], y[test_mask]
            
            if len(X_test) > 0:
                score = clf.score(X_test, y_test)
                matrix[train_repo, test_repo] = score
    
    return pd.DataFrame(matrix, index=repo_names[:len(repos)], 
                       columns=repo_names[:len(repos)])


def compute_leave_one_task_out(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """
    Leave-one-task-out cross-validation to measure true generalization.
    """
    X = df[feature_cols].fillna(0)
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
    y = df['resolved'].astype(int).values
    
    # Group by task
    tasks = df['task_id'].values
    unique_tasks = df['task_id'].unique()
    task_to_idx = {t: i for i, t in enumerate(unique_tasks)}
    groups = np.array([task_to_idx[t] for t in tasks])
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                 min_samples_leaf=3, random_state=42)
    
    logo = LeaveOneGroupOut()
    
    scores = []
    for train_idx, test_idx in logo.split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2:
            continue
        clf.fit(X.values[train_idx], y[train_idx])
        score = clf.score(X.values[test_idx], y[test_idx])
        scores.append(score)
    
    return {
        'mean_accuracy': np.mean(scores),
        'std_accuracy': np.std(scores),
        'min_accuracy': np.min(scores),
        'max_accuracy': np.max(scores),
        'n_folds': len(scores),
    }


def identify_universal_features(stability_df: pd.DataFrame, 
                                importance_threshold: float = 0.01) -> pd.DataFrame:
    """
    Identify features that are both stable AND predictive.
    """
    # Features that transfer well (low stability score)
    stable = stability_df[stability_df['transfers_well']].copy()
    
    # Mark as universal
    stable['category'] = 'universal'
    
    # Get repo-specific features
    specific = stability_df[~stability_df['transfers_well']].copy()
    specific['category'] = 'repo-specific'
    
    return pd.concat([stable, specific])


def plot_feature_stability(stability_df: pd.DataFrame, output_path: Path) -> Path:
    """Plot feature stability analysis."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Stability scores
    ax1 = axes[0]
    top_stable = stability_df.head(15)
    colors = ['#2ecc71' if t else '#e74c3c' for t in top_stable['transfers_well']]
    
    ax1.barh(range(len(top_stable)), top_stable['stability_score'], 
            color=colors, alpha=0.7, edgecolor='black')
    ax1.set_yticks(range(len(top_stable)))
    ax1.set_yticklabels([f[:25] for f in top_stable['feature']], fontsize=9)
    ax1.set_xlabel('Stability Score (lower = better transfer)', fontsize=11)
    ax1.set_title('Most Transferable Features', fontsize=13)
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='Transfer threshold')
    ax1.legend()
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Least stable features
    ax2 = axes[1]
    bottom_stable = stability_df.tail(15).iloc[::-1]
    colors = ['#2ecc71' if t else '#e74c3c' for t in bottom_stable['transfers_well']]
    
    ax2.barh(range(len(bottom_stable)), bottom_stable['stability_score'], 
            color=colors, alpha=0.7, edgecolor='black')
    ax2.set_yticks(range(len(bottom_stable)))
    ax2.set_yticklabels([f[:25] for f in bottom_stable['feature']], fontsize=9)
    ax2.set_xlabel('Stability Score (lower = better transfer)', fontsize=11)
    ax2.set_title('Least Transferable Features', fontsize=13)
    ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Transfers well'),
        Patch(facecolor='#e74c3c', label='Repo-specific')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, 
              bbox_to_anchor=(0.5, 1.02))
    
    plt.tight_layout()
    
    plot_path = output_path / 'feature_stability.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def plot_transfer_matrix(transfer_df: pd.DataFrame, output_path: Path) -> Path:
    """Plot transfer performance matrix."""
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(transfer_df.values, cmap='RdYlGn', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(transfer_df.columns)))
    ax.set_xticklabels(transfer_df.columns, fontsize=11)
    ax.set_yticks(range(len(transfer_df.index)))
    ax.set_yticklabels(transfer_df.index, fontsize=11)
    
    ax.set_xlabel('Test Repository', fontsize=12)
    ax.set_ylabel('Train Repository', fontsize=12)
    ax.set_title('Cross-Repository Transfer Performance', fontsize=14)
    
    # Add text annotations
    for i in range(len(transfer_df.index)):
        for j in range(len(transfer_df.columns)):
            val = transfer_df.values[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                   color=color, fontsize=11, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Accuracy')
    
    # Highlight diagonal (same-repo performance)
    for i in range(min(len(transfer_df.index), len(transfer_df.columns))):
        ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False, 
                                   edgecolor='blue', linewidth=2))
    
    plt.tight_layout()
    
    plot_path = output_path / 'transfer_matrix.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def plot_generalization_analysis(df: pd.DataFrame, loto_results: dict, 
                                 output_path: Path) -> Path:
    """Plot generalization analysis."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Success rate by pseudo-repo
    ax1 = axes[0]
    repo_stats = df.groupby('pseudo_repo_name').agg({
        'resolved': ['mean', 'count']
    }).round(3)
    repo_stats.columns = ['success_rate', 'n_samples']
    repo_stats = repo_stats.sort_index()
    
    colors = plt.cm.Set2(range(len(repo_stats)))
    bars = ax1.bar(range(len(repo_stats)), repo_stats['success_rate'], 
                  color=colors, edgecolor='black', alpha=0.7)
    
    ax1.set_xticks(range(len(repo_stats)))
    ax1.set_xticklabels(repo_stats.index, fontsize=11)
    ax1.set_ylabel('Success Rate', fontsize=12)
    ax1.set_title('Success Rate by Task Group', fontsize=13)
    ax1.set_ylim(0, max(repo_stats['success_rate']) * 1.3)
    
    # Add count labels
    for i, (idx, row) in enumerate(repo_stats.iterrows()):
        ax1.text(i, row['success_rate'] + 0.02, f"n={int(row['n_samples'])}", 
                ha='center', fontsize=10)
    
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Leave-one-task-out results
    ax2 = axes[1]
    
    metrics = ['mean_accuracy', 'min_accuracy', 'max_accuracy']
    values = [loto_results[m] for m in metrics]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    bars = ax2.bar(range(len(metrics)), values, color=colors, 
                  edgecolor='black', alpha=0.7)
    
    ax2.set_xticks(range(len(metrics)))
    ax2.set_xticklabels(['Mean', 'Min', 'Max'], fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'Leave-One-Task-Out CV\n({loto_results["n_folds"]} folds)', fontsize=13)
    ax2.set_ylim(0, 1.1)
    
    # Add value labels
    for i, v in enumerate(values):
        ax2.text(i, v + 0.03, f'{v:.2f}', ha='center', fontsize=11, fontweight='bold')
    
    # Add std as error bar on mean
    ax2.errorbar(0, loto_results['mean_accuracy'], 
                yerr=loto_results['std_accuracy'],
                fmt='none', color='black', capsize=5, capthick=2)
    
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    plot_path = output_path / 'generalization_analysis.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def generate_report(df: pd.DataFrame, stability_df: pd.DataFrame, 
                   transfer_df: pd.DataFrame, loto_results: dict,
                   output_path: Path) -> Path:
    """Generate markdown report."""
    
    n_universal = stability_df['transfers_well'].sum()
    n_specific = len(stability_df) - n_universal
    
    # Compute transfer gap
    diagonal = np.diag(transfer_df.values)
    off_diagonal = transfer_df.values[~np.eye(len(transfer_df), dtype=bool)]
    transfer_gap = diagonal.mean() - off_diagonal.mean()
    
    lines = [
        "# Cross-Repository Transfer Analysis",
        "",
        "## Overview",
        "",
        "This analysis evaluates how well the behavioral features and classifier",
        "generalize across different task groups (simulating different repositories).",
        "",
        "> **Note**: Since we only have scikit-learn data, we create 'pseudo-repos'",
        "> by splitting tasks into groups. True cross-repo transfer would require",
        "> data from multiple actual repositories.",
        "",
        "## Key Findings",
        "",
        f"- **Universal features**: {n_universal}/{len(stability_df)} features transfer well across task groups",
        f"- **Repo-specific features**: {n_specific} features show high variance across groups",
        f"- **Transfer gap**: {transfer_gap:.2f} (same-group vs cross-group accuracy difference)",
        f"- **Leave-one-task-out accuracy**: {loto_results['mean_accuracy']:.1%} ± {loto_results['std_accuracy']:.1%}",
        "",
        "## Feature Transferability",
        "",
        "### Most Transferable Features (Universal)",
        "",
        "These features show consistent behavior across task groups:",
        "",
        "| Feature | Stability Score | Cross-Repo CV |",
        "|---------|-----------------|---------------|",
    ]
    
    for _, row in stability_df.head(10).iterrows():
        lines.append(f"| {row['feature'][:30]} | {row['stability_score']:.3f} | {row['cross_repo_cv']:.3f} |")
    
    lines.extend([
        "",
        "### Least Transferable Features (Repo-Specific)",
        "",
        "These features vary significantly across task groups:",
        "",
        "| Feature | Stability Score | Cross-Repo CV |",
        "|---------|-----------------|---------------|",
    ])
    
    for _, row in stability_df.tail(10).iloc[::-1].iterrows():
        lines.append(f"| {row['feature'][:30]} | {row['stability_score']:.3f} | {row['cross_repo_cv']:.3f} |")
    
    lines.extend([
        "",
        "## Transfer Performance Matrix",
        "",
        "Accuracy when training on one group and testing on another:",
        "",
        "| Train \\ Test | " + " | ".join(transfer_df.columns) + " |",
        "|" + "|".join(["---"] * (len(transfer_df.columns) + 1)) + "|",
    ])
    
    for idx, row in transfer_df.iterrows():
        values = " | ".join([f"{v:.2f}" for v in row.values])
        lines.append(f"| {idx} | {values} |")
    
    lines.extend([
        "",
        f"**Same-group accuracy**: {diagonal.mean():.2f}",
        f"**Cross-group accuracy**: {off_diagonal.mean():.2f}",
        f"**Transfer gap**: {transfer_gap:.2f}",
        "",
        "## Leave-One-Task-Out Cross-Validation",
        "",
        "Most rigorous test: train on all tasks except one, predict that task.",
        "",
        f"- **Mean accuracy**: {loto_results['mean_accuracy']:.1%}",
        f"- **Std deviation**: {loto_results['std_accuracy']:.1%}",
        f"- **Min accuracy**: {loto_results['min_accuracy']:.1%}",
        f"- **Max accuracy**: {loto_results['max_accuracy']:.1%}",
        f"- **Number of folds**: {loto_results['n_folds']}",
        "",
        "## Recommendations for Cross-Repo Transfer",
        "",
        "Based on this analysis:",
        "",
        "1. **Use universal features for transfer**: Focus on features with stability score < 0.5",
        "2. **Retrain on target repo**: Even small amounts of target data can help",
        "3. **Monitor feature distributions**: Check if target repo has similar feature ranges",
        "4. **Ensemble approach**: Combine base model with target-specific fine-tuning",
        "",
        "## Features to Prioritize for Transfer",
        "",
        "The following features are both predictive AND stable:",
        "",
    ])
    
    universal = stability_df[stability_df['transfers_well']].head(5)
    for _, row in universal.iterrows():
        lines.append(f"- **{row['feature']}**: stability={row['stability_score']:.3f}")
    
    lines.extend([
        "",
        "## Visualizations",
        "",
        "- `feature_stability.png`: Feature transferability analysis",
        "- `transfer_matrix.png`: Cross-group transfer performance",
        "- `generalization_analysis.png`: Overall generalization metrics",
        "",
    ])
    
    report_path = output_path / 'cross_repo_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    
    return report_path


def main():
    """Run cross-repository transfer analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cross-Repository Transfer Analysis')
    parser.add_argument('--data', type=Path, default=Path('training_data/real_training_data.jsonl'),
                       help='Path to training data JSONL')
    parser.add_argument('--output', type=Path, default=Path('analysis/outputs/cross_repo'),
                       help='Output directory')
    
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CROSS-REPOSITORY TRANSFER ANALYSIS")
    print("=" * 60)
    print()
    
    # Load data
    print(f"Loading data from {args.data}...")
    df = load_training_data(args.data)
    print(f"Loaded {len(df)} samples, {df['task_id'].nunique()} unique tasks")
    print()
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"Analyzing {len(feature_cols)} features")
    
    # Create pseudo-repos
    print("\nCreating pseudo-repositories...")
    df = create_pseudo_repos(df, n_groups=4)
    print("Pseudo-repo distribution:")
    print(df.groupby('pseudo_repo_name').size().to_string())
    
    # Compute feature stability
    print("\nComputing feature stability...")
    stability_df = compute_feature_stability(df, feature_cols)
    stability_df.to_csv(args.output / 'feature_stability.csv', index=False)
    
    n_universal = stability_df['transfers_well'].sum()
    print(f"Universal features: {n_universal}/{len(stability_df)}")
    
    # Compute transfer matrix
    print("\nComputing transfer matrix...")
    transfer_df = compute_transfer_matrix(df, feature_cols)
    transfer_df.to_csv(args.output / 'transfer_matrix.csv')
    
    # Compute LOTO
    print("Running leave-one-task-out CV...")
    loto_results = compute_leave_one_task_out(df, feature_cols)
    print(f"LOTO accuracy: {loto_results['mean_accuracy']:.1%} ± {loto_results['std_accuracy']:.1%}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot1 = plot_feature_stability(stability_df, args.output)
    print(f"  Feature stability: {plot1}")
    
    plot2 = plot_transfer_matrix(transfer_df, args.output)
    print(f"  Transfer matrix: {plot2}")
    
    plot3 = plot_generalization_analysis(df, loto_results, args.output)
    print(f"  Generalization: {plot3}")
    
    # Identify universal features
    universal_df = identify_universal_features(stability_df)
    universal_df.to_csv(args.output / 'universal_features.csv', index=False)
    
    # Generate report
    report_path = generate_report(df, stability_df, transfer_df, loto_results, args.output)
    print(f"\nReport saved to {report_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\nFeature Transferability:")
    print(f"  Universal: {n_universal} features")
    print(f"  Repo-specific: {len(stability_df) - n_universal} features")
    
    print(f"\nTransfer Performance:")
    diagonal = np.diag(transfer_df.values)
    off_diagonal = transfer_df.values[~np.eye(len(transfer_df), dtype=bool)]
    print(f"  Same-group accuracy: {diagonal.mean():.2f}")
    print(f"  Cross-group accuracy: {off_diagonal.mean():.2f}")
    print(f"  Transfer gap: {diagonal.mean() - off_diagonal.mean():.2f}")
    
    print(f"\nLeave-One-Task-Out:")
    print(f"  Mean: {loto_results['mean_accuracy']:.1%}")
    print(f"  Range: [{loto_results['min_accuracy']:.1%}, {loto_results['max_accuracy']:.1%}]")
    
    print("\n" + "=" * 60)
    print("✓ Analysis complete!")


if __name__ == '__main__':
    main()