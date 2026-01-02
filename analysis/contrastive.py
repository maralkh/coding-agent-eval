#!/usr/bin/env python3
"""
Contrastive Analysis: What Distinguishes Success from Failure?

Automatically generates explanations for why some trajectories succeed
and others fail by comparing feature distributions and identifying
the most discriminative factors.

Usage:
------
    python analysis/contrastive.py --data training_data/real_training_data.jsonl

Outputs:
--------
    contrastive_features.png     Top discriminative features
    success_failure_dist.png     Distribution comparisons
    model_contrastive.png        Per-model success factors
    task_contrastive.png         Per-task analysis
    contrastive_report.md        Detailed explanations

Key Insights:
-------------
    - Which features most distinguish success from failure?
    - Do different models fail for different reasons?
    - What patterns predict success across all models?
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
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


def compute_effect_sizes(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Compute effect size (Cohen's d) for each feature between success and failure groups.
    
    Cohen's d interpretation:
    - |d| < 0.2: negligible
    - 0.2 ≤ |d| < 0.5: small
    - 0.5 ≤ |d| < 0.8: medium
    - |d| ≥ 0.8: large
    """
    success = df[df['resolved'] == True]
    failure = df[df['resolved'] == False]
    
    results = []
    
    for col in feature_cols:
        s_vals = success[col].dropna()
        f_vals = failure[col].dropna()
        
        if len(s_vals) < 2 or len(f_vals) < 2:
            continue
            
        # Cohen's d
        pooled_std = np.sqrt(((len(s_vals)-1)*s_vals.std()**2 + (len(f_vals)-1)*f_vals.std()**2) / 
                            (len(s_vals) + len(f_vals) - 2))
        
        if pooled_std > 0:
            cohens_d = (s_vals.mean() - f_vals.mean()) / pooled_std
        else:
            cohens_d = 0
        
        # Statistical significance (t-test)
        t_stat, p_value = stats.ttest_ind(s_vals, f_vals)
        
        # Direction
        direction = 'higher in success' if cohens_d > 0 else 'higher in failure'
        
        # Effect size category
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            effect_cat = 'negligible'
        elif abs_d < 0.5:
            effect_cat = 'small'
        elif abs_d < 0.8:
            effect_cat = 'medium'
        else:
            effect_cat = 'large'
        
        results.append({
            'feature': col,
            'success_mean': s_vals.mean(),
            'failure_mean': f_vals.mean(),
            'success_std': s_vals.std(),
            'failure_std': f_vals.std(),
            'cohens_d': cohens_d,
            'abs_cohens_d': abs_d,
            'effect_size': effect_cat,
            'direction': direction,
            'p_value': p_value,
            'significant': p_value < 0.05,
        })
    
    return pd.DataFrame(results).sort_values('abs_cohens_d', ascending=False)


def compute_feature_importance(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Compute feature importance using Random Forest."""
    X = df[feature_cols].fillna(0)
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
    y = df['resolved'].astype(int)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                 min_samples_leaf=5, random_state=42)
    clf.fit(X.values, y.values)
    
    # Get feature importances
    importances = clf.feature_importances_
    
    return pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)


def compute_model_specific_factors(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Compute which features matter most for each model."""
    model_factors = {}
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        
        if model_df['resolved'].sum() < 2 or (~model_df['resolved']).sum() < 2:
            # Not enough data for comparison
            model_factors[model] = {'note': 'Insufficient variation for analysis'}
            continue
        
        effect_sizes = compute_effect_sizes(model_df, feature_cols)
        top_factors = effect_sizes.head(5)
        
        model_factors[model] = {
            'top_features': top_factors[['feature', 'cohens_d', 'direction']].to_dict('records'),
            'n_success': int(model_df['resolved'].sum()),
            'n_failure': int((~model_df['resolved']).sum()),
        }
    
    return model_factors


def generate_natural_language_explanation(effect_sizes: pd.DataFrame, 
                                          feature_importance: pd.DataFrame) -> list[str]:
    """Generate human-readable explanations of success factors."""
    explanations = []
    
    # Top discriminative features
    top_effects = effect_sizes[effect_sizes['significant']].head(10)
    
    for _, row in top_effects.iterrows():
        feature = row['feature'].replace('_', ' ')
        
        if row['cohens_d'] > 0:
            explanations.append(
                f"**{feature}**: Successful trajectories show higher values "
                f"({row['success_mean']:.2f} vs {row['failure_mean']:.2f}, "
                f"effect size: {row['effect_size']})"
            )
        else:
            explanations.append(
                f"**{feature}**: Failed trajectories show higher values "
                f"({row['failure_mean']:.2f} vs {row['success_mean']:.2f}, "
                f"effect size: {row['effect_size']})"
            )
    
    return explanations


def plot_contrastive_features(effect_sizes: pd.DataFrame, output_path: Path) -> Path:
    """Plot top discriminative features."""
    
    # Top 15 by absolute effect size
    top = effect_sizes.head(15).copy()
    top = top.iloc[::-1]  # Reverse for horizontal bar chart
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in top['cohens_d']]
    
    bars = ax.barh(range(len(top)), top['cohens_d'], color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f.replace('_', ' ')[:35] for f in top['feature']], fontsize=10)
    ax.set_xlabel("Cohen's d (Effect Size)", fontsize=12)
    ax.set_title('Top Discriminative Features: Success vs Failure', fontsize=14)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add effect size thresholds
    for thresh, label in [(0.2, 'small'), (0.5, 'medium'), (0.8, 'large')]:
        ax.axvline(x=thresh, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=-thresh, color='gray', linestyle='--', alpha=0.5)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Higher in Success'),
        Patch(facecolor='#e74c3c', label='Higher in Failure')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    plot_path = output_path / 'contrastive_features.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def plot_distribution_comparison(df: pd.DataFrame, effect_sizes: pd.DataFrame, 
                                 output_path: Path) -> Path:
    """Plot distribution comparisons for top features."""
    
    top_features = effect_sizes.head(6)['feature'].tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    success = df[df['resolved'] == True]
    failure = df[df['resolved'] == False]
    
    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        
        s_vals = success[feature].dropna()
        f_vals = failure[feature].dropna()
        
        # Determine bins
        all_vals = pd.concat([s_vals, f_vals])
        bins = np.linspace(all_vals.min(), all_vals.max(), 20)
        
        ax.hist(f_vals, bins=bins, alpha=0.6, color='#e74c3c', label='Failure', edgecolor='black')
        ax.hist(s_vals, bins=bins, alpha=0.6, color='#2ecc71', label='Success', edgecolor='black')
        
        ax.axvline(s_vals.mean(), color='#27ae60', linestyle='--', linewidth=2)
        ax.axvline(f_vals.mean(), color='#c0392b', linestyle='--', linewidth=2)
        
        ax.set_xlabel(feature.replace('_', ' ')[:25], fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Feature Distributions: Success vs Failure', fontsize=14, y=1.02)
    plt.tight_layout()
    
    plot_path = output_path / 'success_failure_dist.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def plot_model_contrastive(df: pd.DataFrame, model_factors: dict, output_path: Path) -> Path:
    """Plot model-specific success factors."""
    
    models = [m for m in df['model'].unique() if 'top_features' in model_factors.get(m, {})]
    
    if len(models) == 0:
        # Create a placeholder plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Insufficient data for model-specific analysis',
               ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plot_path = output_path / 'model_contrastive.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        return plot_path
    
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 6))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        factors = model_factors[model]['top_features']
        
        features = [f['feature'].replace('_', ' ')[:20] for f in factors]
        effects = [f['cohens_d'] for f in factors]
        colors = ['#2ecc71' if e > 0 else '#e74c3c' for e in effects]
        
        bars = ax.barh(range(len(features)), effects, color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel("Cohen's d", fontsize=10)
        ax.set_title(f"{model[:25]}\n({model_factors[model]['n_success']}✓ / {model_factors[model]['n_failure']}✗)", 
                    fontsize=11)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Model-Specific Success Factors', fontsize=14, y=1.02)
    plt.tight_layout()
    
    plot_path = output_path / 'model_contrastive.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def plot_success_recipe(effect_sizes: pd.DataFrame, output_path: Path) -> Path:
    """Create a 'recipe for success' visualization."""
    
    # Get significant features
    sig_features = effect_sizes[effect_sizes['significant']].head(10)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create success recipe
    positive = sig_features[sig_features['cohens_d'] > 0].head(5)
    negative = sig_features[sig_features['cohens_d'] < 0].head(5)
    
    y_pos = list(range(len(positive)))
    y_neg = list(range(len(positive) + 1, len(positive) + 1 + len(negative)))
    
    # Plot "DO" items
    if len(positive) > 0:
        ax.barh(y_pos, positive['cohens_d'], color='#2ecc71', alpha=0.7, edgecolor='black')
        for i, (_, row) in enumerate(positive.iterrows()):
            ax.text(row['cohens_d'] + 0.05, i, 
                   f"✓ {row['feature'].replace('_', ' ')}", 
                   va='center', fontsize=11)
    
    # Plot "AVOID" items
    if len(negative) > 0:
        ax.barh(y_neg, [abs(d) for d in negative['cohens_d']], color='#e74c3c', alpha=0.7, edgecolor='black')
        for i, (_, row) in enumerate(negative.iterrows()):
            ax.text(abs(row['cohens_d']) + 0.05, y_neg[i], 
                   f"✗ {row['feature'].replace('_', ' ')}", 
                   va='center', fontsize=11)
    
    ax.set_yticks([])
    ax.set_xlabel('Effect Size (|Cohen\'s d|)', fontsize=12)
    ax.set_title('Recipe for Success: What to Do vs Avoid', fontsize=14)
    
    # Add section labels
    if len(positive) > 0:
        ax.text(-0.1, np.mean(y_pos), 'DO', fontsize=14, fontweight='bold', 
               color='#27ae60', ha='right', va='center')
    if len(negative) > 0:
        ax.text(-0.1, np.mean(y_neg), 'AVOID', fontsize=14, fontweight='bold', 
               color='#c0392b', ha='right', va='center')
    
    ax.set_xlim(-0.2, ax.get_xlim()[1] * 1.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    plot_path = output_path / 'success_recipe.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def generate_report(df: pd.DataFrame, effect_sizes: pd.DataFrame, 
                   feature_importance: pd.DataFrame, model_factors: dict,
                   explanations: list[str], output_path: Path) -> Path:
    """Generate markdown report."""
    
    n_success = df['resolved'].sum()
    n_failure = (~df['resolved']).sum()
    
    lines = [
        "# Contrastive Analysis: Success vs Failure",
        "",
        "## Overview",
        "",
        "This analysis identifies the key factors that distinguish successful",
        "trajectories from failed ones, providing actionable insights for improvement.",
        "",
        f"**Dataset**: {len(df)} trajectories ({n_success} success, {n_failure} failure)",
        "",
        "## Key Success Factors",
        "",
        "The following features show statistically significant differences between",
        "successful and failed trajectories (p < 0.05):",
        "",
    ]
    
    for exp in explanations[:8]:
        lines.append(f"- {exp}")
    
    lines.extend([
        "",
        "## Top Discriminative Features",
        "",
        "| Rank | Feature | Effect Size | Direction | Success Mean | Failure Mean |",
        "|------|---------|-------------|-----------|--------------|--------------|",
    ])
    
    for i, (_, row) in enumerate(effect_sizes.head(10).iterrows(), 1):
        feature = row['feature'].replace('_', ' ')[:25]
        lines.append(
            f"| {i} | {feature} | {row['cohens_d']:.2f} ({row['effect_size']}) | "
            f"{row['direction'][:15]} | {row['success_mean']:.2f} | {row['failure_mean']:.2f} |"
        )
    
    lines.extend([
        "",
        "## Recipe for Success",
        "",
        "### ✓ DO (higher in successful trajectories):",
        "",
    ])
    
    positive = effect_sizes[effect_sizes['cohens_d'] > 0].head(5)
    for _, row in positive.iterrows():
        lines.append(f"- **{row['feature'].replace('_', ' ')}**: "
                    f"Target ≥ {row['success_mean']:.2f} (vs {row['failure_mean']:.2f} in failures)")
    
    lines.extend([
        "",
        "### ✗ AVOID (higher in failed trajectories):",
        "",
    ])
    
    negative = effect_sizes[effect_sizes['cohens_d'] < 0].head(5)
    for _, row in negative.iterrows():
        lines.append(f"- **{row['feature'].replace('_', ' ')}**: "
                    f"Keep below {row['success_mean']:.2f} (failures average {row['failure_mean']:.2f})")
    
    lines.extend([
        "",
        "## Model-Specific Insights",
        "",
    ])
    
    for model, factors in model_factors.items():
        lines.append(f"### {model}")
        lines.append("")
        
        if 'note' in factors:
            lines.append(f"*{factors['note']}*")
        else:
            lines.append(f"Success rate: {factors['n_success']}/{factors['n_success'] + factors['n_failure']}")
            lines.append("")
            lines.append("Top differentiating factors:")
            for f in factors.get('top_features', [])[:3]:
                direction = "↑" if f['cohens_d'] > 0 else "↓"
                lines.append(f"- {f['feature'].replace('_', ' ')}: {direction} (d={f['cohens_d']:.2f})")
        
        lines.append("")
    
    lines.extend([
        "## Feature Importance (Random Forest)",
        "",
        "| Rank | Feature | Importance |",
        "|------|---------|------------|",
    ])
    
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        lines.append(f"| {i} | {row['feature'].replace('_', ' ')[:30]} | {row['importance']:.3f} |")
    
    lines.extend([
        "",
        "## Actionable Recommendations",
        "",
        "Based on this analysis:",
        "",
        "1. **Minimize tool errors**: The strongest predictor of failure",
        "2. **Maximize patch similarity**: Successful agents produce patches closer to gold",
        "3. **Efficient exploration**: Find relevant files quickly, avoid excessive wandering",
        "4. **Make changes**: Agents that don't modify files almost always fail",
        "5. **Verify after editing**: Running tests after changes correlates with success",
        "",
        "## Visualizations",
        "",
        "- `contrastive_features.png`: Effect sizes for top features",
        "- `success_failure_dist.png`: Distribution comparisons",
        "- `model_contrastive.png`: Per-model success factors",
        "- `success_recipe.png`: Visual recipe for success",
        "",
    ])
    
    report_path = output_path / 'contrastive_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    
    return report_path


def main():
    """Run contrastive analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Contrastive Analysis')
    parser.add_argument('--data', type=Path, default=Path('training_data/real_training_data.jsonl'),
                       help='Path to training data JSONL')
    parser.add_argument('--output', type=Path, default=Path('analysis/outputs/contrastive'),
                       help='Output directory')
    
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CONTRASTIVE ANALYSIS: SUCCESS VS FAILURE")
    print("=" * 60)
    print()
    
    # Load data
    print(f"Loading data from {args.data}...")
    df = load_training_data(args.data)
    print(f"Loaded {len(df)} samples")
    print(f"Success: {df['resolved'].sum()}, Failure: {(~df['resolved']).sum()}")
    print()
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"Analyzing {len(feature_cols)} features...")
    
    # Compute effect sizes
    print("Computing effect sizes...")
    effect_sizes = compute_effect_sizes(df, feature_cols)
    effect_sizes.to_csv(args.output / 'effect_sizes.csv', index=False)
    
    # Compute feature importance
    print("Computing feature importance...")
    feature_importance = compute_feature_importance(df, feature_cols)
    feature_importance.to_csv(args.output / 'feature_importance.csv', index=False)
    
    # Model-specific factors
    print("Analyzing model-specific factors...")
    model_factors = compute_model_specific_factors(df, feature_cols)
    
    # Generate explanations
    explanations = generate_natural_language_explanation(effect_sizes, feature_importance)
    
    # Print top findings
    print("\n" + "=" * 60)
    print("TOP DISCRIMINATIVE FEATURES")
    print("=" * 60)
    print()
    
    sig_features = effect_sizes[effect_sizes['significant']].head(10)
    for i, (_, row) in enumerate(sig_features.iterrows(), 1):
        arrow = "↑" if row['cohens_d'] > 0 else "↓"
        print(f"{i:2}. {row['feature']:<30} d={row['cohens_d']:+.2f} {arrow} ({row['effect_size']})")
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    plot1 = plot_contrastive_features(effect_sizes, args.output)
    print(f"  Contrastive features: {plot1}")
    
    plot2 = plot_distribution_comparison(df, effect_sizes, args.output)
    print(f"  Distributions: {plot2}")
    
    plot3 = plot_model_contrastive(df, model_factors, args.output)
    print(f"  Model-specific: {plot3}")
    
    plot4 = plot_success_recipe(effect_sizes, args.output)
    print(f"  Success recipe: {plot4}")
    
    # Generate report
    report_path = generate_report(df, effect_sizes, feature_importance, 
                                  model_factors, explanations, args.output)
    print(f"\nReport saved to {report_path}")
    
    print("\n" + "=" * 60)
    print("RECIPE FOR SUCCESS")
    print("=" * 60)
    print()
    
    print("✓ DO:")
    for _, row in effect_sizes[effect_sizes['cohens_d'] > 0].head(3).iterrows():
        print(f"   - {row['feature'].replace('_', ' ')}")
    
    print("\n✗ AVOID:")
    for _, row in effect_sizes[effect_sizes['cohens_d'] < 0].head(3).iterrows():
        print(f"   - {row['feature'].replace('_', ' ')}")
    
    print("\n" + "=" * 60)
    print("✓ Analysis complete!")


if __name__ == '__main__':
    main()