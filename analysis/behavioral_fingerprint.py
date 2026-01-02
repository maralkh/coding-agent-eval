#!/usr/bin/env python3
"""
Behavioral Fingerprinting Analysis

Creates "personality profiles" for coding agents based on their behavioral patterns.
Visualizes how models differ in their approach to solving tasks.

Usage:
------
    # Basic usage with defaults
    python analysis/behavioral_fingerprint.py

    # Custom parameters
    python analysis/behavioral_fingerprint.py \
        --data training_data/real_training_data.jsonl \
        --output analysis/outputs/fingerprint

Arguments:
----------
    --data        Path to training data JSONL file
    --output      Output directory for results and plots

Outputs:
--------
    model_fingerprints.csv      Mean feature values per model
    fingerprint_2d.png          2D PCA projection of model behaviors
    radar_chart.png             Radar chart comparing behavioral dimensions
    behavioral_clusters.png     Hierarchical clustering dendrogram
    fingerprint_report.md       Markdown summary report

Key Questions Answered:
-----------------------
    - Which models behave similarly?
    - What behavioral dimensions differentiate models?
    - Are some models more "exploratory" vs "direct"?
    - How do success patterns differ across models?
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')


# Define behavioral dimensions for radar chart
BEHAVIORAL_DIMENSIONS = {
    'reasoning': [
        'reasoning_quality_score',
        'has_explicit_reasoning', 
        'hypothesizes_before_acting',
        'explains_changes',
    ],
    'exploration': [
        'exploration_pct',
        'files_explored',
        'exploration_efficiency',
        'search_to_read_ratio',
    ],
    'implementation': [
        'implementation_pct',
        'trajectory_efficiency',
        'steps_per_file',
        'edit_to_explore_ratio',
    ],
    'verification': [
        'verification_pct',
        'ran_tests',
        'verifies_after_change',
        'followed_test_after_change',
    ],
    'efficiency': [
        'trajectory_efficiency',
        'unnecessary_steps',
        'wasted_explorations',
        'stuck_episodes',
    ],
    'robustness': [
        'recovery_rate',
        'total_errors',
        'tool_errors_count',
        'had_regression',
    ],
}


def load_training_data(data_path: Path) -> pd.DataFrame:
    """Load training data from JSONL file."""
    records = []
    with open(data_path) as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get numeric feature columns (exclude metadata)."""
    exclude = {'task_id', 'model', 'provider', 'timestamp', 'resolved', 'exploration_strategy'}
    return [c for c in df.columns if c not in exclude and df[c].dtype in ['int64', 'float64', 'bool']]


def compute_model_fingerprints(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean feature vector per model."""
    feature_cols = get_feature_columns(df)
    
    # Convert booleans to int
    df_numeric = df.copy()
    for col in feature_cols:
        if df_numeric[col].dtype == 'bool':
            df_numeric[col] = df_numeric[col].astype(int)
    
    # Group by model and compute mean
    fingerprints = df_numeric.groupby('model')[feature_cols].mean()
    
    return fingerprints


def compute_dimension_scores(fingerprints: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate scores for each behavioral dimension."""
    dimension_scores = {}
    
    for dim_name, features in BEHAVIORAL_DIMENSIONS.items():
        # Get features that exist in the data
        available = [f for f in features if f in fingerprints.columns]
        if available:
            # Normalize each feature to [0, 1] then average
            dim_data = fingerprints[available].copy()
            for col in available:
                col_min, col_max = dim_data[col].min(), dim_data[col].max()
                if col_max > col_min:
                    dim_data[col] = (dim_data[col] - col_min) / (col_max - col_min)
                else:
                    dim_data[col] = 0.5
            
            # Invert "bad" metrics (higher = worse)
            invert_cols = ['unnecessary_steps', 'wasted_explorations', 'stuck_episodes', 
                          'total_errors', 'tool_errors_count', 'had_regression']
            for col in invert_cols:
                if col in dim_data.columns:
                    dim_data[col] = 1 - dim_data[col]
            
            dimension_scores[dim_name] = dim_data.mean(axis=1)
    
    return pd.DataFrame(dimension_scores)


def plot_2d_projection(fingerprints: pd.DataFrame, output_path: Path) -> Path:
    """Create 2D PCA projection of model behaviors."""
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(fingerprints.values)
    
    # PCA to 2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
    models = fingerprints.index.tolist()
    
    for i, model in enumerate(models):
        ax.scatter(X_2d[i, 0], X_2d[i, 1], 
                  s=300, c=colors[i % len(colors)], 
                  label=model, edgecolors='black', linewidth=2, zorder=3)
        ax.annotate(model, (X_2d[i, 0], X_2d[i, 1]), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=11, fontweight='bold')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title('Model Behavioral Fingerprints (PCA Projection)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    
    # Add legend
    ax.legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    
    plot_path = output_path / 'fingerprint_2d.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also return PCA loadings for interpretation
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=fingerprints.columns
    )
    loadings['abs_PC1'] = loadings['PC1'].abs()
    loadings = loadings.sort_values('abs_PC1', ascending=False)
    
    return plot_path, loadings.head(10)


def plot_radar_chart(dimension_scores: pd.DataFrame, output_path: Path) -> Path:
    """Create radar chart comparing behavioral dimensions."""
    
    categories = list(dimension_scores.columns)
    n_cats = len(categories)
    
    # Compute angle for each category
    angles = [n / float(n_cats) * 2 * np.pi for n in range(n_cats)]
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
    
    for i, model in enumerate(dimension_scores.index):
        values = dimension_scores.loc[model].values.tolist()
        values += values[:1]  # Complete the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, 
               label=model, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.capitalize() for c in categories], size=12)
    
    # Set radial limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], size=9)
    
    ax.set_title('Behavioral Profile by Model', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    
    plot_path = output_path / 'radar_chart.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def plot_hierarchical_clustering(fingerprints: pd.DataFrame, output_path: Path) -> Path:
    """Create dendrogram showing behavioral similarity between models."""
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(fingerprints.values)
    
    # Compute linkage
    Z = linkage(X_scaled, method='ward')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    dendrogram(Z, labels=fingerprints.index.tolist(), ax=ax,
              leaf_font_size=12, leaf_rotation=0)
    
    ax.set_title('Model Behavioral Clustering (Ward Linkage)', fontsize=14)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    
    plt.tight_layout()
    
    plot_path = output_path / 'behavioral_clusters.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def plot_feature_heatmap(fingerprints: pd.DataFrame, output_path: Path) -> Path:
    """Create heatmap of standardized feature values per model."""
    
    # Select most variable features
    feature_std = fingerprints.std()
    top_features = feature_std.nlargest(15).index.tolist()
    
    # Standardize for visualization
    data = fingerprints[top_features].copy()
    data = (data - data.mean()) / data.std()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    im = ax.imshow(data.values, cmap='RdYlGn', aspect='auto', vmin=-2, vmax=2)
    
    # Labels
    ax.set_xticks(range(len(top_features)))
    ax.set_xticklabels(top_features, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=11)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Standardized Value', fontsize=11)
    
    ax.set_title('Model Feature Comparison (Top 15 Variable Features)', fontsize=14)
    
    plt.tight_layout()
    
    plot_path = output_path / 'feature_heatmap.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def compute_model_similarity(fingerprints: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise similarity between models."""
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(fingerprints.values)
    
    # Cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(X_scaled)
    
    return pd.DataFrame(sim_matrix, 
                       index=fingerprints.index, 
                       columns=fingerprints.index)


def generate_report(
    fingerprints: pd.DataFrame,
    dimension_scores: pd.DataFrame,
    similarity: pd.DataFrame,
    pca_loadings: pd.DataFrame,
    output_path: Path
) -> Path:
    """Generate markdown report."""
    
    lines = [
        "# Behavioral Fingerprinting Analysis",
        "",
        "## Overview",
        "",
        "This analysis creates 'behavioral profiles' for each coding agent model,",
        "revealing how they differ in their approach to solving tasks.",
        "",
        "## Model Behavioral Profiles",
        "",
    ]
    
    # Dimension scores table
    lines.extend([
        "### Behavioral Dimension Scores",
        "",
        "Scores normalized to [0, 1] where higher = better performance in that dimension.",
        "",
        "| Model | " + " | ".join(dimension_scores.columns.str.capitalize()) + " |",
        "|-------|" + "|".join(["--------"] * len(dimension_scores.columns)) + "|",
    ])
    
    for model in dimension_scores.index:
        scores = dimension_scores.loc[model]
        row = f"| {model} | " + " | ".join([f"{s:.2f}" for s in scores]) + " |"
        lines.append(row)
    
    lines.extend([
        "",
        "### Key Differentiators",
        "",
    ])
    
    # Find strongest/weakest dimension per model
    for model in dimension_scores.index:
        scores = dimension_scores.loc[model]
        strongest = scores.idxmax()
        weakest = scores.idxmin()
        lines.append(f"- **{model}**: Strongest in _{strongest}_ ({scores[strongest]:.2f}), "
                    f"weakest in _{weakest}_ ({scores[weakest]:.2f})")
    
    lines.extend([
        "",
        "## Model Similarity",
        "",
        "Cosine similarity based on behavioral features (1.0 = identical behavior):",
        "",
    ])
    
    # Similarity matrix
    header = "| Model | " + " | ".join(similarity.columns) + " |"
    sep = "|-------|" + "|".join(["------"] * len(similarity.columns)) + "|"
    lines.extend([header, sep])
    
    for model in similarity.index:
        row = f"| {model} | " + " | ".join([f"{similarity.loc[model, m]:.2f}" for m in similarity.columns]) + " |"
        lines.append(row)
    
    lines.extend([
        "",
        "## PCA Interpretation",
        "",
        "Top features contributing to the first principal component (explains most variance):",
        "",
        "| Feature | PC1 Loading |",
        "|---------|-------------|",
    ])
    
    for feat, row in pca_loadings.head(10).iterrows():
        lines.append(f"| {feat} | {row['PC1']:+.3f} |")
    
    lines.extend([
        "",
        "## Insights",
        "",
    ])
    
    # Auto-generate insights based on data
    # Find most similar/different pairs (using upper triangle only)
    models = similarity.index.tolist()
    pairs = []
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i < j:
                pairs.append((m1, m2, similarity.loc[m1, m2]))
    
    pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
    most_similar = (pairs_sorted[0][0], pairs_sorted[0][1])
    max_sim = pairs_sorted[0][2]
    most_different = (pairs_sorted[-1][0], pairs_sorted[-1][1])
    min_sim = pairs_sorted[-1][2]
    
    lines.extend([
        f"1. **Most similar models**: {most_similar[0]} and {most_similar[1]} (similarity: {max_sim:.2f})",
        f"2. **Most different models**: {most_different[0]} and {most_different[1]} (similarity: {min_sim:.2f})",
        "",
    ])
    
    # Find exploration-heavy vs implementation-heavy
    if 'exploration' in dimension_scores.columns and 'implementation' in dimension_scores.columns:
        explore_heavy = dimension_scores['exploration'].idxmax()
        impl_heavy = dimension_scores['implementation'].idxmax()
        lines.extend([
            f"3. **Most exploratory**: {explore_heavy} (exploration score: {dimension_scores.loc[explore_heavy, 'exploration']:.2f})",
            f"4. **Most direct/implementation-focused**: {impl_heavy} (implementation score: {dimension_scores.loc[impl_heavy, 'implementation']:.2f})",
            "",
        ])
    
    lines.extend([
        "## Visualizations",
        "",
        "- `fingerprint_2d.png`: 2D PCA projection showing model positions in behavior space",
        "- `radar_chart.png`: Radar chart comparing behavioral dimensions",
        "- `behavioral_clusters.png`: Hierarchical clustering dendrogram",
        "- `feature_heatmap.png`: Heatmap of top distinguishing features",
        "",
    ])
    
    report_path = output_path / 'fingerprint_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    
    return report_path


def main():
    """Run behavioral fingerprinting analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Behavioral Fingerprinting Analysis')
    parser.add_argument('--data', type=Path, default=Path('training_data/training_data.jsonl'),
                       help='Path to training data JSONL')
    parser.add_argument('--output', type=Path, default=Path('analysis/outputs/fingerprint'),
                       help='Output directory')
    
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("BEHAVIORAL FINGERPRINTING ANALYSIS")
    print("=" * 60)
    print()
    
    # Load data
    print(f"Loading data from {args.data}...")
    df = load_training_data(args.data)
    print(f"Loaded {len(df)} samples from {df['model'].nunique()} models")
    print(f"Models: {sorted(df['model'].unique())}")
    print()
    
    # Compute fingerprints
    print("Computing model fingerprints...")
    fingerprints = compute_model_fingerprints(df)
    print(f"Fingerprint dimensions: {fingerprints.shape[1]} features")
    
    # Save fingerprints
    fingerprints.to_csv(args.output / 'model_fingerprints.csv')
    print(f"Saved fingerprints to {args.output / 'model_fingerprints.csv'}")
    print()
    
    # Compute dimension scores
    print("Computing behavioral dimension scores...")
    dimension_scores = compute_dimension_scores(fingerprints)
    dimension_scores.to_csv(args.output / 'dimension_scores.csv')
    
    # Print dimension scores
    print("\nBehavioral Dimensions (0-1 scale, higher = better):")
    print("-" * 70)
    print(dimension_scores.round(2).to_string())
    print()
    
    # Generate plots
    print("Generating visualizations...")
    
    plot_2d, pca_loadings = plot_2d_projection(fingerprints, args.output)
    print(f"  2D projection: {plot_2d}")
    
    plot_radar = plot_radar_chart(dimension_scores, args.output)
    print(f"  Radar chart: {plot_radar}")
    
    plot_cluster = plot_hierarchical_clustering(fingerprints, args.output)
    print(f"  Clustering: {plot_cluster}")
    
    plot_heatmap = plot_feature_heatmap(fingerprints, args.output)
    print(f"  Heatmap: {plot_heatmap}")
    
    # Compute similarity
    similarity = compute_model_similarity(fingerprints)
    similarity.to_csv(args.output / 'model_similarity.csv')
    
    print("\nModel Similarity Matrix:")
    print("-" * 50)
    print(similarity.round(2).to_string())
    print()
    
    # Generate report
    report_path = generate_report(
        fingerprints, dimension_scores, similarity, pca_loadings, args.output
    )
    print(f"Report saved to {report_path}")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    
    # Most similar/different (using upper triangle only)
    models = similarity.index.tolist()
    pairs = []
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i < j:  # Upper triangle only
                pairs.append((m1, m2, similarity.loc[m1, m2]))
    
    pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
    most_similar = (pairs_sorted[0][0], pairs_sorted[0][1])
    max_sim = pairs_sorted[0][2]
    most_different = (pairs_sorted[-1][0], pairs_sorted[-1][1])
    min_sim = pairs_sorted[-1][2]
    
    print(f"\nMost similar pair: {most_similar[0]} ↔ {most_similar[1]} ({max_sim:.2f})")
    print(f"Most different pair: {most_different[0]} ↔ {most_different[1]} ({min_sim:.2f})")
    
    # Strongest dimension per model
    print("\nModel strengths:")
    for model in dimension_scores.index:
        strongest = dimension_scores.loc[model].idxmax()
        score = dimension_scores.loc[model, strongest]
        print(f"  {model}: {strongest} ({score:.2f})")
    
    print("\n" + "=" * 60)
    print("✓ Analysis complete!")


if __name__ == '__main__':
    main()
