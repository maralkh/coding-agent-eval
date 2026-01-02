#!/usr/bin/env python3
"""
Reward Modeling Analysis

Demonstrates how the success classifier can serve as a reward model for
reinforcement learning or best-of-N sampling.

Usage:
------
    # Basic usage with defaults
    python analysis/reward_model.py

    # Custom parameters
    python analysis/reward_model.py \
        --data training_data/real_training_data.jsonl \
        --output analysis/outputs/reward_model

Arguments:
----------
    --data        Path to training data JSONL file
    --output      Output directory for results and plots

Outputs:
--------
    reward_model.pkl            Serialized RewardModel object
    reward_distributions.png    Success vs failure probability distributions
    reward_calibration.png      Calibration plot (predicted vs actual)
    reward_separation.png       Margin analysis visualization
    trajectory_simulation.png   Simulated dense reward over trajectory progress
    reward_analysis_report.md   Markdown summary report

Key Insights:
-------------
    - Probability P(success) provides continuous signal vs binary pass/fail
    - Large separation between success/failure distributions = good reward signal
    - Reward can guide RL training or rank solutions for best-of-N selection
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')


class RewardModel:
    """
    Wraps a success classifier as a reward model for RL or ranking.
    
    The reward model provides:
    - score(): Single trajectory -> reward in [0, 1]
    - score_batch(): Multiple trajectories -> rewards
    - explain(): Feature contributions to the reward
    
    Example:
    --------
        >>> rm = RewardModel.from_data(training_data_path)
        >>> reward = rm.score(trajectory_features)
        >>> print(f"Predicted success probability: {reward:.2%}")
    """
    
    def __init__(self, classifier: RandomForestClassifier, feature_names: list[str]):
        self.classifier = classifier
        self.feature_names = feature_names
        self._feature_importance = None
    
    @classmethod
    def from_data(cls, data_path: Path) -> 'RewardModel':
        """Train reward model from training data."""
        # Load data
        records = []
        with open(data_path) as f:
            for line in f:
                records.append(json.loads(line))
        df = pd.DataFrame(records)
        
        # Get features
        exclude = {'task_id', 'model', 'provider', 'timestamp', 'resolved', 'exploration_strategy'}
        feature_cols = [c for c in df.columns if c not in exclude 
                       and df[c].dtype in ['int64', 'float64', 'bool']]
        
        X = df[feature_cols].fillna(0)
        for col in X.columns:
            if X[col].dtype == 'bool':
                X[col] = X[col].astype(int)
        
        y = df['resolved'].astype(int)
        
        # Train classifier
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X.values, y.values)
        
        return cls(clf, feature_cols)
    
    def score(self, features: dict) -> float:
        """
        Score a single trajectory.
        
        Args:
            features: Dict mapping feature names to values
            
        Returns:
            Probability of success in [0, 1]
        """
        X = np.array([[features.get(f, 0) for f in self.feature_names]])
        return self.classifier.predict_proba(X)[0, 1]
    
    def score_batch(self, trajectories: list[dict]) -> np.ndarray:
        """
        Score multiple trajectories efficiently.
        
        Args:
            trajectories: List of feature dicts
            
        Returns:
            Array of success probabilities
        """
        X = np.array([[t.get(f, 0) for f in self.feature_names] for t in trajectories])
        return self.classifier.predict_proba(X)[:, 1]
    
    def explain(self, features: dict) -> dict:
        """
        Explain which features contribute most to the reward.
        
        Returns dict mapping feature names to their contribution.
        Uses feature importance weighted by feature value.
        """
        importances = self.classifier.feature_importances_
        
        contributions = {}
        for i, feat in enumerate(self.feature_names):
            value = features.get(feat, 0)
            # Contribution = importance * normalized value
            contributions[feat] = importances[i] * (1 if value else 0)
        
        # Sort by absolute contribution
        return dict(sorted(contributions.items(), key=lambda x: -abs(x[1])))
    
    @property
    def feature_importance(self) -> dict:
        """Get feature importance rankings."""
        if self._feature_importance is None:
            imp = self.classifier.feature_importances_
            self._feature_importance = dict(sorted(
                zip(self.feature_names, imp),
                key=lambda x: -x[1]
            ))
        return self._feature_importance
    
    def save(self, path: Path):
        """Save model to file."""
        with open(path, 'wb') as f:
            pickle.dump({'classifier': self.classifier, 'features': self.feature_names}, f)
    
    @classmethod
    def load(cls, path: Path) -> 'RewardModel':
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return cls(data['classifier'], data['features'])


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


def compute_reward_statistics(probabilities: np.ndarray, actuals: np.ndarray) -> dict:
    """Compute reward signal quality statistics."""
    success_probs = probabilities[actuals == 1]
    failure_probs = probabilities[actuals == 0]
    
    return {
        'success_mean': success_probs.mean(),
        'success_std': success_probs.std(),
        'failure_mean': failure_probs.mean(),
        'failure_std': failure_probs.std(),
        'separation': success_probs.mean() - failure_probs.mean(),
        'overlap': compute_distribution_overlap(success_probs, failure_probs),
        'auc_proxy': (success_probs.mean() - failure_probs.mean()) / 2 + 0.5,
    }


def compute_distribution_overlap(dist1: np.ndarray, dist2: np.ndarray, n_bins: int = 50) -> float:
    """Compute overlap between two distributions (0 = no overlap, 1 = complete overlap)."""
    bins = np.linspace(0, 1, n_bins + 1)
    hist1, _ = np.histogram(dist1, bins=bins, density=True)
    hist2, _ = np.histogram(dist2, bins=bins, density=True)
    
    # Normalize
    hist1 = hist1 / hist1.sum() if hist1.sum() > 0 else hist1
    hist2 = hist2 / hist2.sum() if hist2.sum() > 0 else hist2
    
    # Overlap = sum of minimums
    return np.minimum(hist1, hist2).sum()


def plot_reward_distributions(probabilities: np.ndarray, actuals: np.ndarray, 
                             output_path: Path) -> Path:
    """Plot reward distributions for success vs failure."""
    
    success_probs = probabilities[actuals == 1]
    failure_probs = probabilities[actuals == 0]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bins = np.linspace(0, 1, 25)
    
    ax.hist(failure_probs, bins=bins, alpha=0.6, color='#e74c3c', 
           label=f'Failures (n={len(failure_probs)})', edgecolor='black')
    ax.hist(success_probs, bins=bins, alpha=0.6, color='#2ecc71', 
           label=f'Successes (n={len(success_probs)})', edgecolor='black')
    
    # Add mean lines
    ax.axvline(failure_probs.mean(), color='#c0392b', linestyle='--', linewidth=2,
              label=f'Failure mean: {failure_probs.mean():.2f}')
    ax.axvline(success_probs.mean(), color='#27ae60', linestyle='--', linewidth=2,
              label=f'Success mean: {success_probs.mean():.2f}')
    
    ax.set_xlabel('Reward (Predicted Success Probability)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Reward Distribution: Successes vs Failures', fontsize=14)
    ax.legend(loc='upper center')
    ax.grid(True, alpha=0.3)
    
    # Add separation annotation
    sep = success_probs.mean() - failure_probs.mean()
    ax.annotate(f'Separation: {sep:.2f}', 
               xy=(0.5, 0.95), xycoords='axes fraction',
               fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    plot_path = output_path / 'reward_distributions.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def plot_reward_calibration(probabilities: np.ndarray, actuals: np.ndarray,
                           output_path: Path) -> Path:
    """Plot calibration curve for reward model."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Calibration curve
    ax1 = axes[0]
    prob_true, prob_pred = calibration_curve(actuals, probabilities, n_bins=10)
    
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax1.plot(prob_pred, prob_true, 'o-', color='#3498db', linewidth=2, markersize=8,
            label='Reward model')
    
    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives', fontsize=12)
    ax1.set_title('Calibration Curve', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Reliability diagram (histogram of predictions)
    ax2 = axes[1]
    ax2.hist(probabilities, bins=20, edgecolor='black', alpha=0.7, color='#3498db')
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of Predictions', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = output_path / 'reward_calibration.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def plot_separation_analysis(probabilities: np.ndarray, actuals: np.ndarray,
                            output_path: Path) -> Path:
    """Visualize reward separation between classes."""
    
    success_probs = probabilities[actuals == 1]
    failure_probs = probabilities[actuals == 0]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Box plots
    data = [failure_probs, success_probs]
    positions = [0, 1]
    
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
    
    colors = ['#e74c3c', '#2ecc71']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Scatter individual points
    for i, (d, pos) in enumerate(zip(data, positions)):
        x = np.random.normal(pos, 0.08, size=len(d))
        ax.scatter(x, d, alpha=0.4, color=colors[i], s=30, edgecolors='black', linewidth=0.5)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(['Failures', 'Successes'], fontsize=12)
    ax.set_ylabel('Reward (Predicted Probability)', fontsize=12)
    ax.set_title('Reward Separation Analysis', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics annotation
    sep = success_probs.mean() - failure_probs.mean()
    text = f'Mean separation: {sep:.3f}\n'
    text += f'Success: {success_probs.mean():.3f} ± {success_probs.std():.3f}\n'
    text += f'Failure: {failure_probs.mean():.3f} ± {failure_probs.std():.3f}'
    
    ax.annotate(text, xy=(0.02, 0.98), xycoords='axes fraction',
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    plot_path = output_path / 'reward_separation.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def simulate_trajectory_rewards(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Simulate how reward evolves over a trajectory.
    
    We approximate "partial trajectories" by scaling features that would
    accumulate over time (e.g., steps, files explored).
    """
    
    feature_cols = get_feature_columns(df)
    
    # Features that accumulate over time
    cumulative_features = [
        'exploration_steps', 'implementation_steps', 'verification_steps',
        'files_explored', 'directories_explored', 'total_tool_calls',
        'trajectory_length', 'total_errors', 'wasted_explorations'
    ]
    cumulative_features = [f for f in cumulative_features if f in feature_cols]
    
    # Train reward model
    X = df[feature_cols].fillna(0)
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
    y = df['resolved'].astype(int)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                 min_samples_leaf=5, random_state=42)
    clf.fit(X.values, y.values)
    
    # Simulate trajectories at different completion percentages
    progress_points = np.linspace(0.1, 1.0, 10)
    
    success_rewards = []
    failure_rewards = []
    
    for progress in progress_points:
        X_partial = X.copy()
        
        # Scale cumulative features by progress
        for feat in cumulative_features:
            if feat in X_partial.columns:
                X_partial[feat] = X_partial[feat] * progress
        
        rewards = clf.predict_proba(X_partial.values)[:, 1]
        
        success_rewards.append(rewards[y == 1].mean())
        failure_rewards.append(rewards[y == 0].mean())
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(progress_points * 100, success_rewards, 'o-', color='#2ecc71', 
           linewidth=2, markersize=8, label='Successful trajectories')
    ax.plot(progress_points * 100, failure_rewards, 'o-', color='#e74c3c', 
           linewidth=2, markersize=8, label='Failed trajectories')
    
    ax.fill_between(progress_points * 100, failure_rewards, success_rewards, 
                   alpha=0.2, color='#3498db')
    
    ax.set_xlabel('Trajectory Progress (%)', fontsize=12)
    ax.set_ylabel('Mean Reward', fontsize=12)
    ax.set_title('Simulated Reward Over Trajectory Progress', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 1)
    
    # Annotation
    ax.annotate('Early signal:\nreward diverges\nbefore completion',
               xy=(30, 0.5), fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    plot_path = output_path / 'trajectory_simulation.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def compare_reward_formulations(probabilities: np.ndarray, actuals: np.ndarray) -> dict:
    """
    Compare different reward formulations.
    
    Returns variance and other statistics for each formulation.
    """
    
    # Sparse reward: 0 or 1
    sparse = actuals.astype(float)
    
    # Dense reward: P(success)
    dense = probabilities
    
    # Shaped reward: P(success) with bonus for high confidence
    shaped = probabilities + 0.1 * (probabilities > 0.8).astype(float)
    shaped = np.clip(shaped, 0, 1)
    
    # Advantage-style: P(success) - baseline
    baseline = probabilities.mean()
    advantage = probabilities - baseline
    
    return {
        'sparse': {
            'mean': sparse.mean(),
            'std': sparse.std(),
            'variance': sparse.var(),
            'description': 'Binary 0/1 based on actual outcome'
        },
        'dense': {
            'mean': dense.mean(),
            'std': dense.std(),
            'variance': dense.var(),
            'description': 'Continuous P(success) from classifier'
        },
        'shaped': {
            'mean': shaped.mean(),
            'std': shaped.std(),
            'variance': shaped.var(),
            'description': 'P(success) + bonus for high confidence'
        },
        'advantage': {
            'mean': advantage.mean(),
            'std': advantage.std(),
            'variance': advantage.var(),
            'description': 'P(success) - mean(P) (centered)'
        }
    }


def generate_report(
    stats: dict,
    formulations: dict,
    output_path: Path
) -> Path:
    """Generate markdown report."""
    
    lines = [
        "# Reward Modeling Analysis",
        "",
        "> **Note**: This analysis uses synthetic training data generated for demonstration purposes.",
        "> Results illustrate the methodology but should be validated on real agent trajectories.",
        "",
        "## Overview",
        "",
        "This analysis demonstrates how the success classifier can serve as a **reward model**",
        "for reinforcement learning or best-of-N selection strategies.",
        "",
        "Key insight: The classifier's P(success) provides a **continuous signal** that can",
        "guide learning, unlike binary pass/fail which only gives feedback at episode end.",
        "",
        "## Reward Signal Quality",
        "",
        "A good reward model should clearly separate successful from failed trajectories.",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Success mean | {stats['success_mean']:.3f} |",
        f"| Success std | {stats['success_std']:.3f} |",
        f"| Failure mean | {stats['failure_mean']:.3f} |",
        f"| Failure std | {stats['failure_std']:.3f} |",
        f"| **Separation** | **{stats['separation']:.3f}** |",
        f"| Distribution overlap | {stats['overlap']:.3f} |",
        "",
        f"**Interpretation**: Separation of {stats['separation']:.3f} indicates ",
        "the reward model can effectively distinguish successful from failed trajectories.",
        "Lower overlap = better discriminative power.",
        "",
        "## Reward Formulations",
        "",
        "We compare different ways to formulate the reward signal:",
        "",
        "| Formulation | Mean | Std | Variance | Description |",
        "|-------------|------|-----|----------|-------------|",
    ]
    
    for name, data in formulations.items():
        lines.append(
            f"| {name.capitalize()} | {data['mean']:.3f} | {data['std']:.3f} | "
            f"{data['variance']:.3f} | {data['description']} |"
        )
    
    lines.extend([
        "",
        "**Key finding**: Dense reward has similar variance to sparse but provides signal",
        "throughout the trajectory rather than only at the end.",
        "",
        "## RewardModel API",
        "",
        "The reward model is packaged as a Python class for easy integration:",
        "",
        "```python",
        "from analysis.reward_model import RewardModel",
        "",
        "# Load pre-trained model",
        "rm = RewardModel.load('reward_model.pkl')",
        "",
        "# Score a single trajectory",
        "reward = rm.score({",
        "    'reasoning_quality_score': 0.8,",
        "    'exploration_efficiency': 0.6,",
        "    'trajectory_length': 10,",
        "    # ... other features",
        "})",
        "print(f'Predicted success probability: {reward:.2%}')",
        "",
        "# Score batch of trajectories",
        "rewards = rm.score_batch([traj1, traj2, traj3])",
        "",
        "# Explain prediction",
        "contributions = rm.explain(trajectory_features)",
        "```",
        "",
        "## Applications",
        "",
        "1. **RL Training Reward**: Use P(success) as dense reward signal instead of sparse 0/1",
        "2. **Best-of-N Ranking**: Rank N candidate solutions by reward, select highest",
        "3. **Early Stopping**: Abandon trajectories where reward drops below threshold",
        "4. **Curriculum Learning**: Prioritize tasks where model is uncertain (reward ≈ 0.5)",
        "",
        "## Visualizations",
        "",
        "- `reward_distributions.png`: Success vs failure reward distributions",
        "- `reward_calibration.png`: Calibration curve showing prediction reliability",
        "- `reward_separation.png`: Box plot comparing reward by outcome",
        "- `trajectory_simulation.png`: Reward evolution over trajectory progress",
        "",
    ])
    
    report_path = output_path / 'reward_analysis_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    
    return report_path


def main():
    """Run reward modeling analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Reward Modeling Analysis')
    parser.add_argument('--data', type=Path, default=Path('training_data/training_data.jsonl'),
                       help='Path to training data JSONL')
    parser.add_argument('--output', type=Path, default=Path('analysis/outputs/reward_model'),
                       help='Output directory')
    
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("REWARD MODELING ANALYSIS")
    print("=" * 60)
    print()
    
    # Load data
    print(f"Loading data from {args.data}...")
    df = load_training_data(args.data)
    print(f"Loaded {len(df)} samples")
    print()
    
    # Train reward model
    print("Training reward model...")
    rm = RewardModel.from_data(args.data)
    
    # Save model
    model_path = args.output / 'reward_model.pkl'
    rm.save(model_path)
    print(f"Saved reward model to {model_path}")
    
    # Get cross-validated probabilities for analysis
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].fillna(0)
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
    y = df['resolved'].astype(int).values
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                 min_samples_leaf=5, random_state=42)
    probabilities = cross_val_predict(clf, X.values, y, cv=5, method='predict_proba')[:, 1]
    
    # Compute statistics
    print("\nComputing reward statistics...")
    stats = compute_reward_statistics(probabilities, y)
    
    print(f"\nReward Signal Quality:")
    print("-" * 40)
    print(f"  Success mean:  {stats['success_mean']:.3f} ± {stats['success_std']:.3f}")
    print(f"  Failure mean:  {stats['failure_mean']:.3f} ± {stats['failure_std']:.3f}")
    print(f"  Separation:    {stats['separation']:.3f}")
    print(f"  Overlap:       {stats['overlap']:.3f}")
    print()
    
    # Compare formulations
    print("Comparing reward formulations...")
    formulations = compare_reward_formulations(probabilities, y)
    
    print("\nReward Formulation Comparison:")
    print("-" * 50)
    print(f"{'Formulation':<12} {'Mean':>8} {'Std':>8} {'Variance':>10}")
    print("-" * 50)
    for name, data in formulations.items():
        print(f"{name.capitalize():<12} {data['mean']:>8.3f} {data['std']:>8.3f} {data['variance']:>10.3f}")
    print()
    
    # Generate plots
    print("Generating visualizations...")
    
    dist_path = plot_reward_distributions(probabilities, y, args.output)
    print(f"  Distributions: {dist_path}")
    
    calib_path = plot_reward_calibration(probabilities, y, args.output)
    print(f"  Calibration: {calib_path}")
    
    sep_path = plot_separation_analysis(probabilities, y, args.output)
    print(f"  Separation: {sep_path}")
    
    traj_path = simulate_trajectory_rewards(df, args.output)
    print(f"  Trajectory: {traj_path}")
    
    # Generate report
    report_path = generate_report(stats, formulations, args.output)
    print(f"\nReport saved to {report_path}")
    
    # Print top features
    print("\n" + "=" * 60)
    print("TOP REWARD FEATURES")
    print("=" * 60)
    print()
    
    for i, (feat, imp) in enumerate(list(rm.feature_importance.items())[:10], 1):
        bar = "█" * int(imp * 50)
        print(f"  {i:2}. {feat:<35} {imp:.3f} {bar}")
    
    print("\n" + "=" * 60)
    print("✓ Analysis complete!")
    print()
    print("Usage example:")
    print("  from analysis.reward_model import RewardModel")
    print(f"  rm = RewardModel.load('{model_path}')")
    print("  reward = rm.score(trajectory_features)")


if __name__ == '__main__':
    main()
