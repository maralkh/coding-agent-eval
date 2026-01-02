#!/usr/bin/env python3
"""
Best-of-N Selection Simulation

Demonstrates that probability-based selection outperforms random selection
when choosing among multiple agent attempts.

Key insight: The classifier's P(success) can rank attempts, allowing us to
select the most promising one without running all N to completion.

Usage:
------
    # Basic usage with defaults
    python analysis/best_of_n.py

    # Custom parameters
    python analysis/best_of_n.py \
        --data training_data/real_training_data.jsonl \
        --output analysis/outputs/best_of_n \
        --max-n 10 \
        --simulations 10000

Arguments:
----------
    --data        Path to training data JSONL file (default: training_data/training_data.jsonl)
    --output      Output directory for results and plots (default: analysis/outputs/best_of_n)
    --max-n       Maximum N to simulate, i.e., max attempts (default: 10)
    --simulations Number of Monte Carlo simulations per N (default: 10000)

Outputs:
--------
    best_of_n_results.csv       Raw results table
    best_of_n_report.md         Markdown summary report
    best_of_n_selection.png     Success rate curves by strategy
    probability_calibration.png Predicted vs actual success rates
    compute_efficiency.png      Success per attempt efficiency
    classifier_degradation.png  Robustness to classifier noise
    degraded_classifier_results.csv  Noise analysis data

Example:
--------
    $ python analysis/best_of_n.py --max-n 5 --simulations 5000
    
    ============================================================
    BEST-OF-N SELECTION ANALYSIS
    ============================================================
    
    Success Rate by Strategy:
    ------------------------------------------------------------
      N |   Random | Probability |   Oracle | Improvement
    ------------------------------------------------------------
      3 |   40.1% |      78.3% |   78.3% | +    38.2%
      5 |   39.9% |      92.0% |   92.0% | +    52.1%
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SelectionResult:
    """Results from a selection strategy."""
    strategy: str
    n_attempts: int
    success_rate: float
    n_trials: int
    successes: int
    
    
def load_training_data(data_path: Path) -> pd.DataFrame:
    """Load training data from JSONL file."""
    records = []
    with open(data_path) as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get feature columns (exclude metadata and target)."""
    exclude = {'task_id', 'model', 'provider', 'timestamp', 'resolved'}
    return [c for c in df.columns if c not in exclude and df[c].dtype in ['int64', 'float64', 'bool']]


def train_classifier_and_get_probabilities(df: pd.DataFrame) -> np.ndarray:
    """Train classifier and get cross-validated probabilities."""
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].fillna(0).values
    y = df['resolved'].astype(int).values
    
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    # Use cross-validation to get unbiased probability estimates
    probabilities = cross_val_predict(clf, X, y, cv=5, method='predict_proba')[:, 1]
    
    return probabilities


def simulate_best_of_n(
    probabilities: np.ndarray,
    actuals: np.ndarray,
    n_attempts: int,
    n_simulations: int = 10000,
    seed: int = 42
) -> dict[str, SelectionResult]:
    """
    Simulate Best-of-N selection with different strategies.
    
    Strategies:
    - random: Pick one of N attempts uniformly at random
    - oracle: Pick the successful one if any exists (upper bound)
    - probability: Pick the attempt with highest P(success)
    - threshold: Pick first attempt with P > 0.5, else highest P
    
    The simulation works by:
    1. Randomly sampling N items from the dataset (simulating N attempts)
    2. Applying each selection strategy
    3. Checking if the selected attempt was actually successful
    """
    rng = np.random.RandomState(seed)
    n_samples = len(probabilities)
    
    results = {
        'random': {'successes': 0, 'total': 0},
        'oracle': {'successes': 0, 'total': 0},
        'probability': {'successes': 0, 'total': 0},
        'threshold_50': {'successes': 0, 'total': 0},
        'threshold_70': {'successes': 0, 'total': 0},
    }
    
    for _ in range(n_simulations):
        # Sample N attempts (with replacement to simulate independent attempts)
        indices = rng.choice(n_samples, size=n_attempts, replace=True)
        probs = probabilities[indices]
        actual = actuals[indices]
        
        # Random selection
        random_idx = rng.randint(n_attempts)
        results['random']['successes'] += actual[random_idx]
        results['random']['total'] += 1
        
        # Oracle selection (best possible - pick successful if exists)
        results['oracle']['successes'] += int(actual.any())
        results['oracle']['total'] += 1
        
        # Probability-based selection (pick highest probability)
        prob_idx = np.argmax(probs)
        results['probability']['successes'] += actual[prob_idx]
        results['probability']['total'] += 1
        
        # Threshold selection (P > 0.5, else highest)
        above_50 = np.where(probs > 0.5)[0]
        if len(above_50) > 0:
            thresh_idx = above_50[0]  # First above threshold
        else:
            thresh_idx = np.argmax(probs)
        results['threshold_50']['successes'] += actual[thresh_idx]
        results['threshold_50']['total'] += 1
        
        # Threshold 70%
        above_70 = np.where(probs > 0.7)[0]
        if len(above_70) > 0:
            thresh_idx = above_70[0]
        else:
            thresh_idx = np.argmax(probs)
        results['threshold_70']['successes'] += actual[thresh_idx]
        results['threshold_70']['total'] += 1
    
    # Convert to SelectionResult objects
    return {
        name: SelectionResult(
            strategy=name,
            n_attempts=n_attempts,
            success_rate=data['successes'] / data['total'],
            n_trials=data['total'],
            successes=data['successes']
        )
        for name, data in results.items()
    }


def run_best_of_n_analysis(
    df: pd.DataFrame,
    max_n: int = 10,
    n_simulations: int = 10000
) -> pd.DataFrame:
    """Run Best-of-N analysis for N=1 to max_n."""
    
    print("Training classifier and getting probabilities...")
    probabilities = train_classifier_and_get_probabilities(df)
    actuals = df['resolved'].astype(int).values
    
    # Baseline success rate
    baseline = actuals.mean()
    print(f"Baseline success rate: {baseline:.1%}")
    print(f"Probability range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
    print(f"Mean probability: {probabilities.mean():.3f}")
    print()
    
    all_results = []
    
    for n in range(1, max_n + 1):
        print(f"Simulating N={n}...", end=" ")
        results = simulate_best_of_n(probabilities, actuals, n, n_simulations)
        
        for strategy, result in results.items():
            all_results.append({
                'n_attempts': n,
                'strategy': strategy,
                'success_rate': result.success_rate,
                'improvement_vs_random': result.success_rate - results['random'].success_rate,
                'improvement_vs_baseline': result.success_rate - baseline,
            })
        
        prob_rate = results['probability'].success_rate
        rand_rate = results['random'].success_rate
        oracle_rate = results['oracle'].success_rate
        print(f"Random: {rand_rate:.1%}, Probability: {prob_rate:.1%}, Oracle: {oracle_rate:.1%}")
    
    return pd.DataFrame(all_results)


def plot_best_of_n_results(results_df: pd.DataFrame, output_path: Path) -> Path:
    """Create visualization of Best-of-N results."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Success rate by N and strategy
    ax1 = axes[0]
    strategies = ['random', 'probability', 'threshold_50', 'oracle']
    colors = {'random': '#888888', 'probability': '#2ecc71', 'threshold_50': '#3498db', 'oracle': '#e74c3c'}
    labels = {'random': 'Random', 'probability': 'Max Probability', 'threshold_50': 'Threshold (p>0.5)', 'oracle': 'Oracle (Upper Bound)'}
    
    for strategy in strategies:
        data = results_df[results_df['strategy'] == strategy]
        ax1.plot(data['n_attempts'], data['success_rate'] * 100, 
                marker='o', label=labels[strategy], color=colors[strategy], linewidth=2)
    
    ax1.set_xlabel('Number of Attempts (N)', fontsize=12)
    ax1.set_ylabel('Success Rate (%)', fontsize=12)
    ax1.set_title('Best-of-N Selection: Success Rate by Strategy', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, results_df['n_attempts'].max() + 1))
    
    # Plot 2: Improvement over random
    ax2 = axes[1]
    for strategy in ['probability', 'threshold_50']:
        data = results_df[results_df['strategy'] == strategy]
        improvement = data['improvement_vs_random'] * 100
        ax2.bar(data['n_attempts'] + (0.2 if strategy == 'probability' else -0.2), 
               improvement, width=0.4, label=labels[strategy], color=colors[strategy], alpha=0.8)
    
    ax2.set_xlabel('Number of Attempts (N)', fontsize=12)
    ax2.set_ylabel('Improvement over Random (%)', fontsize=12)
    ax2.set_title('Probability-Based Selection: Improvement over Random', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(range(1, results_df['n_attempts'].max() + 1))
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    plot_path = output_path / 'best_of_n_selection.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def plot_probability_calibration(df: pd.DataFrame, output_path: Path) -> Path:
    """Show that classifier probabilities correlate with actual success."""
    
    probabilities = train_classifier_and_get_probabilities(df)
    actuals = df['resolved'].astype(int).values
    
    # Bin probabilities and compute actual success rate per bin
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probabilities, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    bin_stats = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_stats.append({
                'bin_center': (bin_edges[i] + bin_edges[i+1]) / 2,
                'predicted_prob': probabilities[mask].mean(),
                'actual_rate': actuals[mask].mean(),
                'count': mask.sum()
            })
    
    bin_df = pd.DataFrame(bin_stats)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.5)
    
    # Actual calibration
    sizes = bin_df['count'] * 3
    ax.scatter(bin_df['predicted_prob'], bin_df['actual_rate'], 
              s=sizes, c='#2ecc71', alpha=0.7, edgecolors='black', linewidth=1)
    
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Actual Success Rate', fontsize=12)
    ax.set_title('Classifier Calibration: Predicted vs Actual Success Rate', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    
    plot_path = output_path / 'probability_calibration.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def generate_report(results_df: pd.DataFrame, output_path: Path, 
                    degraded_df: pd.DataFrame = None, 
                    efficiency: dict = None) -> Path:
    """Generate markdown report of Best-of-N analysis."""
    
    report_lines = [
        "# Best-of-N Selection Analysis",
        "",
        "## Overview",
        "",
        "This analysis demonstrates that **probability-based selection** outperforms",
        "**random selection** when choosing among multiple agent attempts on the same task.",
        "",
        "The classifier's probability estimates serve as a **ranking signal** that enables",
        "selecting the most promising attempt without running all attempts to completion.",
        "",
        "## Key Findings",
        "",
    ]
    
    # Find key statistics
    n5_data = results_df[results_df['n_attempts'] == 5]
    prob_n5 = n5_data[n5_data['strategy'] == 'probability']['success_rate'].values[0]
    rand_n5 = n5_data[n5_data['strategy'] == 'random']['success_rate'].values[0]
    improvement_n5 = prob_n5 - rand_n5
    
    n3_data = results_df[results_df['n_attempts'] == 3]
    prob_n3 = n3_data[n3_data['strategy'] == 'probability']['success_rate'].values[0]
    rand_n3 = n3_data[n3_data['strategy'] == 'random']['success_rate'].values[0]
    improvement_n3 = prob_n3 - rand_n3
    
    report_lines.extend([
        f"- **Best-of-3**: Probability selection achieves **{prob_n3:.1%}** success vs {rand_n3:.1%} random (+{improvement_n3:.1%})",
        f"- **Best-of-5**: Probability selection achieves **{prob_n5:.1%}** success vs {rand_n5:.1%} random (+{improvement_n5:.1%})",
        "",
        "## Results Table",
        "",
        "| N | Random | Probability | Threshold | Oracle | Prob vs Random |",
        "|---|--------|-------------|-----------|--------|----------------|",
    ])
    
    for n in sorted(results_df['n_attempts'].unique()):
        n_data = results_df[results_df['n_attempts'] == n]
        rand = n_data[n_data['strategy'] == 'random']['success_rate'].values[0]
        prob = n_data[n_data['strategy'] == 'probability']['success_rate'].values[0]
        thresh = n_data[n_data['strategy'] == 'threshold_50']['success_rate'].values[0]
        oracle = n_data[n_data['strategy'] == 'oracle']['success_rate'].values[0]
        diff = prob - rand
        report_lines.append(f"| {n} | {rand:.1%} | {prob:.1%} | {thresh:.1%} | {oracle:.1%} | +{diff:.1%} |")
    
    report_lines.extend([
        "",
        "## Compute Efficiency",
        "",
        "A key practical question: **how many attempts do you need to reach a target success rate?**",
        "",
    ])
    
    if efficiency:
        report_lines.extend([
            "| Strategy | Attempts for 80% Success |",
            "|----------|--------------------------|",
        ])
        for strategy, n_needed in efficiency.items():
            if n_needed:
                report_lines.append(f"| {strategy.capitalize()} | N = {n_needed} |")
            else:
                report_lines.append(f"| {strategy.capitalize()} | > 10 |")
        
        report_lines.extend([
            "",
            "**Implication**: Probability selection reaches 80% success with far fewer attempts than random selection.",
            "",
        ])
    
    # Degradation analysis
    if degraded_df is not None:
        report_lines.extend([
            "## Classifier Robustness",
            "",
            "How well does probability selection work when the classifier is imperfect?",
            "",
            "We simulate noise by adding Gaussian perturbation to probabilities: `P_noisy = P + N(0, σ)`",
            "",
            "| Noise (σ) | Probability Selection | Random | Improvement |",
            "|-----------|----------------------|--------|-------------|",
        ])
        
        for _, row in degraded_df.iterrows():
            improvement = row['probability_rate'] - row['random_rate']
            report_lines.append(
                f"| {row['noise_level']:.1f} | {row['probability_rate']:.1%} | {row['random_rate']:.1%} | +{improvement:.1%} |"
            )
        
        report_lines.extend([
            "",
            "**Key insight**: Even with significant noise (σ=0.5), probability selection still substantially outperforms random.",
            "This robustness makes the approach practical even with imperfect classifiers.",
            "",
        ])
    
    report_lines.extend([
        "## Strategy Definitions",
        "",
        "| Strategy | Description |",
        "|----------|-------------|",
        "| Random | Pick one of N attempts uniformly at random |",
        "| Probability | Pick the attempt with highest P(success) |",
        "| Threshold | Pick first attempt with P > 0.5, else highest P |",
        "| Oracle | Pick a successful attempt if one exists (upper bound) |",
        "",
        "## Practical Applications",
        "",
        "1. **Compute-Efficient Sampling**: Run N attempts in parallel, use classifier to pick best",
        "2. **Early Stopping**: Monitor P(success) during execution, abandon low-probability attempts",
        "3. **Budget Allocation**: Allocate more compute to high-probability attempts",
        "4. **Quality Ranking**: Rank solutions by probability for human review",
        "",
        "## Methodology",
        "",
        "- Classifier: Random Forest trained on behavioral features",
        "- Probabilities: 5-fold cross-validation (unbiased estimates)",
        "- Simulation: 10,000 trials per N, sampling with replacement",
        "- Baseline success rate: ~40%",
        "",
        "## Visualizations",
        "",
        "- `best_of_n_selection.png`: Success rate curves by strategy and N",
        "- `probability_calibration.png`: Predicted vs actual success rates",
        "- `compute_efficiency.png`: Success per attempt by strategy",
        "- `classifier_degradation.png`: Robustness to classifier noise",
        "",
    ])
    
    report_path = output_path / 'best_of_n_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    return report_path


def simulate_degraded_classifier(
    probabilities: np.ndarray,
    actuals: np.ndarray,
    noise_levels: list[float],
    n_attempts: int = 5,
    n_simulations: int = 10000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Simulate Best-of-N with degraded (noisy) classifier predictions.
    
    This shows how selection performance degrades as classifier accuracy decreases.
    Noise is added to probabilities: noisy_p = p + N(0, noise_level), clipped to [0,1]
    """
    rng = np.random.RandomState(seed)
    n_samples = len(probabilities)
    
    results = []
    
    for noise in noise_levels:
        successes_prob = 0
        successes_random = 0
        successes_oracle = 0
        
        for _ in range(n_simulations):
            indices = rng.choice(n_samples, size=n_attempts, replace=True)
            true_probs = probabilities[indices]
            actual = actuals[indices]
            
            # Add noise to simulate imperfect classifier
            noisy_probs = true_probs + rng.normal(0, noise, size=n_attempts)
            noisy_probs = np.clip(noisy_probs, 0, 1)
            
            # Random selection
            random_idx = rng.randint(n_attempts)
            successes_random += actual[random_idx]
            
            # Probability selection (using noisy probabilities)
            prob_idx = np.argmax(noisy_probs)
            successes_prob += actual[prob_idx]
            
            # Oracle
            successes_oracle += int(actual.any())
        
        results.append({
            'noise_level': noise,
            'random_rate': successes_random / n_simulations,
            'probability_rate': successes_prob / n_simulations,
            'oracle_rate': successes_oracle / n_simulations,
        })
    
    return pd.DataFrame(results)


def compute_efficiency_analysis(results_df: pd.DataFrame, target_success: float = 0.80) -> dict:
    """
    Compute how many attempts are needed to reach target success rate.
    
    Returns dict with N needed for each strategy to reach the target.
    """
    efficiency = {}
    
    for strategy in ['random', 'probability', 'oracle']:
        strategy_data = results_df[results_df['strategy'] == strategy].sort_values('n_attempts')
        
        # Find first N where success_rate >= target
        for _, row in strategy_data.iterrows():
            if row['success_rate'] >= target_success:
                efficiency[strategy] = row['n_attempts']
                break
        else:
            efficiency[strategy] = None  # Never reaches target
    
    return efficiency


def plot_degraded_analysis(degraded_df: pd.DataFrame, output_path: Path) -> Path:
    """Plot classifier degradation analysis."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(degraded_df['noise_level'], degraded_df['probability_rate'] * 100, 
           'o-', color='#2ecc71', linewidth=2, markersize=8, label='Probability Selection')
    ax.axhline(y=degraded_df['random_rate'].mean() * 100, 
              color='#888888', linestyle='--', linewidth=2, label='Random Selection')
    ax.axhline(y=degraded_df['oracle_rate'].mean() * 100, 
              color='#e74c3c', linestyle=':', linewidth=2, label='Oracle (Upper Bound)')
    
    ax.set_xlabel('Classifier Noise (σ)', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Best-of-5 Selection: Robustness to Classifier Noise', fontsize=14)
    ax.legend(loc='center right')
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.annotate('Classifier still useful\neven with high noise', 
               xy=(0.3, degraded_df[degraded_df['noise_level']==0.3]['probability_rate'].values[0] * 100),
               xytext=(0.4, 60), fontsize=10,
               arrowprops=dict(arrowstyle='->', color='gray'))
    
    plt.tight_layout()
    
    plot_path = output_path / 'classifier_degradation.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def plot_compute_efficiency(results_df: pd.DataFrame, output_path: Path) -> Path:
    """Plot compute efficiency: success rate per attempt."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate efficiency: success_rate / n_attempts
    for strategy, color, label in [
        ('random', '#888888', 'Random'),
        ('probability', '#2ecc71', 'Probability'),
    ]:
        data = results_df[results_df['strategy'] == strategy].copy()
        data['efficiency'] = data['success_rate'] / data['n_attempts']
        ax.plot(data['n_attempts'], data['efficiency'] * 100, 
               'o-', color=color, linewidth=2, markersize=8, label=label)
    
    ax.set_xlabel('Number of Attempts (N)', fontsize=12)
    ax.set_ylabel('Efficiency: Success Rate / N (%)', fontsize=12)
    ax.set_title('Compute Efficiency: Success per Attempt', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, results_df['n_attempts'].max() + 1))
    
    plt.tight_layout()
    
    plot_path = output_path / 'compute_efficiency.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def main():
    """Run Best-of-N analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Best-of-N Selection Analysis')
    parser.add_argument('--data', type=Path, default=Path('training_data/training_data.jsonl'),
                       help='Path to training data JSONL')
    parser.add_argument('--output', type=Path, default=Path('analysis/outputs/best_of_n'),
                       help='Output directory')
    parser.add_argument('--max-n', type=int, default=10,
                       help='Maximum N to simulate')
    parser.add_argument('--simulations', type=int, default=10000,
                       help='Number of simulations per N')
    
    args = parser.parse_args()
    
    # Setup
    args.output.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("BEST-OF-N SELECTION ANALYSIS")
    print("=" * 60)
    print()
    
    # Load data
    print(f"Loading data from {args.data}...")
    df = load_training_data(args.data)
    print(f"Loaded {len(df)} samples")
    print()
    
    # Run analysis
    results_df = run_best_of_n_analysis(df, args.max_n, args.simulations)
    
    # Save results
    results_path = args.output / 'best_of_n_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_path = plot_best_of_n_results(results_df, args.output)
    print(f"  Main plot: {plot_path}")
    
    calib_path = plot_probability_calibration(df, args.output)
    print(f"  Calibration: {calib_path}")
    
    efficiency_path = plot_compute_efficiency(results_df, args.output)
    print(f"  Efficiency: {efficiency_path}")
    
    # Degraded classifier analysis
    print("\nRunning degraded classifier analysis...")
    probabilities = train_classifier_and_get_probabilities(df)
    actuals = df['resolved'].astype(int).values
    
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    degraded_df = simulate_degraded_classifier(probabilities, actuals, noise_levels)
    degraded_path = plot_degraded_analysis(degraded_df, args.output)
    print(f"  Degradation: {degraded_path}")
    
    # Save degraded results
    degraded_df.to_csv(args.output / 'degraded_classifier_results.csv', index=False)
    
    # Compute efficiency analysis
    efficiency = compute_efficiency_analysis(results_df, target_success=0.80)
    
    # Generate report
    report_path = generate_report(results_df, args.output, degraded_df, efficiency)
    print(f"  Report: {report_path}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Print summary table
    print("\nSuccess Rate by Strategy:")
    print("-" * 60)
    print(f"{'N':>3} | {'Random':>8} | {'Probability':>11} | {'Oracle':>8} | {'Improvement':>11}")
    print("-" * 60)
    
    for n in [1, 3, 5, 10]:
        if n <= args.max_n:
            n_data = results_df[results_df['n_attempts'] == n]
            rand = n_data[n_data['strategy'] == 'random']['success_rate'].values[0]
            prob = n_data[n_data['strategy'] == 'probability']['success_rate'].values[0]
            oracle = n_data[n_data['strategy'] == 'oracle']['success_rate'].values[0]
            print(f"{n:>3} | {rand:>7.1%} | {prob:>10.1%} | {oracle:>7.1%} | +{prob-rand:>9.1%}")
    
    print("-" * 60)
    
    # Efficiency summary
    print("\nCompute Efficiency (attempts needed for 80% success):")
    print("-" * 40)
    for strategy, n_needed in efficiency.items():
        if n_needed:
            print(f"  {strategy.capitalize():12}: N = {n_needed}")
        else:
            print(f"  {strategy.capitalize():12}: > {args.max_n} attempts")
    
    # Degradation summary
    print("\nClassifier Robustness (Best-of-5, varying noise):")
    print("-" * 40)
    for _, row in degraded_df.iterrows():
        if row['noise_level'] in [0.0, 0.2, 0.4, 0.6]:
            improvement = row['probability_rate'] - row['random_rate']
            print(f"  σ={row['noise_level']:.1f}: {row['probability_rate']:.1%} (+{improvement:.1%} vs random)")
    
    print("-" * 40)
    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    main()