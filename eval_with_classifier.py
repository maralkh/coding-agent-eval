#!/usr/bin/env python
"""
Evaluate benchmark results using a trained classifier.

Uses the trained classifier from the classifier/ folder to score benchmark
results and compare with heuristic scoring methods.

Usage:
    # Score results with trained classifier
    python eval_with_classifier.py --results results/benchmark/ --model models/classifier.joblib
    
    # Compare classifier with heuristic methods
    python eval_with_classifier.py --results results/benchmark/ --model models/classifier.joblib --compare
    
    # Generate comparison plots
    python eval_with_classifier.py --results results/benchmark/ --model models/classifier.joblib --plot
    
    # Output scores to JSON
    python eval_with_classifier.py --results results/benchmark/ --model models/classifier.joblib -o scores.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from eval.harness.results import (
    BenchmarkAnalyzer,
    load_benchmark_dir,
    BenchmarkTaskResult,
)
from eval.harness.scoring import (
    UnifiedScorer,
    compute_elo_ratings,
)
from classifier.train_classifier import (
    load_model,
    FEATURE_COLUMNS,
    LEAKY_FEATURES,
)


def extract_classifier_features(task_result: BenchmarkTaskResult, detailed_metrics: dict = None) -> dict:
    """
    Extract features for the classifier from a task result.
    
    Args:
        task_result: BenchmarkTaskResult object with core metrics
        detailed_metrics: Optional dict with detailed metrics from --detailed-metrics run
        
    Returns:
        Dict of features suitable for the classifier
    """
    # Start with basic metrics from task result
    features = {
        "reasoning_quality_score": task_result.reasoning_score,
        "exploration_efficiency": task_result.exploration_efficiency,
        "trajectory_efficiency": task_result.trajectory_efficiency,
        "trajectory_length": task_result.steps,
        "submitted": 1 if task_result.submitted else 0,
    }
    
    # Add similarity as final_similarity (if not using as leaky feature)
    features["final_similarity"] = task_result.similarity_score
    
    # If detailed metrics are available, use them
    if detailed_metrics:
        # Reasoning features
        reasoning = detailed_metrics.get("reasoning", {})
        features.update({
            "has_explicit_reasoning": int(reasoning.get("has_explicit_reasoning", False)),
            "mentions_issue_keywords": int(reasoning.get("mentions_issue_keywords", False)),
            "mentions_relevant_files": int(reasoning.get("mentions_relevant_files", False)),
            "hypothesizes_before_acting": int(reasoning.get("hypothesizes_before_acting", False)),
            "explains_changes": int(reasoning.get("explains_changes", False)),
            "verifies_after_change": int(reasoning.get("verifies_after_change", False)),
        })
        
        # Phase features
        phases = detailed_metrics.get("phases", {})
        features.update({
            "exploration_steps": phases.get("exploration_steps", 0),
            "implementation_steps": phases.get("implementation_steps", 0),
            "verification_steps": phases.get("verification_steps", 0),
            "exploration_pct": phases.get("exploration_pct", 0),
            "implementation_pct": phases.get("implementation_pct", 0),
            "verification_pct": phases.get("verification_pct", 0),
            "phase_transitions": phases.get("phase_transitions", 0),
            "followed_read_before_write": int(phases.get("followed_read_before_write", False)),
            "followed_test_after_change": int(phases.get("followed_test_after_change", False)),
        })
        
        # Exploration features
        exploration = detailed_metrics.get("exploration", {})
        features.update({
            "files_explored": exploration.get("files_explored", 0),
            "directories_explored": exploration.get("directories_explored", 0),
            "relevant_file_discovery_step": exploration.get("relevant_file_discovery_step", -1),
            "wasted_explorations": exploration.get("wasted_explorations", 0),
            "search_to_read_ratio": exploration.get("search_to_read_ratio", 0),
        })
        
        # Trajectory features
        trajectory = detailed_metrics.get("trajectory", {})
        features.update({
            "optimal_length": trajectory.get("optimal_length", 3),
            "unnecessary_steps": trajectory.get("unnecessary_steps", 0),
        })
        
        # Convergence features
        convergence = detailed_metrics.get("convergence", {})
        features.update({
            "max_progress": convergence.get("max_progress", 0),
            "converged": int(convergence.get("converged", False)),
            "monotonic_progress": int(convergence.get("monotonic_progress", False)),
            "had_regression": int(convergence.get("had_regression", False)),
            "progress_volatility": convergence.get("progress_volatility", 0),
        })
        
        # Error recovery features
        error_recovery = detailed_metrics.get("error_recovery", {})
        features.update({
            "total_errors": error_recovery.get("total_errors", 0),
            "recovered_errors": error_recovery.get("recovered_errors", 0),
            "recovery_rate": error_recovery.get("recovery_rate", 0),
            "max_repetition": error_recovery.get("max_repetition", 0),
            "stuck_episodes": error_recovery.get("stuck_episodes", 0),
            "max_stuck_duration": error_recovery.get("max_stuck_duration", 0),
        })
        
        # Tool usage features
        tool_usage = detailed_metrics.get("tool_usage", {})
        features.update({
            "total_tool_calls": tool_usage.get("total_calls", 0),
            "read_relevant_files": int(tool_usage.get("read_relevant_files", False)),
            "used_str_replace": int(tool_usage.get("used_str_replace", False)),
            "used_write_file": int(tool_usage.get("used_write_file", False)),
            "ran_tests": int(tool_usage.get("ran_tests", False)),
            "tool_errors_count": tool_usage.get("tool_errors", 0),
        })
        
        # Patch quality features
        patch_quality = detailed_metrics.get("patch_quality", {})
        features.update({
            "correct_files_touched": int(patch_quality.get("correct_files_touched", False)),
            "lines_added": patch_quality.get("lines_added", 0),
            "lines_removed": patch_quality.get("lines_removed", 0),
            "patch_too_large": int(patch_quality.get("patch_too_large", False)),
        })
        
        # Derived features
        steps = task_result.steps or 1
        files = exploration.get("files_explored", 1) or 1
        features["steps_per_file"] = steps / files
        
        explore_ratio = phases.get("exploration_pct", 0) or 0.01
        impl_ratio = phases.get("implementation_pct", 0) or 0.01
        features["edit_to_explore_ratio"] = impl_ratio / explore_ratio if explore_ratio > 0 else 0
    else:
        # Estimate features from available data
        features.update({
            "has_explicit_reasoning": 1 if task_result.reasoning_score > 0.3 else 0,
            "mentions_issue_keywords": 1 if task_result.reasoning_score > 0.2 else 0,
            "mentions_relevant_files": 1 if task_result.exploration_efficiency > 0.5 else 0,
            "hypothesizes_before_acting": 1 if task_result.reasoning_score > 0.5 else 0,
            "explains_changes": 1 if task_result.reasoning_score > 0.4 else 0,
            "verifies_after_change": 0,
            "exploration_steps": int(task_result.steps * 0.4),
            "implementation_steps": int(task_result.steps * 0.4),
            "verification_steps": int(task_result.steps * 0.2),
            "exploration_pct": 0.4,
            "implementation_pct": 0.4,
            "verification_pct": 0.2,
            "phase_transitions": min(task_result.steps - 1, 5),
            "followed_read_before_write": 1,
            "followed_test_after_change": 0,
            "files_explored": max(1, int(task_result.steps * 0.3)),
            "directories_explored": max(1, int(task_result.steps * 0.1)),
            "relevant_file_discovery_step": 2 if task_result.exploration_efficiency > 0.5 else 5,
            "wasted_explorations": int((1 - task_result.exploration_efficiency) * task_result.steps * 0.3),
            "search_to_read_ratio": 0.3,
            "optimal_length": 5,
            "unnecessary_steps": int((1 - task_result.trajectory_efficiency) * task_result.steps),
            "max_progress": task_result.similarity_score,
            "converged": 1 if task_result.similarity_score > 0.5 else 0,
            "monotonic_progress": 1 if task_result.resolved else 0,
            "had_regression": 0,
            "progress_volatility": 0.1,
            "total_errors": 0,
            "recovered_errors": 0,
            "recovery_rate": 1.0,
            "max_repetition": 1,
            "stuck_episodes": 0,
            "max_stuck_duration": 0,
            "total_tool_calls": task_result.steps,
            "read_relevant_files": 1 if task_result.exploration_efficiency > 0.5 else 0,
            "used_str_replace": 1,
            "used_write_file": 0,
            "ran_tests": 0,
            "tool_errors_count": 0,
            "correct_files_touched": 1 if task_result.similarity_score > 0.1 else 0,
            "lines_added": 5,
            "lines_removed": 2,
            "patch_too_large": 0,
            "steps_per_file": task_result.steps / max(1, int(task_result.steps * 0.3)),
            "edit_to_explore_ratio": 1.0,
        })
        
        # Semantic features (estimate from similarity)
        features.update({
            "fixes_same_file": 1 if task_result.similarity_score > 0.05 else 0,
            "fixes_same_function": 1 if task_result.similarity_score > 0.1 else 0,
            "fixes_same_class": 1 if task_result.similarity_score > 0.1 else 0,
            "fixes_same_code_region": 1 if task_result.similarity_score > 0.2 else 0,
            "location_score": min(1.0, task_result.similarity_score * 2),
            "change_type_match": 1 if task_result.similarity_score > 0.3 else 0,
            "modifies_same_variable": 1 if task_result.similarity_score > 0.4 else 0,
            "modifies_same_call": 1 if task_result.similarity_score > 0.4 else 0,
        })
    
    return features


def score_with_classifier(
    clf,
    scaler,
    feature_names: list,
    task_result: BenchmarkTaskResult,
    detailed_metrics: dict = None,
) -> dict:
    """
    Score a task result using the trained classifier.
    
    Returns:
        Dict with prediction, probability, and confidence
    """
    # Extract features
    features = extract_classifier_features(task_result, detailed_metrics)
    
    # Build feature vector in correct order
    feature_vector = []
    for feat in feature_names:
        value = features.get(feat, 0)
        if isinstance(value, bool):
            value = int(value)
        if value is None:
            value = 0
        if feat == "relevant_file_discovery_step" and value == -1:
            value = 100
        feature_vector.append(float(value))
    
    X = np.array([feature_vector])
    
    # Normalize
    if scaler is not None:
        X = scaler.transform(X)
    
    # Predict
    prediction = clf.predict(X)[0]
    
    result = {
        "prediction": bool(prediction),
        "predicted_label": "resolved" if prediction else "failed",
    }
    
    if hasattr(clf, "predict_proba"):
        probability = clf.predict_proba(X)[0]
        result["probability_resolved"] = float(probability[1])
        result["probability_failed"] = float(probability[0])
        result["confidence"] = float(max(probability))
        result["classifier_score"] = float(probability[1])  # Use as unified score
    else:
        result["classifier_score"] = float(prediction)
    
    return result


def print_header(title: str, char: str = "=", width: int = 70):
    """Print a formatted header."""
    print(char * width)
    print(f" {title}")
    print(char * width)


def print_table(headers: list, rows: list, col_widths: list = None):
    """Print a formatted table."""
    if col_widths is None:
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 
                      for i in range(len(headers))]
    
    header_str = "".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    print(header_str)
    print("-" * sum(col_widths))
    
    for row in rows:
        row_str = "".join(f"{str(v):<{w}}" for v, w in zip(row, col_widths))
        print(row_str)


def generate_classifier_plots(
    classifier_scores: list,
    heuristic_scores: dict,
    outcomes: list,
    output_dir: Path,
):
    """Generate plots comparing classifier with heuristic methods."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    outcomes = np.array(outcomes, dtype=float)
    classifier_scores = np.array(classifier_scores)
    
    colors = {
        'classifier': '#e74c3c',  # Red for classifier
        'resolved': '#2ecc71',
        'not_resolved': '#95a5a6',
    }
    
    # =========================================================================
    # 1. Classifier vs Heuristic Correlation
    # =========================================================================
    n_methods = len(heuristic_scores)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    
    for idx, (method, scores) in enumerate(list(heuristic_scores.items())[:6]):
        ax = axes[idx]
        scores = np.array(scores)
        
        # Color by outcome
        colors_scatter = [colors['resolved'] if o else colors['not_resolved'] for o in outcomes]
        
        ax.scatter(scores, classifier_scores, c=colors_scatter, alpha=0.6, s=50)
        
        # Correlation
        corr = np.corrcoef(scores, classifier_scores)[0, 1]
        
        # Trend line
        z = np.polyfit(scores, classifier_scores, 1)
        p = np.poly1d(z)
        x_line = np.linspace(scores.min(), scores.max(), 100)
        ax.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=1.5)
        
        ax.set_xlabel(f'{method.capitalize()} Score')
        ax.set_ylabel('Classifier Score')
        ax.set_title(f'{method.capitalize()} (r={corr:.2f})', fontweight='bold')
        ax.grid(alpha=0.3)
    
    for idx in range(len(heuristic_scores), 6):
        axes[idx].set_visible(False)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['resolved'], label='Resolved'),
        Patch(facecolor=colors['not_resolved'], label='Not Resolved'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.suptitle('Classifier vs Heuristic Scores', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'classifier_vs_heuristic.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'classifier_vs_heuristic.png'}")
    
    # =========================================================================
    # 2. All Methods ROC Comparison
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    
    all_scores = {'classifier': classifier_scores, **heuristic_scores}
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
    colors_list = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#34495e']
    
    for idx, (method, scores) in enumerate(all_scores.items()):
        scores = np.array(scores)
        
        thresholds = np.linspace(0, 1, 50)
        tpr_list = []
        fpr_list = []
        
        for thresh in thresholds:
            pred_positive = scores >= thresh
            
            # True positive rate
            if outcomes.sum() > 0:
                tpr = (pred_positive & (outcomes == 1)).sum() / outcomes.sum()
            else:
                tpr = 0
            
            # False positive rate
            if (outcomes == 0).sum() > 0:
                fpr = (pred_positive & (outcomes == 0)).sum() / (outcomes == 0).sum()
            else:
                fpr = 0
            
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        
        # Sort by FPR for proper ROC curve
        sorted_points = sorted(zip(fpr_list, tpr_list))
        fpr_sorted = [p[0] for p in sorted_points]
        tpr_sorted = [p[1] for p in sorted_points]
        
        # Calculate AUC (handle numpy version differences)
        try:
            auc = np.trapezoid(tpr_sorted, fpr_sorted)
        except AttributeError:
            # Fallback for older numpy versions
            auc = np.trapz(tpr_sorted, fpr_sorted)
        
        linewidth = 3 if method == 'classifier' else 2
        ax.plot(
            fpr_sorted, tpr_sorted,
            label=f'{method.capitalize()} (AUC={auc:.2f})',
            linewidth=linewidth,
            linestyle=line_styles[idx % len(line_styles)],
            color=colors_list[idx % len(colors_list)],
        )
    
    ax.plot([0, 1], [0, 1], 'k:', alpha=0.5, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves: Classifier vs Heuristic Methods', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'roc_comparison.png'}")
    
    # =========================================================================
    # 3. Classifier Score Distribution
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Box plot
    ax = axes[0]
    resolved_scores = classifier_scores[outcomes == 1]
    not_resolved_scores = classifier_scores[outcomes == 0]
    
    bp = ax.boxplot(
        [not_resolved_scores, resolved_scores],
        tick_labels=['Not Resolved', 'Resolved'],
        patch_artist=True,
    )
    bp['boxes'][0].set_facecolor(colors['not_resolved'])
    bp['boxes'][1].set_facecolor(colors['resolved'])
    
    ax.set_ylabel('Classifier Score (P(resolved))')
    ax.set_title('Classifier Score by Outcome', fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    
    # Histogram
    ax = axes[1]
    bins = np.linspace(0, 1, 20)
    ax.hist(not_resolved_scores, bins=bins, alpha=0.6, color=colors['not_resolved'], 
            label='Not Resolved', edgecolor='white')
    ax.hist(resolved_scores, bins=bins, alpha=0.6, color=colors['resolved'],
            label='Resolved', edgecolor='white')
    
    ax.set_xlabel('Classifier Score')
    ax.set_ylabel('Count')
    ax.set_title('Classifier Score Distribution', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'classifier_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'classifier_distribution.png'}")
    
    # =========================================================================
    # 4. Calibration Plot
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Bin predictions and compute actual rates
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    actual_rates = []
    predicted_means = []
    counts = []
    
    for i in range(n_bins):
        mask = (classifier_scores >= bin_edges[i]) & (classifier_scores < bin_edges[i+1])
        if mask.sum() > 0:
            actual_rates.append(outcomes[mask].mean())
            predicted_means.append(classifier_scores[mask].mean())
            counts.append(mask.sum())
        else:
            actual_rates.append(np.nan)
            predicted_means.append(np.nan)
            counts.append(0)
    
    # Plot calibration curve
    ax.plot([0, 1], [0, 1], 'k:', label='Perfect calibration')
    ax.scatter(predicted_means, actual_rates, s=[c*10 for c in counts], 
               c=colors['classifier'], alpha=0.7, label='Classifier')
    ax.plot(predicted_means, actual_rates, c=colors['classifier'], alpha=0.5)
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Actual Resolution Rate', fontsize=12)
    ax.set_title('Classifier Calibration Plot', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'classifier_calibration.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'classifier_calibration.png'}")


def eval_with_classifier(
    results_path: str,
    model_path: str,
    compare_heuristics: bool = False,
    plot_dir: str = None,
    output_path: str = None,
    verbose: bool = True,
):
    """
    Evaluate benchmark results using trained classifier.
    """
    # Load classifier
    if verbose:
        print(f"\nLoading classifier from: {model_path}")
    
    try:
        clf, scaler, feature_names = load_model(model_path)
    except Exception as e:
        print(f"Error loading classifier: {e}")
        sys.exit(1)
    
    if verbose:
        print(f"  Features: {len(feature_names)}")
        print(f"  Classifier type: {type(clf).__name__}")
    
    # Load results
    if verbose:
        print(f"\nLoading results from: {results_path}")
    
    try:
        analyzer = BenchmarkAnalyzer(results_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    benchmark = analyzer.benchmark
    
    if verbose:
        print_header("CLASSIFIER EVALUATION")
        print(f"Tasks: {len(benchmark.tasks)}")
        print(f"Models: {len(benchmark.models)}")
        print(f"Total runs: {sum(len(m.task_results) for m in benchmark.models)}")
        print()
    
    # Score each result with classifier
    results = {
        "timestamp": datetime.now().isoformat(),
        "classifier_model": str(model_path),
        "source": str(results_path),
        "model_scores": {},
        "task_scores": [],
    }
    
    all_classifier_scores = []
    all_outcomes = []
    all_heuristic_scores = defaultdict(list)
    
    # Get reference population for heuristic methods
    all_metrics = analyzer.get_metrics_dicts()
    
    for m in benchmark.models:
        model_scores = []
        
        for tr in m.task_results:
            # Score with classifier
            clf_result = score_with_classifier(clf, scaler, feature_names, tr)
            
            # Score with heuristic methods if comparing
            heuristic_results = {}
            if compare_heuristics:
                metrics = tr.to_metrics_dict()
                for method in ["weighted", "geometric", "hierarchical"]:
                    scorer = UnifiedScorer(method=method)
                    heuristic_results[method] = scorer.score(metrics)
                    all_heuristic_scores[method].append(heuristic_results[method])
                
                # Add comparison methods with reference population
                if len(all_metrics) >= 3:
                    for method in ["percentile", "topsis"]:
                        scorer = UnifiedScorer(method=method, reference_population=all_metrics)
                        heuristic_results[method] = scorer.score(metrics)
                        all_heuristic_scores[method].append(heuristic_results[method])
            
            score_entry = {
                "task_id": tr.task_id,
                "model": m.model,
                "resolved": tr.resolved,
                "classifier_score": clf_result["classifier_score"],
                "classifier_prediction": clf_result["predicted_label"],
                "classifier_confidence": clf_result.get("confidence", 0),
                "correct_prediction": clf_result["prediction"] == tr.resolved,
            }
            
            if compare_heuristics:
                score_entry["heuristic_scores"] = heuristic_results
            
            model_scores.append(score_entry)
            results["task_scores"].append(score_entry)
            
            all_classifier_scores.append(clf_result["classifier_score"])
            all_outcomes.append(tr.resolved)
        
        # Aggregate model scores
        classifier_scores = [s["classifier_score"] for s in model_scores]
        correct_predictions = sum(1 for s in model_scores if s["correct_prediction"])
        
        results["model_scores"][m.model] = {
            "mean_classifier_score": np.mean(classifier_scores),
            "std_classifier_score": np.std(classifier_scores),
            "prediction_accuracy": correct_predictions / len(model_scores) if model_scores else 0,
            "total_runs": len(model_scores),
            "resolved_count": sum(1 for s in model_scores if s["resolved"]),
        }
    
    # Print results
    if verbose:
        print_header("CLASSIFIER SCORES BY MODEL", "-")
        
        model_rows = []
        for model, data in sorted(results["model_scores"].items(), 
                                   key=lambda x: -x[1]["mean_classifier_score"]):
            resolved = f"{data['resolved_count']}/{data['total_runs']}"
            model_rows.append([
                model[:30],
                f"{data['mean_classifier_score']:.3f}",
                f"{data['std_classifier_score']:.3f}",
                f"{data['prediction_accuracy']:.0%}",
                resolved,
            ])
        
        print_table(
            ["Model", "Mean Score", "Std", "Pred Acc", "Resolved"],
            model_rows,
            [32, 12, 10, 10, 12],
        )
        print()
    
    # Compute overall accuracy
    all_classifier_scores = np.array(all_classifier_scores)
    all_outcomes = np.array(all_outcomes)
    predictions = all_classifier_scores >= 0.5
    accuracy = (predictions == all_outcomes).mean()
    
    # Correlation with outcome
    corr = np.corrcoef(all_classifier_scores, all_outcomes)[0, 1]
    
    if verbose:
        print_header("CLASSIFIER PERFORMANCE", "-")
        print(f"Overall prediction accuracy: {accuracy:.1%}")
        print(f"Correlation with outcome: {corr:.3f}")
        print(f"Mean score (resolved): {all_classifier_scores[all_outcomes == 1].mean():.3f}")
        print(f"Mean score (not resolved): {all_classifier_scores[all_outcomes == 0].mean():.3f}")
        print()
    
    results["overall"] = {
        "prediction_accuracy": float(accuracy),
        "correlation": float(corr),
        "mean_score_resolved": float(all_classifier_scores[all_outcomes == 1].mean()) if all_outcomes.sum() > 0 else 0,
        "mean_score_not_resolved": float(all_classifier_scores[all_outcomes == 0].mean()) if (all_outcomes == 0).sum() > 0 else 0,
    }
    
    # Compare with heuristics
    if compare_heuristics and verbose:
        print_header("COMPARISON WITH HEURISTIC METHODS", "-")
        
        comparison_rows = [["Classifier", f"{corr:.3f}", f"{accuracy:.0%}"]]
        
        for method, scores in all_heuristic_scores.items():
            scores = np.array(scores)
            method_corr = np.corrcoef(scores, all_outcomes)[0, 1]
            method_pred = scores >= 0.5
            method_acc = (method_pred == all_outcomes).mean()
            comparison_rows.append([method.capitalize(), f"{method_corr:.3f}", f"{method_acc:.0%}"])
        
        print_table(
            ["Method", "Correlation", "Accuracy"],
            comparison_rows,
            [20, 15, 12],
        )
        print()
        
        results["heuristic_comparison"] = {
            method: {
                "correlation": float(np.corrcoef(np.array(scores), all_outcomes)[0, 1]),
                "accuracy": float(((np.array(scores) >= 0.5) == all_outcomes).mean()),
            }
            for method, scores in all_heuristic_scores.items()
        }
    
    # Generate plots
    if plot_dir:
        if verbose:
            print_header("GENERATING PLOTS", "-")
        
        generate_classifier_plots(
            classifier_scores=all_classifier_scores,
            heuristic_scores=dict(all_heuristic_scores) if compare_heuristics else {},
            outcomes=all_outcomes,
            output_dir=Path(plot_dir),
        )
    
    # Save results
    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        if verbose:
            print(f"\nResults saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate benchmark results using trained classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python eval_with_classifier.py --results results/benchmark/ --model models/classifier.joblib
    python eval_with_classifier.py --results results/benchmark/ --model models/classifier.joblib --compare
    python eval_with_classifier.py --results results/benchmark/ --model models/classifier.joblib --plot
    python eval_with_classifier.py --results results/benchmark/ --model models/classifier.joblib -o scores.json
        """,
    )
    
    parser.add_argument(
        "--results", "-r",
        required=True,
        help="Path to results directory or benchmark_result.json",
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to trained classifier model (.joblib)",
    )
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="Compare classifier with heuristic scoring methods",
    )
    parser.add_argument(
        "--plot", "-p",
        nargs="?",
        const="classifier_plots",
        help="Generate comparison plots (optionally specify output directory)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Save scores to JSON file",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output",
    )
    
    args = parser.parse_args()
    
    eval_with_classifier(
        results_path=args.results,
        model_path=args.model,
        compare_heuristics=args.compare,
        plot_dir=args.plot,
        output_path=args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
