#!/usr/bin/env python
"""
Train a classifier to predict agent success from behavioral metrics.

This classifier learns to predict whether an agent will successfully resolve
a task based on metrics collected from benchmark runs.

Usage:
    # Train from benchmark result JSON files
    python train_classifier.py --data results/benchmark/runs/
    
    # Train with specific model type
    python train_classifier.py --data results/ --classifier random_forest
    
    # Evaluate only (no training)
    python train_classifier.py --data results/ --evaluate-only
    
    # Show feature importance
    python train_classifier.py --data results/ --show-correlations
    
    # Use trained model for prediction
    python train_classifier.py --predict --model models/classifier.joblib --metrics result.json
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import warnings

# Suppress sklearn warnings
warnings.filterwarnings("ignore")

import numpy as np


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# Features extracted from metrics.core
CORE_FEATURES = [
    "similarity_score",
    "reasoning_score",
    "exploration_efficiency",
    "trajectory_efficiency",
]

# Features extracted from metrics.detailed.tool_usage
TOOL_USAGE_FEATURES = [
    "total_calls",
    "read_relevant_files",
    "used_str_replace",
    "used_write_file",
    "ran_tests",
    "submitted",
]

# Features extracted from metrics.detailed.patch_quality
PATCH_QUALITY_FEATURES = [
    "correct_files_touched",
    "lines_added",
    "lines_removed",
    "patch_too_large",
]

# Features extracted from metrics.detailed.reasoning
REASONING_FEATURES = [
    "has_explicit_reasoning",
    "mentions_issue_keywords",
    "mentions_relevant_files",
    "hypothesizes_before_acting",
    "explains_changes",
    "verifies_after_change",
]

# Features extracted from metrics.detailed.phases
PHASE_FEATURES = [
    "exploration_steps",
    "implementation_steps",
    "verification_steps",
    "exploration_pct",
    "phase_transitions",
    "followed_read_before_write",
    "followed_test_after_change",
]

# Features extracted from metrics.detailed.exploration
EXPLORATION_FEATURES = [
    "files_explored",
    "directories_explored",
    "relevant_file_discovery_step",
    "wasted_explorations",
]

# Features extracted from metrics.detailed.trajectory
TRAJECTORY_FEATURES = [
    "length",
    "optimal_length",
    "unnecessary_steps",
]

# Features extracted from metrics.detailed.convergence
CONVERGENCE_FEATURES = [
    "final_similarity",
    "max_progress",
    "converged",
    "monotonic_progress",
    "had_regression",
    "progress_volatility",
]

# Features extracted from metrics.detailed.error_recovery
ERROR_RECOVERY_FEATURES = [
    "total_errors",
    "recovered_errors",
    "recovery_rate",
    "max_repetition",
    "stuck_episodes",
    "max_stuck_duration",
]

# Features extracted from metrics.detailed.failure_analysis
FAILURE_ANALYSIS_FEATURES = [
    "hit_max_steps",
    "agent_submitted",
    "no_changes_made",
    "wrong_files_modified",
    "tool_errors_occurred",
    "model_got_stuck",
]

# Features that might leak target information (use with caution)
LEAKY_FEATURES = [
    "similarity_score",
    "final_similarity",
    "max_progress",
    "converged",
    "correct_files_touched",
]


@dataclass
class ClassifierConfig:
    """Configuration for the classifier."""
    classifier_type: str = "random_forest"  # logistic, random_forest, gradient_boosting, svm
    use_leaky_features: bool = False
    normalize: bool = True
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42


def load_benchmark_results(data_path: str) -> list[dict]:
    """
    Load benchmark result JSON files from a directory.
    
    Supports:
    - Directory containing individual JSON files (one per run)
    - Single JSON file with nested results
    - JSONL file with one result per line
    """
    path = Path(data_path)
    results = []
    
    if path.is_file():
        if path.suffix == ".jsonl":
            # JSONL format
            with open(path) as f:
                for line in f:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        else:
            # Single JSON file
            with open(path) as f:
                data = json.load(f)
                # Check if it's a single result or contains multiple
                if "metrics" in data and "evaluation" in data:
                    results.append(data)
                elif "results" in data:
                    # Nested results format
                    results.extend(data["results"])
    else:
        # Directory of JSON files
        for json_file in sorted(path.glob("**/*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    if "metrics" in data and "evaluation" in data:
                        results.append(data)
            except (json.JSONDecodeError, KeyError):
                continue
    
    return results


def extract_feature_value(data: dict, path: str, default=0):
    """Extract a value from nested dict using dot notation path."""
    keys = path.split(".")
    value = data
    try:
        for key in keys:
            value = value[key]
        if value is None:
            return default
        if isinstance(value, bool):
            return int(value)
        return float(value)
    except (KeyError, TypeError):
        return default


def extract_features(results: list[dict], config: ClassifierConfig) -> tuple:
    """
    Extract feature matrix X and labels y from benchmark results.
    
    Returns:
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,)
        feature_names: list of feature names
        feature_map: dict mapping feature names to JSON paths
    """
    # Build feature mapping: feature_name -> path in JSON
    feature_map = {}
    
    # Core features
    for feat in CORE_FEATURES:
        feature_map[feat] = f"metrics.core.{feat}"
    
    # Tool usage features
    for feat in TOOL_USAGE_FEATURES:
        feature_map[f"tool_{feat}"] = f"metrics.detailed.tool_usage.{feat}"
    
    # Patch quality features
    for feat in PATCH_QUALITY_FEATURES:
        feature_map[f"patch_{feat}"] = f"metrics.detailed.patch_quality.{feat}"
    
    # Reasoning features
    for feat in REASONING_FEATURES:
        feature_map[f"reasoning_{feat}"] = f"metrics.detailed.reasoning.{feat}"
    
    # Phase features
    for feat in PHASE_FEATURES:
        feature_map[f"phase_{feat}"] = f"metrics.detailed.phases.{feat}"
    
    # Exploration features
    for feat in EXPLORATION_FEATURES:
        feature_map[f"explore_{feat}"] = f"metrics.detailed.exploration.{feat}"
    
    # Trajectory features
    for feat in TRAJECTORY_FEATURES:
        feature_map[f"traj_{feat}"] = f"metrics.detailed.trajectory.{feat}"
    
    # Convergence features
    for feat in CONVERGENCE_FEATURES:
        feature_map[f"conv_{feat}"] = f"metrics.detailed.convergence.{feat}"
    
    # Error recovery features
    for feat in ERROR_RECOVERY_FEATURES:
        feature_map[f"error_{feat}"] = f"metrics.detailed.error_recovery.{feat}"
    
    # Failure analysis features
    for feat in FAILURE_ANALYSIS_FEATURES:
        feature_map[f"fail_{feat}"] = f"metrics.detailed.failure_analysis.{feat}"
    
    # Add agent-level features
    feature_map["agent_steps"] = "agent.steps"
    feature_map["agent_success"] = "agent.success"
    
    # Filter out leaky features if requested
    if not config.use_leaky_features:
        leaky_set = set(LEAKY_FEATURES)
        feature_map = {k: v for k, v in feature_map.items() 
                      if not any(leak in v for leak in leaky_set)}
    
    feature_names = list(feature_map.keys())
    
    X = []
    y = []
    
    for result in results:
        # Extract label
        resolved = result.get("evaluation", {}).get("resolved", False)
        
        # Extract features
        features = []
        for feat_name in feature_names:
            path = feature_map[feat_name]
            value = extract_feature_value(result, path)
            
            # Handle special cases
            if "discovery_step" in feat_name and value == -1:
                value = 100  # Large value for "never found"
            
            features.append(value)
        
        X.append(features)
        y.append(int(resolved))
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y, feature_names, feature_map


def create_classifier(config: ClassifierConfig):
    """Create a classifier based on config."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    
    if config.classifier_type == "logistic":
        return LogisticRegression(
            random_state=config.random_state,
            max_iter=1000,
            class_weight="balanced",
        )
    elif config.classifier_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            random_state=config.random_state,
            class_weight="balanced",
            max_depth=10,
        )
    elif config.classifier_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=100,
            random_state=config.random_state,
            max_depth=5,
        )
    elif config.classifier_type == "svm":
        return SVC(
            random_state=config.random_state,
            class_weight="balanced",
            probability=True,
        )
    else:
        raise ValueError(f"Unknown classifier type: {config.classifier_type}")


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    config: ClassifierConfig,
) -> dict:
    """Train classifier and evaluate with cross-validation."""
    from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score
    )
    
    results = {
        "config": {
            "classifier_type": config.classifier_type,
            "n_samples": len(y),
            "n_features": len(feature_names),
            "positive_rate": float(y.mean()),
        }
    }
    
    # Normalize if requested
    scaler = None
    if config.normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Create classifier
    clf = create_classifier(config)
    
    # Cross-validation
    print("Running cross-validation...")
    cv_scores = cross_val_score(clf, X, y, cv=config.cv_folds, scoring="accuracy")
    cv_f1 = cross_val_score(clf, X, y, cv=config.cv_folds, scoring="f1")
    cv_precision = cross_val_score(clf, X, y, cv=config.cv_folds, scoring="precision")
    cv_recall = cross_val_score(clf, X, y, cv=config.cv_folds, scoring="recall")
    
    results["cross_validation"] = {
        "accuracy": {"mean": float(cv_scores.mean()), "std": float(cv_scores.std())},
        "f1": {"mean": float(cv_f1.mean()), "std": float(cv_f1.std())},
        "precision": {"mean": float(cv_precision.mean()), "std": float(cv_precision.std())},
        "recall": {"mean": float(cv_recall.mean()), "std": float(cv_recall.std())},
    }
    
    # Get cross-validated predictions for confusion matrix
    y_pred_cv = cross_val_predict(clf, X, y, cv=config.cv_folds)
    results["confusion_matrix"] = confusion_matrix(y, y_pred_cv).tolist()
    
    # Train on all data for feature importance
    clf.fit(X, y)
    
    # Feature importance
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        importances = np.zeros(len(feature_names))
    
    # Sort by importance
    importance_pairs = sorted(
        zip(feature_names, importances),
        key=lambda x: -x[1]
    )
    results["feature_importance"] = [
        {"feature": name, "importance": float(imp)}
        for name, imp in importance_pairs
    ]
    
    # Train/test split for final metrics
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )
    
    clf_final = create_classifier(config)
    clf_final.fit(X_train, y_train)
    y_pred = clf_final.predict(X_test)
    
    results["test_metrics"] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    
    # ROC AUC if we have probabilities
    y_prob = None
    if hasattr(clf_final, "predict_proba"):
        y_prob = clf_final.predict_proba(X_test)[:, 1]
        results["test_metrics"]["roc_auc"] = float(roc_auc_score(y_test, y_prob))
    
    # Store test data for visualization
    test_data = {
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
    
    return results, clf, scaler, test_data


def print_results(results: dict) -> None:
    """Pretty print evaluation results."""
    print()
    print("=" * 70)
    print("CLASSIFIER EVALUATION RESULTS")
    print("=" * 70)
    print()
    
    # Config
    cfg = results["config"]
    print(f"Classifier: {cfg['classifier_type']}")
    print(f"Samples: {cfg['n_samples']}")
    print(f"Features: {cfg['n_features']}")
    print(f"Positive rate: {cfg['positive_rate']:.1%}")
    print()
    
    # Cross-validation
    print("Cross-Validation Results (5-fold):")
    print("-" * 40)
    cv = results["cross_validation"]
    print(f"  Accuracy:  {cv['accuracy']['mean']:.3f} ± {cv['accuracy']['std']:.3f}")
    print(f"  Precision: {cv['precision']['mean']:.3f} ± {cv['precision']['std']:.3f}")
    print(f"  Recall:    {cv['recall']['mean']:.3f} ± {cv['recall']['std']:.3f}")
    print(f"  F1 Score:  {cv['f1']['mean']:.3f} ± {cv['f1']['std']:.3f}")
    print()
    
    # Confusion matrix
    print("Confusion Matrix:")
    print("-" * 40)
    cm = results["confusion_matrix"]
    print(f"  True Neg:  {cm[0][0]:<5}  False Pos: {cm[0][1]}")
    print(f"  False Neg: {cm[1][0]:<5}  True Pos:  {cm[1][1]}")
    print()
    
    # Test metrics
    print("Test Set Metrics (20% holdout):")
    print("-" * 40)
    tm = results["test_metrics"]
    print(f"  Accuracy:  {tm['accuracy']:.3f}")
    print(f"  Precision: {tm['precision']:.3f}")
    print(f"  Recall:    {tm['recall']:.3f}")
    print(f"  F1 Score:  {tm['f1']:.3f}")
    if "roc_auc" in tm:
        print(f"  ROC AUC:   {tm['roc_auc']:.3f}")
    print()
    
    # Feature importance (top 15)
    print("Top 15 Most Important Features:")
    print("-" * 40)
    for i, fi in enumerate(results["feature_importance"][:15], 1):
        bar = "█" * int(fi["importance"] * 50)
        print(f"  {i:2}. {fi['feature']:<35} {fi['importance']:.3f} {bar}")
    print()


def generate_visualizations(
    results: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Path,
) -> dict:
    """Generate visualization plots for classifier results."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Warning: matplotlib/seaborn not available, skipping visualizations")
        return {}
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_files = {}
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12']
    
    # 1. Feature Importance Bar Chart
    fig, ax = plt.subplots(figsize=(10, 8))
    importance_data = results["feature_importance"][:20]
    features = [d["feature"] for d in importance_data][::-1]
    importances = [d["importance"] for d in importance_data][::-1]
    
    bars = ax.barh(features, importances, color=colors[2], edgecolor='white', linewidth=0.5)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title("Top 20 Feature Importances", fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels
    for bar, val in zip(bars, importances):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plot_file = output_dir / "feature_importance.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    plot_files["feature_importance"] = str(plot_file)
    
    # 2. Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = np.array(results["confusion_matrix"])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Predicted\nFailed', 'Predicted\nResolved'],
                yticklabels=['Actual\nFailed', 'Actual\nResolved'],
                annot_kws={'size': 16})
    ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plot_file = output_dir / "confusion_matrix.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    plot_files["confusion_matrix"] = str(plot_file)
    
    # 3. ROC Curve (if probabilities available)
    if y_prob is not None:
        from sklearn.metrics import roc_curve, auc
        
        fig, ax = plt.subplots(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=colors[2], lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
        ax.fill_between(fpr, tpr, alpha=0.3, color=colors[2])
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plot_file = output_dir / "roc_curve.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        plot_files["roc_curve"] = str(plot_file)
    
    # 4. Precision-Recall Curve
    if y_prob is not None:
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        fig, ax = plt.subplots(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        avg_precision = average_precision_score(y_test, y_prob)
        
        ax.plot(recall, precision, color=colors[3], lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        ax.fill_between(recall, precision, alpha=0.3, color=colors[3])
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plot_file = output_dir / "precision_recall_curve.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        plot_files["precision_recall"] = str(plot_file)
    
    # 5. Cross-validation scores boxplot
    fig, ax = plt.subplots(figsize=(8, 6))
    cv_data = results["cross_validation"]
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    means = [cv_data[m]['mean'] for m in metrics]
    stds = [cv_data[m]['std'] for m in metrics]
    
    x_pos = np.arange(len(metrics))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors[:4], 
                  edgecolor='white', linewidth=0.5)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Cross-Validation Metrics (5-fold)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plot_file = output_dir / "cv_metrics.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    plot_files["cv_metrics"] = str(plot_file)
    
    print(f"Visualizations saved to: {output_dir}/")
    return plot_files


def generate_html_report(
    results: dict,
    plot_files: dict,
    output_file: Path,
) -> None:
    """Generate an HTML report with tables and embedded visualizations."""
    
    cfg = results["config"]
    cv = results["cross_validation"]
    tm = results["test_metrics"]
    cm = results["confusion_matrix"]
    fi = results["feature_importance"]
    
    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Classifier Training Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6; 
            color: #333; 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{ 
            color: #2c3e50; 
            border-bottom: 3px solid #3498db; 
            padding-bottom: 10px; 
            margin-bottom: 30px;
        }}
        h2 {{ 
            color: #34495e; 
            margin: 30px 0 15px 0;
            padding-bottom: 5px;
            border-bottom: 1px solid #ddd;
        }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .metric-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-box.green {{ background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }}
        .metric-box.blue {{ background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }}
        .metric-box.orange {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }}
        .metric-box.purple {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        .metric-value {{ font-size: 2em; font-weight: bold; }}
        .metric-label {{ font-size: 0.9em; opacity: 0.9; }}
        table {{ 
            width: 100%; 
            border-collapse: collapse; 
            margin: 15px 0;
            background: white;
        }}
        th, td {{ 
            padding: 12px 15px; 
            text-align: left; 
            border-bottom: 1px solid #eee;
        }}
        th {{ 
            background: #f8f9fa; 
            font-weight: 600;
            color: #2c3e50;
        }}
        tr:hover {{ background: #f8f9fa; }}
        .importance-bar {{
            background: #3498db;
            height: 20px;
            border-radius: 3px;
            display: inline-block;
            vertical-align: middle;
            margin-left: 10px;
        }}
        .plot-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .plot-card {{
            background: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .plot-card img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .cm-table {{ width: auto; margin: 0 auto; }}
        .cm-table td {{ 
            width: 100px; 
            height: 60px; 
            text-align: center; 
            font-weight: bold;
            font-size: 1.2em;
        }}
        .cm-tn {{ background: #d4edda; color: #155724; }}
        .cm-fp {{ background: #f8d7da; color: #721c24; }}
        .cm-fn {{ background: #fff3cd; color: #856404; }}
        .cm-tp {{ background: #cce5ff; color: #004085; }}
        .summary {{ 
            background: #e8f4fd; 
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <h1>🤖 Classifier Training Report</h1>
    
    <div class="card">
        <h2>📊 Configuration</h2>
        <div class="metrics-grid">
            <div class="metric-box purple">
                <div class="metric-value">{cfg['classifier_type'].replace('_', ' ').title()}</div>
                <div class="metric-label">Classifier Type</div>
            </div>
            <div class="metric-box blue">
                <div class="metric-value">{cfg['n_samples']}</div>
                <div class="metric-label">Training Samples</div>
            </div>
            <div class="metric-box green">
                <div class="metric-value">{cfg['n_features']}</div>
                <div class="metric-label">Features</div>
            </div>
            <div class="metric-box orange">
                <div class="metric-value">{cfg['positive_rate']:.1%}</div>
                <div class="metric-label">Positive Rate</div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <h2>📈 Test Set Performance</h2>
        <div class="metrics-grid">
            <div class="metric-box green">
                <div class="metric-value">{tm['accuracy']:.1%}</div>
                <div class="metric-label">Accuracy</div>
            </div>
            <div class="metric-box blue">
                <div class="metric-value">{tm['precision']:.1%}</div>
                <div class="metric-label">Precision</div>
            </div>
            <div class="metric-box purple">
                <div class="metric-value">{tm['recall']:.1%}</div>
                <div class="metric-label">Recall</div>
            </div>
            <div class="metric-box orange">
                <div class="metric-value">{tm['f1']:.1%}</div>
                <div class="metric-label">F1 Score</div>
            </div>
        </div>
        {"<p><strong>ROC AUC:</strong> " + f"{tm['roc_auc']:.3f}</p>" if 'roc_auc' in tm else ""}
    </div>
    
    <div class="card">
        <h2>🔄 Cross-Validation Results (5-fold)</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Mean</th>
                    <th>Std Dev</th>
                    <th>Range</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Accuracy</td>
                    <td><strong>{cv['accuracy']['mean']:.3f}</strong></td>
                    <td>± {cv['accuracy']['std']:.3f}</td>
                    <td>{cv['accuracy']['mean'] - cv['accuracy']['std']:.3f} - {cv['accuracy']['mean'] + cv['accuracy']['std']:.3f}</td>
                </tr>
                <tr>
                    <td>Precision</td>
                    <td><strong>{cv['precision']['mean']:.3f}</strong></td>
                    <td>± {cv['precision']['std']:.3f}</td>
                    <td>{cv['precision']['mean'] - cv['precision']['std']:.3f} - {cv['precision']['mean'] + cv['precision']['std']:.3f}</td>
                </tr>
                <tr>
                    <td>Recall</td>
                    <td><strong>{cv['recall']['mean']:.3f}</strong></td>
                    <td>± {cv['recall']['std']:.3f}</td>
                    <td>{cv['recall']['mean'] - cv['recall']['std']:.3f} - {cv['recall']['mean'] + cv['recall']['std']:.3f}</td>
                </tr>
                <tr>
                    <td>F1 Score</td>
                    <td><strong>{cv['f1']['mean']:.3f}</strong></td>
                    <td>± {cv['f1']['std']:.3f}</td>
                    <td>{cv['f1']['mean'] - cv['f1']['std']:.3f} - {cv['f1']['mean'] + cv['f1']['std']:.3f}</td>
                </tr>
            </tbody>
        </table>
    </div>
    
    <div class="card">
        <h2>🎯 Confusion Matrix</h2>
        <table class="cm-table">
            <tr>
                <td></td>
                <td><strong>Predicted Failed</strong></td>
                <td><strong>Predicted Resolved</strong></td>
            </tr>
            <tr>
                <td><strong>Actual Failed</strong></td>
                <td class="cm-tn">TN: {cm[0][0]}</td>
                <td class="cm-fp">FP: {cm[0][1]}</td>
            </tr>
            <tr>
                <td><strong>Actual Resolved</strong></td>
                <td class="cm-fn">FN: {cm[1][0]}</td>
                <td class="cm-tp">TP: {cm[1][1]}</td>
            </tr>
        </table>
        <div class="summary">
            <strong>Interpretation:</strong>
            <ul style="margin-top: 10px; padding-left: 20px;">
                <li><strong>True Negatives ({cm[0][0]}):</strong> Correctly predicted failures</li>
                <li><strong>True Positives ({cm[1][1]}):</strong> Correctly predicted successes</li>
                <li><strong>False Positives ({cm[0][1]}):</strong> Predicted success but actually failed</li>
                <li><strong>False Negatives ({cm[1][0]}):</strong> Predicted failure but actually succeeded</li>
            </ul>
        </div>
    </div>
    
    <div class="card">
        <h2>⭐ Feature Importance (Top 20)</h2>
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Feature</th>
                    <th>Importance</th>
                    <th>Visualization</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # Add feature importance rows
    max_importance = fi[0]["importance"] if fi else 1
    for i, f in enumerate(fi[:20], 1):
        bar_width = int((f["importance"] / max_importance) * 200)
        html += f"""                <tr>
                    <td>{i}</td>
                    <td><code>{f['feature']}</code></td>
                    <td><strong>{f['importance']:.4f}</strong></td>
                    <td><div class="importance-bar" style="width: {bar_width}px;"></div></td>
                </tr>
"""
    
    html += """            </tbody>
        </table>
    </div>
"""
    
    # Add plots if available
    if plot_files:
        html += """
    <div class="card">
        <h2>📉 Visualizations</h2>
        <div class="plot-container">
"""
        for name, path in plot_files.items():
            # Use relative path or embed as base64
            import base64
            try:
                with open(path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode()
                html += f"""            <div class="plot-card">
                <img src="data:image/png;base64,{img_data}" alt="{name}">
            </div>
"""
            except:
                pass
        
        html += """        </div>
    </div>
"""
    
    # Footer
    from datetime import datetime
    html += f"""
    <div class="footer">
        <p>Generated by Coding Agent Eval Framework</p>
        <p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>
"""
    
    # Write file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write(html)
    
    print(f"HTML report saved to: {output_file}")


def save_model(clf, scaler, feature_names: list[str], feature_map: dict, results: dict, output_dir: str) -> Path:
    """Save trained model and metadata."""
    import joblib
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_file = output_path / "classifier.joblib"
    joblib.dump({
        "classifier": clf,
        "scaler": scaler,
        "feature_names": feature_names,
        "feature_map": feature_map,
    }, model_file)
    
    # Save results
    results_file = output_path / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Model saved to: {model_file}")
    print(f"Results saved to: {results_file}")
    
    return model_file


def predict(model_path: str, result_data: dict) -> dict:
    """Use trained model to predict success from a benchmark result JSON."""
    import joblib
    
    data = joblib.load(model_path)
    clf = data["classifier"]
    scaler = data["scaler"]
    feature_names = data["feature_names"]
    feature_map = data.get("feature_map", {})
    
    # Extract features using the same mapping used during training
    features = []
    for feat_name in feature_names:
        if feat_name in feature_map:
            path = feature_map[feat_name]
            value = extract_feature_value(result_data, path)
        else:
            value = 0
        
        if "discovery_step" in feat_name and value == -1:
            value = 100
        features.append(value)
    
    X = np.array([features])
    
    # Normalize
    if scaler is not None:
        X = scaler.transform(X)
    
    # Predict
    prediction = clf.predict(X)[0]
    probability = clf.predict_proba(X)[0] if hasattr(clf, "predict_proba") else None
    
    result = {
        "prediction": bool(prediction),
        "predicted_label": "resolved" if prediction else "failed",
    }
    
    if probability is not None:
        result["probability_resolved"] = float(probability[1])
        result["probability_failed"] = float(probability[0])
        result["confidence"] = float(max(probability))
    
    return result


def analyze_feature_correlations(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
    """Analyze correlations between features and target."""
    from scipy.stats import pointbiserialr
    
    print()
    print("Feature Correlations with Target:")
    print("-" * 50)
    
    correlations = []
    for i, feat in enumerate(feature_names):
        corr, pval = pointbiserialr(y, X[:, i])
        correlations.append((feat, corr, pval))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: -abs(x[1]))
    
    print(f"{'Feature':<35} {'Correlation':<12} {'P-value':<10}")
    print("-" * 60)
    for feat, corr, pval in correlations[:20]:
        sig = "*" if pval < 0.05 else ""
        print(f"{feat:<35} {corr:>+.3f}{sig:<4}     {pval:.4f}")
    print()
    print("* = statistically significant (p < 0.05)")


def main():
    parser = argparse.ArgumentParser(
        description="Train classifier to predict agent success",
    )
    
    parser.add_argument(
        "--data",
        default="training_data",
        help="Path to training data (file or directory)",
    )
    parser.add_argument(
        "--classifier",
        choices=["logistic", "random_forest", "gradient_boosting", "svm"],
        default="random_forest",
        help="Classifier type (default: random_forest)",
    )
    parser.add_argument(
        "-o", "--output",
        default="models",
        help="Output directory for model (default: models)",
    )
    parser.add_argument(
        "--use-leaky-features",
        action="store_true",
        help="Include features that may leak target info",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Don't normalize features",
    )
    parser.add_argument(
        "--show-correlations",
        action="store_true",
        help="Show feature correlations with target",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only evaluate, don't save model",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate HTML report with visualizations",
    )
    
    # Prediction mode
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Use trained model to make prediction",
    )
    parser.add_argument(
        "--model",
        help="Path to trained model (for --predict)",
    )
    parser.add_argument(
        "--metrics",
        help="Path to metrics JSON file (for --predict)",
    )
    
    args = parser.parse_args()
    
    # Prediction mode
    if args.predict:
        if not args.model or not args.metrics:
            parser.error("--predict requires --model and --metrics")
        
        with open(args.metrics) as f:
            metrics = json.load(f)
        
        result = predict(args.model, metrics)
        print(json.dumps(result, indent=2))
        return
    
    # Training mode
    print("Loading benchmark results...")
    results = load_benchmark_results(args.data)
    print(f"Loaded {len(results)} examples")
    
    if len(results) < 10:
        print("ERROR: Need at least 10 examples for training")
        return
    
    config = ClassifierConfig(
        classifier_type=args.classifier,
        use_leaky_features=args.use_leaky_features,
        normalize=not args.no_normalize,
    )
    
    # Extract features
    print("Extracting features...")
    X, y, feature_names, feature_map = extract_features(results, config)
    print(f"Feature matrix: {X.shape}")
    print(f"Positive examples: {y.sum()} ({y.mean():.1%})") 
    
    # Show correlations if requested
    if args.show_correlations:
        analyze_feature_correlations(X, y, feature_names)
    
    # Train and evaluate
    print()
    eval_results, clf, scaler, test_data = train_and_evaluate(X, y, feature_names, config)
    
    # Print results
    print_results(eval_results)
    
    # Save model
    output_path = Path(args.output)
    if not args.evaluate_only:
        save_model(clf, scaler, feature_names, feature_map, eval_results, args.output)
    
    # Generate report with visualizations
    if args.report:
        print()
        print("Generating visualizations...")
        plot_files = generate_visualizations(
            eval_results,
            test_data["X_test"],
            test_data["y_test"],
            test_data["y_pred"],
            test_data["y_prob"],
            output_path / "plots",
        )
        
        print("Generating HTML report...")
        generate_html_report(
            eval_results,
            plot_files,
            output_path / "classifier_report.html",
        )


if __name__ == "__main__":
    main()