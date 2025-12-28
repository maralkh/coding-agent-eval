#!/usr/bin/env python
"""
Train a classifier to predict agent success from behavioral metrics.

This classifier learns to predict whether an agent will successfully resolve
a task based on metrics like reasoning quality, exploration efficiency,
tool usage patterns, etc.

Usage:
    # Train classifier
    python train_classifier.py --data training_data/training_data.jsonl
    
    # Train with specific model type
    python train_classifier.py --data training_data/ --classifier random_forest
    
    # Evaluate only (no training)
    python train_classifier.py --data training_data/ --evaluate-only
    
    # Show feature importance
    python train_classifier.py --data training_data/ --show-importance
    
    # Use trained model for prediction
    python train_classifier.py --predict --model models/classifier.joblib --metrics metrics.json
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


# Feature definitions - these map to TrainingExample attributes
FEATURE_COLUMNS = [
    # Reasoning features
    "reasoning_quality_score",
    "has_explicit_reasoning",
    "mentions_issue_keywords",
    "mentions_relevant_files",
    "hypothesizes_before_acting",
    "explains_changes",
    "verifies_after_change",
    
    # Phase features
    "exploration_steps",
    "implementation_steps",
    "verification_steps",
    "exploration_pct",
    "implementation_pct",
    "verification_pct",
    "phase_transitions",
    "followed_read_before_write",
    "followed_test_after_change",
    
    # Exploration features
    "files_explored",
    "directories_explored",
    "relevant_file_discovery_step",
    "exploration_efficiency",
    "wasted_explorations",
    "search_to_read_ratio",
    
    # Trajectory features
    "trajectory_length",
    "optimal_length",
    "trajectory_efficiency",
    "unnecessary_steps",
    
    # Convergence features
    "final_similarity",
    "max_progress",
    "converged",
    "monotonic_progress",
    "had_regression",
    "progress_volatility",
    
    # Error recovery features
    "total_errors",
    "recovered_errors",
    "recovery_rate",
    "max_repetition",
    "stuck_episodes",
    "max_stuck_duration",
    
    # Tool usage features
    "total_tool_calls",
    "read_relevant_files",
    "used_str_replace",
    "used_write_file",
    "ran_tests",
    "submitted",
    "tool_errors_count",
    
    # Patch quality features (careful - some leak target info)
    "correct_files_touched",
    # "patch_similarity",  # This is very correlated with target, might leak
    # "line_level_similarity",  # Same issue
    "lines_added",
    "lines_removed",
    "patch_too_large",
    
    # Derived features
    "steps_per_file",
    "edit_to_explore_ratio",
]

# Features that might leak target information (use with caution)
LEAKY_FEATURES = [
    "patch_similarity",
    "line_level_similarity",
    "final_similarity",
    "max_progress",
    "converged",
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


def load_training_data(data_path: str) -> list[dict]:
    """Load training examples from JSONL file."""
    path = Path(data_path)
    
    # Handle directory or file
    if path.is_dir():
        data_file = path / "training_data.jsonl"
    else:
        data_file = path
    
    if not data_file.exists():
        raise FileNotFoundError(f"Training data not found: {data_file}")
    
    examples = []
    with open(data_file) as f:
        for line in f:
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    return examples


def extract_features(examples: list[dict], config: ClassifierConfig) -> tuple:
    """
    Extract feature matrix X and labels y from examples.
    
    Returns:
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,)
        feature_names: list of feature names
    """
    # Determine which features to use
    feature_names = FEATURE_COLUMNS.copy()
    if config.use_leaky_features:
        feature_names.extend(LEAKY_FEATURES)
    
    X = []
    y = []
    
    for ex in examples:
        # Extract features
        features = []
        for feat in feature_names:
            value = ex.get(feat, 0)
            # Convert bools to int
            if isinstance(value, bool):
                value = int(value)
            # Handle missing/None
            if value is None:
                value = 0
            # Handle special cases
            if feat == "relevant_file_discovery_step" and value == -1:
                value = 100  # Large value for "never found"
            features.append(float(value))
        
        X.append(features)
        y.append(int(ex.get("resolved", False)))
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y, feature_names


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
    if hasattr(clf_final, "predict_proba"):
        y_prob = clf_final.predict_proba(X_test)[:, 1]
        results["test_metrics"]["roc_auc"] = float(roc_auc_score(y_test, y_prob))
    
    return results, clf, scaler


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


def save_model(clf, scaler, feature_names: list[str], results: dict, output_dir: str) -> Path:
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
    }, model_file)
    
    # Save results
    results_file = output_path / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Model saved to: {model_file}")
    print(f"Results saved to: {results_file}")
    
    return model_file


def load_model(model_path: str) -> tuple:
    """Load a trained model."""
    import joblib
    
    data = joblib.load(model_path)
    return data["classifier"], data["scaler"], data["feature_names"]


def predict(model_path: str, metrics: dict) -> dict:
    """Use trained model to predict success from metrics."""
    clf, scaler, feature_names = load_model(model_path)
    
    # Extract features
    features = []
    for feat in feature_names:
        value = metrics.get(feat, 0)
        if isinstance(value, bool):
            value = int(value)
        if value is None:
            value = 0
        if feat == "relevant_file_discovery_step" and value == -1:
            value = 100
        features.append(float(value))
    
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
    print("Loading training data...")
    examples = load_training_data(args.data)
    print(f"Loaded {len(examples)} examples")
    
    if len(examples) < 10:
        print("ERROR: Need at least 10 examples for training")
        return
    
    config = ClassifierConfig(
        classifier_type=args.classifier,
        use_leaky_features=args.use_leaky_features,
        normalize=not args.no_normalize,
    )
    
    # Extract features
    print("Extracting features...")
    X, y, feature_names = extract_features(examples, config)
    print(f"Feature matrix: {X.shape}")
    print(f"Positive examples: {y.sum()} ({y.mean():.1%})")
    
    # Show correlations if requested
    if args.show_correlations:
        analyze_feature_correlations(X, y, feature_names)
    
    # Train and evaluate
    print()
    results, clf, scaler = train_and_evaluate(X, y, feature_names, config)
    
    # Print results
    print_results(results)
    
    # Save model
    if not args.evaluate_only:
        save_model(clf, scaler, feature_names, results, args.output)


if __name__ == "__main__":
    main()
