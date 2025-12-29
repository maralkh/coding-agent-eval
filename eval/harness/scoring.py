"""
Unified scoring methods for combining multiple metrics into a single score.

Provides several approaches:
- Weighted linear combination
- Geometric mean (multiplicative)
- Hierarchical scoring
- Percentile rank aggregation
- TOPSIS (multi-criteria decision analysis)
- Pareto rank / dominance count
- PCA-based scoring
- Elo ratings for model comparison

Usage:
    from eval.harness.scoring import UnifiedScorer, compute_elo_ratings
    
    # Single run scoring
    scorer = UnifiedScorer(method="hierarchical")
    score = scorer.score(metrics_dict)
    
    # With reference population (for percentile, TOPSIS, etc.)
    scorer = UnifiedScorer(method="percentile", reference_population=all_runs)
    score = scorer.score(metrics_dict)
    
    # Model comparison with Elo
    elo_ratings = compute_elo_ratings(task_results)
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
from collections import defaultdict
import numpy as np


# Default metric keys used for scoring
DEFAULT_METRICS = [
    "resolved",
    "trajectory_efficiency",
    "exploration_efficiency",
    "reasoning_score",
    "similarity_score",
]


@dataclass
class ScoringConfig:
    """Configuration for unified scoring."""
    
    # Scoring method
    method: str = "hierarchical"  # weighted, geometric, hierarchical, percentile, topsis, pareto, pca
    
    # Weights for weighted/hierarchical methods
    weights: dict = field(default_factory=lambda: {
        "resolved": 0.30,
        "trajectory_efficiency": 0.20,
        "exploration_efficiency": 0.15,
        "reasoning_score": 0.15,
        "similarity_score": 0.10,
        "no_errors": 0.10,
    })
    
    # Metrics to use (subset of available)
    metric_keys: list = field(default_factory=lambda: DEFAULT_METRICS.copy())
    
    # For methods requiring reference population
    reference_population: list = field(default_factory=list)


class UnifiedScorer:
    """
    Combines multiple metrics into a single unified score.
    
    Methods:
        - weighted: Weighted linear combination
        - geometric: Geometric mean (multiplicative)
        - hierarchical: Two-level aggregation (category → final)
        - percentile: Percentile rank vs reference population
        - topsis: TOPSIS multi-criteria decision analysis
        - pareto: Pareto dominance count
        - pca: PCA projection onto first component
    """
    
    def __init__(
        self,
        method: str = "hierarchical",
        weights: Optional[dict] = None,
        metric_keys: Optional[list] = None,
        reference_population: Optional[list] = None,
    ):
        self.method = method
        self.weights = weights or ScoringConfig().weights
        self.metric_keys = metric_keys or DEFAULT_METRICS.copy()
        self.reference_population = reference_population or []
        
        # Validate method
        valid_methods = ["weighted", "geometric", "hierarchical", "percentile", "topsis", "pareto", "pca"]
        if method not in valid_methods:
            raise ValueError(f"Unknown method: {method}. Valid: {valid_methods}")
        
        # Check if reference population is needed
        if method in ["percentile", "topsis", "pareto", "pca"] and not reference_population:
            raise ValueError(f"Method '{method}' requires reference_population")
        
        # Fit PCA if needed
        self._pca_model = None
        self._pca_scaler = None
        if method == "pca" and reference_population:
            self._fit_pca()
    
    def score(self, metrics: dict) -> float:
        """
        Compute unified score for a single run.
        
        Args:
            metrics: Dict with metric values (e.g., resolved, trajectory_efficiency, etc.)
            
        Returns:
            Unified score in [0, 1] range
        """
        method_fn = {
            "weighted": self._weighted_score,
            "geometric": self._geometric_score,
            "hierarchical": self._hierarchical_score,
            "percentile": self._percentile_score,
            "topsis": self._topsis_score,
            "pareto": self._pareto_score,
            "pca": self._pca_score,
        }
        
        return method_fn[self.method](metrics)
    
    def score_batch(self, metrics_list: list[dict]) -> list[float]:
        """Score multiple runs."""
        return [self.score(m) for m in metrics_list]
    
    def _weighted_score(self, metrics: dict) -> float:
        """Weighted linear combination of metrics."""
        score = 0.0
        total_weight = 0.0
        
        for key, weight in self.weights.items():
            if key in metrics:
                score += metrics[key] * weight
                total_weight += weight
            elif key == "no_errors" and "error_rate" in metrics:
                score += (1 - metrics["error_rate"]) * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _geometric_score(self, metrics: dict) -> float:
        """Geometric mean - penalizes poor performance on any metric."""
        values = []
        for key in self.metric_keys:
            val = metrics.get(key, 0)
            # Avoid zeros (would make entire product zero)
            values.append(max(0.01, val))
        
        if not values:
            return 0.0
        
        return float(np.prod(values) ** (1 / len(values)))
    
    def _hierarchical_score(self, metrics: dict) -> float:
        """Two-level aggregation: category scores → final score."""
        # Outcome (binary)
        outcome = metrics.get("resolved", 0)
        
        # Efficiency (average of efficiency metrics)
        efficiency_metrics = [
            metrics.get("trajectory_efficiency", 0),
            metrics.get("exploration_efficiency", 0),
        ]
        efficiency = np.mean(efficiency_metrics) if efficiency_metrics else 0
        
        # Quality (average of quality metrics)
        quality_metrics = [
            metrics.get("reasoning_score", 0),
            metrics.get("similarity_score", 0),
        ]
        quality = np.mean(quality_metrics) if quality_metrics else 0
        
        # Robustness (inverse of error rate)
        error_rate = metrics.get("error_rate", 0)
        if "total_errors" in metrics and "steps" in metrics and metrics["steps"] > 0:
            error_rate = metrics["total_errors"] / metrics["steps"]
        robustness = 1 - min(1, error_rate)
        
        # Weighted combination of categories
        return (
            0.40 * outcome +
            0.25 * efficiency +
            0.20 * quality +
            0.15 * robustness
        )
    
    def _percentile_score(self, metrics: dict) -> float:
        """Score based on percentile rank vs reference population."""
        if not self.reference_population:
            return 0.0
        
        percentiles = []
        for key in self.metric_keys:
            if key not in metrics:
                continue
            
            ref_values = [r.get(key, 0) for r in self.reference_population]
            if not ref_values:
                continue
            
            # Compute percentile
            value = metrics[key]
            below = sum(1 for v in ref_values if v < value)
            equal = sum(1 for v in ref_values if v == value)
            percentile = (below + 0.5 * equal) / len(ref_values)
            percentiles.append(percentile)
        
        return np.mean(percentiles) if percentiles else 0.0
    
    def _topsis_score(self, metrics: dict) -> float:
        """TOPSIS: Technique for Order Preference by Similarity to Ideal Solution."""
        if not self.reference_population:
            return 0.0
        
        # Build matrix
        matrix = []
        for run in self.reference_population:
            row = [run.get(k, 0) for k in self.metric_keys]
            matrix.append(row)
        matrix = np.array(matrix)
        
        current = np.array([metrics.get(k, 0) for k in self.metric_keys])
        
        # Normalize columns (vector normalization)
        col_norms = np.sqrt((matrix ** 2).sum(axis=0))
        col_norms = np.where(col_norms == 0, 1, col_norms)  # Avoid division by zero
        
        norm_matrix = matrix / col_norms
        norm_current = current / col_norms
        
        # Ideal best and worst (assuming higher is better for all metrics)
        ideal_best = norm_matrix.max(axis=0)
        ideal_worst = norm_matrix.min(axis=0)
        
        # Distance to ideal best and worst
        dist_best = np.sqrt(((norm_current - ideal_best) ** 2).sum())
        dist_worst = np.sqrt(((norm_current - ideal_worst) ** 2).sum())
        
        # TOPSIS score: closer to best, farther from worst
        if dist_best + dist_worst == 0:
            return 0.5
        return float(dist_worst / (dist_best + dist_worst))
    
    def _pareto_score(self, metrics: dict) -> float:
        """Score based on Pareto dominance - fraction not dominated by."""
        if not self.reference_population:
            return 0.0
        
        current = [metrics.get(k, 0) for k in self.metric_keys]
        
        dominated_by = 0
        for other in self.reference_population:
            other_vals = [other.get(k, 0) for k in self.metric_keys]
            
            # other dominates current if better or equal on all, strictly better on at least one
            all_geq = all(o >= c for o, c in zip(other_vals, current))
            any_gt = any(o > c for o, c in zip(other_vals, current))
            
            if all_geq and any_gt:
                dominated_by += 1
        
        return 1 - (dominated_by / len(self.reference_population))
    
    def _fit_pca(self):
        """Fit PCA model on reference population."""
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError("PCA scoring requires scikit-learn: pip install scikit-learn")
        
        # Build matrix
        matrix = []
        for run in self.reference_population:
            row = [run.get(k, 0) for k in self.metric_keys]
            matrix.append(row)
        matrix = np.array(matrix)
        
        # Standardize
        self._pca_scaler = StandardScaler()
        matrix_scaled = self._pca_scaler.fit_transform(matrix)
        
        # Fit PCA
        self._pca_model = PCA(n_components=1)
        self._pca_model.fit(matrix_scaled)
        
        # Store min/max for normalization
        all_scores = self._pca_model.transform(matrix_scaled)[:, 0]
        self._pca_min = all_scores.min()
        self._pca_max = all_scores.max()
    
    def _pca_score(self, metrics: dict) -> float:
        """Project onto first principal component."""
        if self._pca_model is None:
            return 0.0
        
        current = np.array([metrics.get(k, 0) for k in self.metric_keys]).reshape(1, -1)
        current_scaled = self._pca_scaler.transform(current)
        
        score = self._pca_model.transform(current_scaled)[0, 0]
        
        # Normalize to 0-1 range
        if self._pca_max == self._pca_min:
            return 0.5
        return float((score - self._pca_min) / (self._pca_max - self._pca_min))


def compute_elo_ratings(
    task_results: dict[str, dict[str, dict]],
    metric_key: str = "resolved",
    initial_elo: float = 1500,
    k_factor: float = 32,
) -> dict[str, float]:
    """
    Compute Elo ratings for models based on head-to-head task performance.
    
    Args:
        task_results: Nested dict: task_results[model][task_id] -> metrics dict
        metric_key: Metric to use for comparison (default: "resolved")
        initial_elo: Starting Elo rating
        k_factor: Update factor (higher = more volatile)
        
    Returns:
        Dict mapping model name to Elo rating
    """
    elo = defaultdict(lambda: initial_elo)
    
    models = list(task_results.keys())
    if not models:
        return {}
    
    # Get all tasks (from first model)
    tasks = list(task_results[models[0]].keys())
    
    for task in tasks:
        # Compare all pairs on this task
        for i, m1 in enumerate(models):
            for m2 in models[i+1:]:
                # Get scores
                s1 = task_results[m1].get(task, {}).get(metric_key, 0)
                s2 = task_results[m2].get(task, {}).get(metric_key, 0)
                
                # Expected scores (Elo formula)
                e1 = 1 / (1 + 10 ** ((elo[m2] - elo[m1]) / 400))
                e2 = 1 - e1
                
                # Actual outcome (1 for win, 0.5 for tie, 0 for loss)
                if s1 > s2:
                    a1, a2 = 1, 0
                elif s2 > s1:
                    a1, a2 = 0, 1
                else:
                    a1, a2 = 0.5, 0.5
                
                # Update Elo
                elo[m1] += k_factor * (a1 - e1)
                elo[m2] += k_factor * (a2 - e2)
    
    return dict(elo)


def compute_composite_score(
    metrics: dict,
    method: str = "hierarchical",
    reference_population: Optional[list] = None,
) -> float:
    """
    Convenience function to compute unified score.
    
    Args:
        metrics: Dict with metric values
        method: Scoring method (weighted, geometric, hierarchical, percentile, topsis, pareto, pca)
        reference_population: List of metric dicts for comparison-based methods
        
    Returns:
        Unified score in [0, 1]
    """
    scorer = UnifiedScorer(
        method=method,
        reference_population=reference_population,
    )
    return scorer.score(metrics)


# Shortcut scoring functions

def weighted_score(metrics: dict, weights: Optional[dict] = None) -> float:
    """Weighted linear combination."""
    scorer = UnifiedScorer(method="weighted", weights=weights)
    return scorer.score(metrics)


def geometric_score(metrics: dict) -> float:
    """Geometric mean of metrics."""
    scorer = UnifiedScorer(method="geometric")
    return scorer.score(metrics)


def hierarchical_score(metrics: dict) -> float:
    """Hierarchical two-level scoring."""
    scorer = UnifiedScorer(method="hierarchical")
    return scorer.score(metrics)


def percentile_score(metrics: dict, reference_population: list) -> float:
    """Percentile rank vs reference population."""
    scorer = UnifiedScorer(method="percentile", reference_population=reference_population)
    return scorer.score(metrics)


def topsis_score(metrics: dict, reference_population: list) -> float:
    """TOPSIS multi-criteria decision score."""
    scorer = UnifiedScorer(method="topsis", reference_population=reference_population)
    return scorer.score(metrics)


def pareto_score(metrics: dict, reference_population: list) -> float:
    """Pareto dominance score."""
    scorer = UnifiedScorer(method="pareto", reference_population=reference_population)
    return scorer.score(metrics)
