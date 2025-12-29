"""
Sampling strategies for improving agent performance.

Supports:
- Best-of-N: Generate N solutions, pick the best by score
- Majority voting: Generate N solutions, pick most common patch
- Pass@k: Report if any of k attempts succeed (for metrics)
"""

from dataclasses import dataclass, field
from typing import Callable, Optional
from collections import Counter
import hashlib


@dataclass
class SamplingConfig:
    """Configuration for sampling strategies."""
    
    # Number of samples to generate
    n_samples: int = 1
    
    # Strategy: "best_of_n", "majority_vote", "pass_at_k", "first"
    strategy: str = "first"
    
    # For best_of_n: scoring function (default uses heuristics)
    # Takes RepoAgentResult and returns float score
    scorer: Optional[Callable] = None
    
    # Temperature override for sampling (None = use default)
    temperature: Optional[float] = None
    
    # Whether to run samples in parallel (future)
    parallel: bool = False
    
    def __post_init__(self):
        if self.n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if self.strategy not in ["best_of_n", "majority_vote", "pass_at_k", "first"]:
            raise ValueError(f"Unknown strategy: {self.strategy}")


@dataclass 
class SamplingResult:
    """Result of sampling multiple agent runs."""
    
    # The selected/final result
    selected_result: "RepoAgentResult"
    
    # All individual results
    all_results: list["RepoAgentResult"] = field(default_factory=list)
    
    # Scores for each result (if best_of_n)
    scores: list[float] = field(default_factory=list)
    
    # Statistics
    n_samples: int = 1
    n_submitted: int = 0
    n_with_patch: int = 0
    
    # For pass@k
    any_resolved: bool = False
    
    # Strategy used
    strategy: str = "first"
    
    # Which sample was selected
    selected_index: int = 0


def default_scorer(result: "RepoAgentResult") -> float:
    """
    Default scoring function for best-of-N selection.
    
    Higher score = better. Uses heuristics based on:
    - Whether a patch was generated
    - Patch size (prefer smaller, focused patches)
    - Number of steps (prefer efficient solutions)
    - Whether agent submitted
    """
    score = 0.0
    
    # Did it submit?
    if result.success:
        score += 10.0
    
    # Does it have a patch?
    if result.patch and len(result.patch.strip()) > 0:
        score += 5.0
        
        # Prefer smaller patches (less likely to be wrong)
        patch_lines = len(result.patch.split('\n'))
        if patch_lines < 20:
            score += 2.0
        elif patch_lines < 50:
            score += 1.0
        elif patch_lines > 200:
            score -= 2.0  # Very large patch is suspicious
    
    # Prefer fewer steps (more efficient)
    if result.steps > 0:
        efficiency_bonus = max(0, 3.0 - (result.steps / 10.0))
        score += efficiency_bonus
    
    # Has explanation?
    if result.explanation and len(result.explanation) > 20:
        score += 1.0
    
    # Penalize errors
    if result.error:
        score -= 5.0
    
    return score


def compute_patch_hash(patch: str) -> str:
    """Compute a normalized hash of a patch for comparison."""
    if not patch:
        return "empty"
    
    # Normalize: strip whitespace, sort lines for comparison
    lines = [line.strip() for line in patch.split('\n') if line.strip()]
    # Remove diff headers that vary (timestamps, etc.)
    lines = [l for l in lines if not l.startswith('diff --git') 
             and not l.startswith('index ')
             and not l.startswith('---')
             and not l.startswith('+++')]
    normalized = '\n'.join(sorted(lines))
    return hashlib.md5(normalized.encode()).hexdigest()


def select_best_of_n(
    results: list["RepoAgentResult"],
    scorer: Optional[Callable] = None,
) -> tuple["RepoAgentResult", int, list[float]]:
    """
    Select the best result from N samples using a scoring function.
    
    Returns: (best_result, best_index, all_scores)
    """
    if not results:
        raise ValueError("No results to select from")
    
    if len(results) == 1:
        return results[0], 0, [0.0]
    
    score_fn = scorer or default_scorer
    scores = [score_fn(r) for r in results]
    
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return results[best_idx], best_idx, scores


def select_majority_vote(
    results: list["RepoAgentResult"],
) -> tuple["RepoAgentResult", int, dict]:
    """
    Select the most common patch from N samples.
    
    Returns: (selected_result, selected_index, vote_counts)
    """
    if not results:
        raise ValueError("No results to select from")
    
    if len(results) == 1:
        return results[0], 0, {"single": 1}
    
    # Group by patch hash
    patch_hashes = [compute_patch_hash(r.patch) for r in results]
    vote_counts = Counter(patch_hashes)
    
    # Find most common
    most_common_hash, count = vote_counts.most_common(1)[0]
    
    # Find first result with this hash
    for i, h in enumerate(patch_hashes):
        if h == most_common_hash:
            return results[i], i, dict(vote_counts)
    
    # Fallback (shouldn't happen)
    return results[0], 0, dict(vote_counts)


def run_sampling(
    solve_fn: Callable[[], "RepoAgentResult"],
    config: SamplingConfig,
    reset_fn: Optional[Callable[[], None]] = None,
) -> SamplingResult:
    """
    Run sampling strategy.
    
    Args:
        solve_fn: Function that runs the agent once and returns result
        config: Sampling configuration
        reset_fn: Optional function to reset repo state between runs
        
    Returns:
        SamplingResult with selected result and statistics
    """
    from .repo_agent import RepoAgentResult  # Import here to avoid circular
    
    results = []
    
    for i in range(config.n_samples):
        # Reset repo state between samples
        if reset_fn and i > 0:
            reset_fn()
        
        # Run agent
        result = solve_fn()
        results.append(result)
    
    # Compute statistics
    n_submitted = sum(1 for r in results if r.success)
    n_with_patch = sum(1 for r in results if r.patch and r.patch.strip())
    any_resolved = n_submitted > 0  # Will be updated by evaluator
    
    # Select based on strategy
    scores = []
    selected_index = 0
    
    if config.strategy == "first" or len(results) == 1:
        selected = results[0]
        selected_index = 0
        
    elif config.strategy == "best_of_n":
        selected, selected_index, scores = select_best_of_n(
            results, config.scorer
        )
        
    elif config.strategy == "majority_vote":
        selected, selected_index, _ = select_majority_vote(results)
        
    elif config.strategy == "pass_at_k":
        # For pass@k, we still need to return one result
        # Prefer one with a patch
        for i, r in enumerate(results):
            if r.patch and r.patch.strip():
                selected = r
                selected_index = i
                break
        else:
            selected = results[0]
            selected_index = 0
    else:
        selected = results[0]
    
    return SamplingResult(
        selected_result=selected,
        all_results=results,
        scores=scores,
        n_samples=config.n_samples,
        n_submitted=n_submitted,
        n_with_patch=n_with_patch,
        any_resolved=any_resolved,
        strategy=config.strategy,
        selected_index=selected_index,
    )
