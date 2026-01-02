# Cross-Repository Transfer Analysis

## Overview

This analysis evaluates how well the behavioral features and classifier
generalize across different task groups (simulating different repositories).

> **Note**: Since we only have scikit-learn data, we create 'pseudo-repos'
> by splitting tasks into groups. True cross-repo transfer would require
> data from multiple actual repositories.

## Key Findings

- **Universal features**: 40/59 features transfer well across task groups
- **Repo-specific features**: 19 features show high variance across groups
- **Transfer gap**: 0.15 (same-group vs cross-group accuracy difference)
- **Leave-one-task-out accuracy**: 94.2% ± 13.5%

## Feature Transferability

### Most Transferable Features (Universal)

These features show consistent behavior across task groups:

| Feature | Stability Score | Cross-Repo CV |
|---------|-----------------|---------------|
| followed_read_before_write | 0.075 | 0.015 |
| recovery_rate | 0.092 | 0.046 |
| read_relevant_files | 0.099 | 0.046 |
| exploration_pct | 0.139 | 0.066 |
| implementation_pct | 0.143 | 0.070 |
| ran_tests | 0.146 | 0.104 |
| exploration_efficiency | 0.153 | 0.109 |
| mentions_issue_keywords | 0.155 | 0.087 |
| no_changes_made | 0.167 | 0.088 |
| recovered_errors | 0.168 | 0.110 |

### Least Transferable Features (Repo-Specific)

These features vary significantly across task groups:

| Feature | Stability Score | Cross-Repo CV |
|---------|-----------------|---------------|
| used_write_file | 3.000 | 2.000 |
| had_regression | 1.438 | 1.277 |
| hypothesizes_before_acting | 1.435 | 1.414 |
| stuck_episodes | 1.415 | 1.267 |
| model_got_stuck | 1.288 | 1.277 |
| patch_too_large | 1.288 | 1.277 |
| verification_pct | 1.000 | 0.000 |
| converged | 1.000 | 0.000 |
| max_stuck_duration | 0.963 | 0.816 |
| line_level_similarity | 0.770 | 0.607 |

## Transfer Performance Matrix

Accuracy when training on one group and testing on another:

| Train \ Test | repo_A | repo_B | repo_C | repo_D |
|---|---|---|---|---|
| repo_A | 0.98 | 0.75 | 0.77 | 0.71 |
| repo_B | 0.96 | 1.00 | 0.90 | 0.88 |
| repo_C | 0.92 | 0.83 | 0.98 | 0.71 |
| repo_D | 0.83 | 0.90 | 0.87 | 1.00 |

**Same-group accuracy**: 0.99
**Cross-group accuracy**: 0.84
**Transfer gap**: 0.15

## Leave-One-Task-Out Cross-Validation

Most rigorous test: train on all tasks except one, predict that task.

- **Mean accuracy**: 94.2%
- **Std deviation**: 13.5%
- **Min accuracy**: 50.0%
- **Max accuracy**: 100.0%
- **Number of folds**: 52

## Recommendations for Cross-Repo Transfer

Based on this analysis:

1. **Use universal features for transfer**: Focus on features with stability score < 0.5
2. **Retrain on target repo**: Even small amounts of target data can help
3. **Monitor feature distributions**: Check if target repo has similar feature ranges
4. **Ensemble approach**: Combine base model with target-specific fine-tuning

## Features to Prioritize for Transfer

The following features are both predictive AND stable:

- **followed_read_before_write**: stability=0.075
- **recovery_rate**: stability=0.092
- **read_relevant_files**: stability=0.099
- **exploration_pct**: stability=0.139
- **implementation_pct**: stability=0.143

## Visualizations

- `feature_stability.png`: Feature transferability analysis
- `transfer_matrix.png`: Cross-group transfer performance
- `generalization_analysis.png`: Overall generalization metrics