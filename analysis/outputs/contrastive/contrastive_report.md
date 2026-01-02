# Contrastive Analysis: Success vs Failure

## Overview

This analysis identifies the key factors that distinguish successful
trajectories from failed ones, providing actionable insights for improvement.

**Dataset**: 208 trajectories (38 success, 170 failure)

## Key Success Factors

The following features show statistically significant differences between
successful and failed trajectories (p < 0.05):

- **tool errors occurred**: Failed trajectories show higher values (0.69 vs 0.00, effect size: large)
- **no changes made**: Failed trajectories show higher values (0.59 vs 0.00, effect size: large)
- **used str replace**: Successful trajectories show higher values (1.00 vs 0.47, effect size: large)
- **submitted**: Successful trajectories show higher values (0.95 vs 0.44, effect size: large)
- **trajectory efficiency**: Successful trajectories show higher values (0.93 vs 0.44, effect size: large)
- **correct files touched**: Successful trajectories show higher values (0.68 vs 0.26, effect size: large)
- **exploration efficiency**: Successful trajectories show higher values (0.79 vs 0.45, effect size: large)
- **recovery rate**: Successful trajectories show higher values (0.94 vs 0.64, effect size: medium)

## Top Discriminative Features

| Rank | Feature | Effect Size | Direction | Success Mean | Failure Mean |
|------|---------|-------------|-----------|--------------|--------------|
| 1 | tool errors occurred | -1.64 (large) | higher in failu | 0.00 | 0.69 |
| 2 | no changes made | -1.33 (large) | higher in failu | 0.00 | 0.59 |
| 3 | used str replace | 1.17 (large) | higher in succe | 1.00 | 0.47 |
| 4 | submitted | 1.11 (large) | higher in succe | 0.95 | 0.44 |
| 5 | trajectory efficiency | 0.95 (large) | higher in succe | 0.93 | 0.44 |
| 6 | correct files touched | 0.94 (large) | higher in succe | 0.68 | 0.26 |
| 7 | exploration efficiency | 0.85 (large) | higher in succe | 0.79 | 0.45 |
| 8 | recovery rate | 0.73 (medium) | higher in succe | 0.94 | 0.64 |
| 9 | followed read before writ | 0.71 (medium) | higher in succe | 1.00 | 0.71 |
| 10 | read relevant files | 0.66 (medium) | higher in succe | 0.95 | 0.65 |

## Recipe for Success

### ✓ DO (higher in successful trajectories):

- **used str replace**: Target ≥ 1.00 (vs 0.47 in failures)
- **submitted**: Target ≥ 0.95 (vs 0.44 in failures)
- **trajectory efficiency**: Target ≥ 0.93 (vs 0.44 in failures)
- **correct files touched**: Target ≥ 0.68 (vs 0.26 in failures)
- **exploration efficiency**: Target ≥ 0.79 (vs 0.45 in failures)

### ✗ AVOID (higher in failed trajectories):

- **tool errors occurred**: Keep below 0.00 (failures average 0.69)
- **no changes made**: Keep below 0.00 (failures average 0.59)
- **wrong files modified**: Keep below 0.00 (failures average 0.16)
- **hit max steps**: Keep below 0.08 (failures average 0.21)
- **verifies after change**: Keep below 0.00 (failures average 0.06)

## Model-Specific Insights

### claude-opus-4-20250514

*Insufficient variation for analysis*

### gpt-4o

Success rate: 14/52

Top differentiating factors:
- tool errors occurred: ↓ (d=-4.87)
- no changes made: ↓ (d=-0.93)
- correct files touched: ↑ (d=0.88)

### gpt-5.1

Success rate: 18/52

Top differentiating factors:
- tool errors occurred: ↓ (d=-6.97)
- wrong files modified: ↓ (d=-0.90)
- phase transitions: ↓ (d=-0.60)

### o4-mini

Success rate: 6/52

Top differentiating factors:
- no changes made: ↓ (d=-1.36)
- used str replace: ↑ (d=1.30)
- implementation steps: ↑ (d=1.15)

## Feature Importance (Random Forest)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | tool errors occurred | 0.195 |
| 2 | similarity score | 0.102 |
| 3 | patch similarity | 0.083 |
| 4 | final similarity | 0.053 |
| 5 | progress volatility | 0.050 |
| 6 | max progress | 0.048 |
| 7 | lines added | 0.042 |
| 8 | edit to explore ratio | 0.036 |
| 9 | no changes made | 0.035 |
| 10 | wrong files modified | 0.027 |

## Actionable Recommendations

Based on this analysis:

1. **Minimize tool errors**: The strongest predictor of failure
2. **Maximize patch similarity**: Successful agents produce patches closer to gold
3. **Efficient exploration**: Find relevant files quickly, avoid excessive wandering
4. **Make changes**: Agents that don't modify files almost always fail
5. **Verify after editing**: Running tests after changes correlates with success

## Visualizations

- `contrastive_features.png`: Effect sizes for top features
- `success_failure_dist.png`: Distribution comparisons
- `model_contrastive.png`: Per-model success factors
- `success_recipe.png`: Visual recipe for success
