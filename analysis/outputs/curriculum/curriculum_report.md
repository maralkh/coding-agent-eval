# Curriculum Learning Signal Analysis

## Overview

This analysis ranks tasks by their **informativeness** for agent training.
Tasks where the classifier is uncertain (P ≈ 0.5) or models disagree are most valuable
for learning, as they represent the decision boundary the model needs to learn.

## Task Difficulty Distribution

| Category | Count | Success Rate Range |
|----------|-------|-------------------|
| Easy | 5 | ≥75% |
| Medium | 10 | 25-75% |
| Hard | 37 | ≤25% |

## Key Metrics

- **Total tasks**: 52
- **Mean success rate**: 18.3%
- **Mean classifier uncertainty**: 0.365
- **Tasks with unanimous model agreement**: 34
- **Tasks with model disagreement**: 18

## Top 10 Most Informative Tasks

These tasks are best for training—they're at the decision boundary where models disagree
or the classifier is uncertain.

| Rank | Task ID | Success Rate | Uncertainty | Informativeness |
|------|---------|--------------|-------------|-----------------|
| 1 | scikit-learn__scikit-learn-30395 | 50% | 0.751 | 0.827 |
| 2 | scikit-learn__scikit-learn-30103 | 50% | 0.727 | 0.800 |
| 3 | scikit-learn__scikit-learn-30898 | 50% | 0.657 | 0.723 |
| 4 | scikit-learn__scikit-learn-30838 | 50% | 0.575 | 0.632 |
| 5 | scikit-learn__scikit-learn-30649 | 50% | 0.521 | 0.573 |
| 6 | scikit-learn__scikit-learn-30454 | 75% | 0.938 | 0.563 |
| 7 | scikit-learn__scikit-learn-30040 | 75% | 0.927 | 0.556 |
| 8 | scikit-learn__scikit-learn-30944 | 50% | 0.478 | 0.525 |
| 9 | scikit-learn__scikit-learn-30644 | 75% | 0.855 | 0.513 |
| 10 | scikit-learn__scikit-learn-30443 | 50% | 0.440 | 0.484 |

## Easiest Tasks (All Models Succeed)

| Task ID | Success Rate | Models Succeeded |
|---------|--------------|------------------|
| scikit-learn__scikit-learn-30040 | 75% | gpt-4o, gpt-5.1, o4-mini |
| scikit-learn__scikit-learn-30454 | 75% | gpt-4o, gpt-5.1, o4-mini |
| scikit-learn__scikit-learn-30535 | 75% | gpt-4o, gpt-5.1, o4-mini |
| scikit-learn__scikit-learn-30644 | 75% | gpt-4o, gpt-5.1, o4-mini |
| scikit-learn__scikit-learn-30956 | 75% | gpt-4o, gpt-5.1, o4-mini |

## Hardest Tasks (All Models Fail)

| Task ID | Success Rate | Models Failed |
|---------|--------------|---------------|
| scikit-learn__scikit-learn-30022 | 0% | claude-opus-4-20250514, gpt-4o, gpt-5.1, |
| scikit-learn__scikit-learn-30047 | 0% | claude-opus-4-20250514, gpt-4o, gpt-5.1, |
| scikit-learn__scikit-learn-30100 | 0% | claude-opus-4-20250514, gpt-4o, gpt-5.1, |
| scikit-learn__scikit-learn-30101 | 0% | claude-opus-4-20250514, gpt-4o, gpt-5.1, |
| scikit-learn__scikit-learn-30112 | 0% | claude-opus-4-20250514, gpt-4o, gpt-5.1, |

## Curriculum Ordering Strategy

For training, order tasks as:

1. **Easy tasks first**: Build confidence, learn basic patterns
2. **Medium tasks (most informative)**: Learn decision boundaries
3. **Hard tasks last**: Tackle edge cases after mastering basics

Within each category, prioritize by informativeness (model disagreement + uncertainty).

## Practical Applications

1. **Curriculum Learning**: Train agents on easy → hard progression
2. **Active Learning**: Focus data collection on uncertain/boundary tasks
3. **Evaluation Design**: Use informative tasks for discriminative benchmarks
4. **Ensemble Training**: Use tasks with model disagreement to train diverse ensembles

## Visualizations

- `curriculum_ordering.png`: Task difficulty spectrum and curriculum order
- `model_agreement.png`: Heatmap of which models solved which tasks
- `boundary_tasks.png`: Top informative tasks for training
