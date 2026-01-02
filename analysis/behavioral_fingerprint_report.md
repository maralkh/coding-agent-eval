# Behavioral Fingerprinting Analysis

> **Note**: Claude Opus 4 data appears incomplete—0% success rate suggests
> the agent may not have been properly configured or runs were interrupted.
> Behavioral metrics for this model should be interpreted with caution.

## Overview

This analysis creates 'behavioral profiles' for each coding agent model,
revealing how they differ in their approach to solving tasks.

## Model Behavioral Profiles

### Behavioral Dimension Scores

Scores normalized to [0, 1] where higher = better performance in that dimension.

| Model | Reasoning | Exploration | Implementation | Verification | Efficiency | Robustness |
|-------|--------|--------|--------|--------|--------|--------|
| claude-opus-4-20250514 | 0.11 | 0.00 | 0.25 | 0.12 | 0.75 | 0.75 |
| gpt-4o | 1.00 | 0.74 | 0.85 | 0.71 | 0.81 | 0.60 |
| gpt-5.1 | 0.00 | 0.74 | 0.69 | 0.60 | 0.49 | 0.40 |
| o4-mini | 0.00 | 0.86 | 0.33 | 0.48 | 0.37 | 0.62 |

### Key Differentiators

- **claude-opus-4-20250514**: Strongest in _efficiency_ (0.75), weakest in _exploration_ (0.00)
- **gpt-4o**: Strongest in _reasoning_ (1.00), weakest in _robustness_ (0.60)
- **gpt-5.1**: Strongest in _exploration_ (0.74), weakest in _reasoning_ (0.00)
- **o4-mini**: Strongest in _exploration_ (0.86), weakest in _reasoning_ (0.00)

## Model Similarity

Cosine similarity based on behavioral features (1.0 = identical behavior):

| Model | claude-opus-4-20250514 | gpt-4o | gpt-5.1 | o4-mini |
|-------|------|------|------|------|
| claude-opus-4-20250514 | 1.00 | -0.53 | -0.63 | -0.38 |
| gpt-4o | -0.53 | 1.00 | 0.01 | -0.29 |
| gpt-5.1 | -0.63 | 0.01 | 1.00 | -0.04 |
| o4-mini | -0.38 | -0.29 | -0.04 | 1.00 |

## PCA Interpretation

Top features contributing to the first principal component (explains most variance):

| Feature | PC1 Loading |
|---------|-------------|
| final_similarity | +0.171 |
| exploration_efficiency | +0.170 |
| followed_read_before_write | +0.170 |
| phase_transitions | +0.169 |
| max_progress | +0.169 |
| recovery_rate | +0.169 |
| followed_test_after_change | +0.167 |
| steps_per_file | +0.166 |
| read_relevant_files | +0.166 |
| search_to_read_ratio | +0.163 |

## Insights

1. **Most similar models**: gpt-4o and gpt-5.1 (similarity: 0.01)
2. **Most different models**: claude-opus-4-20250514 and gpt-5.1 (similarity: -0.63)

3. **Most exploratory**: o4-mini (exploration score: 0.86)
4. **Most direct/implementation-focused**: gpt-4o (implementation score: 0.85)

## Visualizations

- `fingerprint_2d.png`: 2D PCA projection showing model positions in behavior space
- `radar_chart.png`: Radar chart comparing behavioral dimensions
- `behavioral_clusters.png`: Hierarchical clustering dendrogram
- `feature_heatmap.png`: Heatmap of top distinguishing features