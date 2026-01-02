# Reward Modeling Analysis


## Overview

This analysis demonstrates how the success classifier can serve as a **reward model**
for reinforcement learning or best-of-N selection strategies.

Key insight: The classifier's P(success) provides a **continuous signal** that can
guide learning, unlike binary pass/fail which only gives feedback at episode end.

## Reward Signal Quality

A good reward model should clearly separate successful from failed trajectories.

| Metric | Value |
|--------|-------|
| Success mean | 0.539 |
| Success std | 0.150 |
| Failure mean | 0.103 |
| Failure std | 0.135 |
| **Separation** | **0.435** |
| Distribution overlap | 0.097 |

**Interpretation**: Separation of 0.435 indicates 
the reward model can effectively distinguish successful from failed trajectories.
Lower overlap = better discriminative power.

## Reward Formulations

We compare different ways to formulate the reward signal:

| Formulation | Mean | Std | Variance | Description |
|-------------|------|-----|----------|-------------|
| Sparse | 0.183 | 0.386 | 0.149 | Binary 0/1 based on actual outcome |
| Dense | 0.183 | 0.217 | 0.047 | Continuous P(success) from classifier |
| Shaped | 0.183 | 0.217 | 0.047 | P(success) + bonus for high confidence |
| Advantage | -0.000 | 0.217 | 0.047 | P(success) - mean(P) (centered) |

**Key finding**: Dense reward has similar variance to sparse but provides signal
throughout the trajectory rather than only at the end.

## RewardModel API

The reward model is packaged as a Python class for easy integration:

```python
from analysis.reward_model import RewardModel

# Load pre-trained model
rm = RewardModel.load('reward_model.pkl')

# Score a single trajectory
reward = rm.score({
    'reasoning_quality_score': 0.8,
    'exploration_efficiency': 0.6,
    'trajectory_length': 10,
    # ... other features
})
print(f'Predicted success probability: {reward:.2%}')

# Score batch of trajectories
rewards = rm.score_batch([traj1, traj2, traj3])

# Explain prediction
contributions = rm.explain(trajectory_features)
```

## Applications

1. **RL Training Reward**: Use P(success) as dense reward signal instead of sparse 0/1
2. **Best-of-N Ranking**: Rank N candidate solutions by reward, select highest
3. **Early Stopping**: Abandon trajectories where reward drops below threshold
4. **Curriculum Learning**: Prioritize tasks where model is uncertain (reward ≈ 0.5)

## Visualizations

- `reward_distributions.png`: Success vs failure reward distributions
- `reward_calibration.png`: Calibration curve showing prediction reliability
- `reward_separation.png`: Box plot comparing reward by outcome
- `trajectory_simulation.png`: Reward evolution over trajectory progress