#!/usr/bin/env python
"""
Generate synthetic training data for classifier demo.

Creates realistic training examples based on patterns observed in successful
vs unsuccessful agent runs.
"""

import json
import random
from pathlib import Path


def generate_successful_example(task_id: int, model: str) -> dict:
    """Generate a training example that looks like a successful run."""
    return {
        "task_id": f"task-{task_id:03d}",
        "model": model,
        "provider": "openai" if "gpt" in model else "anthropic",
        "timestamp": "2024-12-28T12:00:00",
        "resolved": True,
        
        # Reasoning - successful agents reason well
        "reasoning_quality_score": random.uniform(0.6, 1.0),
        "has_explicit_reasoning": random.random() > 0.2,
        "mentions_issue_keywords": random.random() > 0.3,
        "mentions_relevant_files": random.random() > 0.4,
        "hypothesizes_before_acting": random.random() > 0.4,
        "explains_changes": random.random() > 0.5,
        "verifies_after_change": random.random() > 0.5,
        
        # Phases - balanced approach
        "exploration_steps": random.randint(2, 5),
        "implementation_steps": random.randint(2, 4),
        "verification_steps": random.randint(1, 3),
        "exploration_pct": random.uniform(0.2, 0.4),
        "implementation_pct": random.uniform(0.3, 0.5),
        "verification_pct": random.uniform(0.1, 0.3),
        "phase_transitions": random.randint(3, 6),
        "followed_read_before_write": random.random() > 0.2,
        "followed_test_after_change": random.random() > 0.4,
        
        # Exploration - efficient
        "exploration_strategy": random.choice(["targeted", "targeted", "breadth_first"]),
        "files_explored": random.randint(2, 5),
        "directories_explored": random.randint(1, 3),
        "relevant_file_discovery_step": random.randint(1, 4),
        "exploration_efficiency": random.uniform(0.5, 1.0),
        "wasted_explorations": random.randint(0, 2),
        "search_to_read_ratio": random.uniform(0.3, 0.8),
        
        # Trajectory - efficient
        "trajectory_length": random.randint(6, 12),
        "optimal_length": random.randint(5, 8),
        "trajectory_efficiency": random.uniform(0.6, 1.0),
        "unnecessary_steps": random.randint(0, 2),
        
        # Convergence - good progress
        "final_similarity": random.uniform(0.7, 1.0),
        "max_progress": random.uniform(0.8, 1.0),
        "converged": random.random() > 0.3,
        "monotonic_progress": random.random() > 0.5,
        "had_regression": random.random() > 0.7,
        "progress_volatility": random.uniform(0.0, 0.2),
        
        # Error recovery - handles errors well
        "total_errors": random.randint(0, 2),
        "recovered_errors": random.randint(0, 2),
        "recovery_rate": random.uniform(0.7, 1.0),
        "max_repetition": random.randint(1, 2),
        "stuck_episodes": random.randint(0, 1),
        "max_stuck_duration": random.randint(0, 2),
        
        # Tool usage - good patterns
        "total_tool_calls": random.randint(6, 15),
        "read_relevant_files": True,
        "used_str_replace": random.random() > 0.3,
        "used_write_file": random.random() > 0.3,
        "ran_tests": random.random() > 0.4,
        "submitted": True,
        "tool_errors_count": random.randint(0, 1),
        
        # Patch quality - good
        "correct_files_touched": True,
        "patch_similarity": random.uniform(0.6, 1.0),
        "line_level_similarity": random.uniform(0.5, 1.0),
        "lines_added": random.randint(1, 20),
        "lines_removed": random.randint(0, 10),
        "patch_too_large": False,
        
        # Derived
        "steps_per_file": random.uniform(1.5, 4.0),
        "edit_to_explore_ratio": random.uniform(0.5, 1.5),
    }


def generate_failed_example(task_id: int, model: str) -> dict:
    """Generate a training example that looks like a failed run."""
    failure_type = random.choice([
        "exploration_failure",
        "implementation_failure",
        "stuck_loop",
        "no_submission",
    ])
    
    example = {
        "task_id": f"task-{task_id:03d}",
        "model": model,
        "provider": "openai" if "gpt" in model else "anthropic",
        "timestamp": "2024-12-28T12:00:00",
        "resolved": False,
        
        # Base values for failed runs
        "reasoning_quality_score": random.uniform(0.0, 0.5),
        "has_explicit_reasoning": random.random() > 0.6,
        "mentions_issue_keywords": random.random() > 0.6,
        "mentions_relevant_files": random.random() > 0.7,
        "hypothesizes_before_acting": random.random() > 0.7,
        "explains_changes": random.random() > 0.6,
        "verifies_after_change": random.random() > 0.8,
        
        "exploration_steps": random.randint(3, 10),
        "implementation_steps": random.randint(0, 3),
        "verification_steps": random.randint(0, 2),
        "exploration_pct": random.uniform(0.4, 0.8),
        "implementation_pct": random.uniform(0.0, 0.3),
        "verification_pct": random.uniform(0.0, 0.2),
        "phase_transitions": random.randint(1, 4),
        "followed_read_before_write": random.random() > 0.5,
        "followed_test_after_change": random.random() > 0.7,
        
        "exploration_strategy": random.choice(["random", "depth_first", "breadth_first"]),
        "files_explored": random.randint(1, 8),
        "directories_explored": random.randint(0, 5),
        "relevant_file_discovery_step": random.choice([-1, -1, random.randint(5, 15)]),
        "exploration_efficiency": random.uniform(0.0, 0.4),
        "wasted_explorations": random.randint(2, 6),
        "search_to_read_ratio": random.uniform(0.8, 2.0),
        
        "trajectory_length": random.randint(10, 20),
        "optimal_length": random.randint(5, 8),
        "trajectory_efficiency": random.uniform(0.2, 0.5),
        "unnecessary_steps": random.randint(3, 10),
        
        "final_similarity": random.uniform(0.0, 0.4),
        "max_progress": random.uniform(0.0, 0.5),
        "converged": False,
        "monotonic_progress": random.random() > 0.8,
        "had_regression": random.random() > 0.4,
        "progress_volatility": random.uniform(0.2, 0.5),
        
        "total_errors": random.randint(1, 5),
        "recovered_errors": random.randint(0, 2),
        "recovery_rate": random.uniform(0.0, 0.5),
        "max_repetition": random.randint(2, 5),
        "stuck_episodes": random.randint(0, 3),
        "max_stuck_duration": random.randint(0, 5),
        
        "total_tool_calls": random.randint(8, 20),
        "read_relevant_files": random.random() > 0.5,
        "used_str_replace": random.random() > 0.6,
        "used_write_file": random.random() > 0.5,
        "ran_tests": random.random() > 0.7,
        "submitted": random.random() > 0.3,
        "tool_errors_count": random.randint(0, 4),
        
        "correct_files_touched": random.random() > 0.5,
        "patch_similarity": random.uniform(0.0, 0.3),
        "line_level_similarity": random.uniform(0.0, 0.3),
        "lines_added": random.randint(0, 50),
        "lines_removed": random.randint(0, 30),
        "patch_too_large": random.random() > 0.7,
        
        "steps_per_file": random.uniform(2.0, 8.0),
        "edit_to_explore_ratio": random.uniform(0.0, 0.5),
    }
    
    # Apply failure-specific patterns
    if failure_type == "exploration_failure":
        example["relevant_file_discovery_step"] = -1
        example["read_relevant_files"] = False
        example["exploration_efficiency"] = random.uniform(0.0, 0.2)
        example["wasted_explorations"] = random.randint(4, 8)
    
    elif failure_type == "stuck_loop":
        example["stuck_episodes"] = random.randint(2, 5)
        example["max_stuck_duration"] = random.randint(3, 8)
        example["max_repetition"] = random.randint(4, 8)
    
    elif failure_type == "no_submission":
        example["submitted"] = False
        example["implementation_steps"] = 0
    
    return example


def generate_dataset(
    n_samples: int = 200,
    success_rate: float = 0.4,
    models: list = None,
) -> list[dict]:
    """Generate a synthetic dataset."""
    if models is None:
        models = ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4-20250514", "o4-mini"]
    
    examples = []
    n_success = int(n_samples * success_rate)
    n_fail = n_samples - n_success
    
    for i in range(n_success):
        model = random.choice(models)
        examples.append(generate_successful_example(i, model))
    
    for i in range(n_fail):
        model = random.choice(models)
        examples.append(generate_failed_example(n_success + i, model))
    
    random.shuffle(examples)
    return examples


def save_dataset(examples: list[dict], output_path: str) -> None:
    """Save dataset to JSONL file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    
    print(f"Saved {len(examples)} examples to {path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("-n", "--samples", type=int, default=200, help="Number of samples")
    parser.add_argument("--success-rate", type=float, default=0.4, help="Success rate")
    parser.add_argument("-o", "--output", default="training_data/training_data.jsonl")
    
    args = parser.parse_args()
    
    print(f"Generating {args.samples} synthetic examples...")
    examples = generate_dataset(args.samples, args.success_rate)
    
    n_success = sum(1 for e in examples if e["resolved"])
    print(f"  Successful: {n_success} ({n_success/len(examples):.1%})")
    print(f"  Failed: {len(examples) - n_success} ({(len(examples)-n_success)/len(examples):.1%})")
    
    save_dataset(examples, args.output)


if __name__ == "__main__":
    main()
