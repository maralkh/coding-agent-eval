#!/usr/bin/env python3
"""
Task Generator: Create Synthetic Tasks Targeting Model Weaknesses

Uses analysis insights to generate task specifications that target
specific model weaknesses or test particular capabilities.

Usage:
------
    python analysis/task_generator.py --data training_data/real_training_data.jsonl

Outputs:
--------
    generated_tasks.json         Generated task specifications
    task_templates.json          Reusable task templates
    weakness_analysis.png        Visualization of model weaknesses
    task_generator_report.md     Summary report

Approach:
---------
    1. Analyze where each model fails (failure modes)
    2. Identify skill gaps from behavioral fingerprints
    3. Generate task specifications that stress-test these gaps
    4. Create templates for different difficulty levels
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Optional
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TaskSpec:
    """Specification for a generated task."""
    task_id: str
    title: str
    description: str
    difficulty: str  # easy, medium, hard
    target_weakness: str  # What weakness this tests
    target_models: list  # Which models this targets
    required_skills: list  # Skills needed to solve
    estimated_steps: int
    file_patterns: list  # Expected file patterns to touch
    success_criteria: str
    rationale: str  # Why this task was generated


@dataclass  
class TaskTemplate:
    """Reusable template for generating tasks."""
    template_id: str
    name: str
    category: str  # exploration, implementation, debugging, refactoring
    difficulty: str
    description_template: str
    required_skills: list
    weakness_targeted: str
    example_tasks: list


def load_training_data(data_path: Path) -> pd.DataFrame:
    """Load training data from JSONL file."""
    records = []
    with open(data_path) as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def analyze_model_weaknesses(df: pd.DataFrame) -> dict:
    """
    Analyze where each model struggles.
    
    Returns dict mapping model -> weakness profile
    """
    weaknesses = {}
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        failures = model_df[~model_df['resolved']]
        successes = model_df[model_df['resolved']]
        
        profile = {
            'success_rate': model_df['resolved'].mean(),
            'n_success': len(successes),
            'n_failure': len(failures),
            'weaknesses': [],
            'strengths': [],
        }
        
        if len(failures) > 0:
            # Analyze failure patterns
            failure_patterns = {
                'tool_errors': failures['tool_errors_occurred'].mean() if 'tool_errors_occurred' in failures else 0,
                'no_changes': failures['no_changes_made'].mean() if 'no_changes_made' in failures else 0,
                'wrong_files': failures['wrong_files_modified'].mean() if 'wrong_files_modified' in failures else 0,
                'hit_max_steps': failures['hit_max_steps'].mean() if 'hit_max_steps' in failures else 0,
                'low_exploration_efficiency': (failures['exploration_efficiency'] < 0.5).mean() if 'exploration_efficiency' in failures else 0,
                'no_verification': (failures['verification_steps'] == 0).mean() if 'verification_steps' in failures else 0,
            }
            
            # Identify top weaknesses (above 50% occurrence in failures)
            for pattern, rate in failure_patterns.items():
                if rate > 0.5:
                    profile['weaknesses'].append({
                        'pattern': pattern,
                        'rate': rate,
                        'description': get_weakness_description(pattern)
                    })
        
        if len(successes) > 0:
            # Analyze success patterns
            profile['avg_success_steps'] = successes['trajectory_length'].mean() if 'trajectory_length' in successes else 0
            profile['avg_success_efficiency'] = successes['trajectory_efficiency'].mean() if 'trajectory_efficiency' in successes else 0
        
        weaknesses[model] = profile
    
    return weaknesses


def get_weakness_description(pattern: str) -> str:
    """Get human-readable description of weakness pattern."""
    descriptions = {
        'tool_errors': 'Frequent tool usage errors, struggles with API',
        'no_changes': 'Fails to make any code changes, gets stuck in exploration',
        'wrong_files': 'Modifies incorrect files, poor localization',
        'hit_max_steps': 'Runs out of steps, inefficient exploration',
        'low_exploration_efficiency': 'Wastes steps on irrelevant exploration',
        'no_verification': 'Does not verify changes with tests',
    }
    return descriptions.get(pattern, pattern)


def generate_task_templates() -> list[TaskTemplate]:
    """Generate reusable task templates for different weakness types."""
    
    templates = [
        # Exploration challenges
        TaskTemplate(
            template_id='exploration_deep',
            name='Deep Codebase Navigation',
            category='exploration',
            difficulty='hard',
            description_template='Fix a bug in {module} that requires understanding the interaction between {component1} and {component2}. The bug manifests as {symptom} but the root cause is in a different file.',
            required_skills=['code navigation', 'dependency tracing', 'systematic search'],
            weakness_targeted='low_exploration_efficiency',
            example_tasks=['Fix import cycle', 'Resolve cross-module type error']
        ),
        TaskTemplate(
            template_id='exploration_targeted',
            name='Targeted File Discovery',
            category='exploration',
            difficulty='medium',
            description_template='The {feature} in {module} is broken. Find the relevant implementation file and fix the issue. Hint: the feature was recently refactored.',
            required_skills=['code search', 'git history analysis', 'feature tracing'],
            weakness_targeted='wrong_files',
            example_tasks=['Find moved function', 'Locate renamed class']
        ),
        
        # Implementation challenges
        TaskTemplate(
            template_id='implementation_precise',
            name='Precise Edit Challenge',
            category='implementation',
            difficulty='medium',
            description_template='Fix the {bug_type} bug in {function}. The fix requires exactly {n_lines} line change(s). Any additional changes will break other functionality.',
            required_skills=['precise editing', 'minimal changes', 'side effect awareness'],
            weakness_targeted='tool_errors',
            example_tasks=['Fix off-by-one error', 'Correct parameter order']
        ),
        TaskTemplate(
            template_id='implementation_multi',
            name='Multi-File Coordination',
            category='implementation',
            difficulty='hard',
            description_template='Implement {feature} which requires coordinated changes across {n_files} files: {file_list}. Changes must be consistent.',
            required_skills=['multi-file editing', 'consistency maintenance', 'API design'],
            weakness_targeted='no_changes',
            example_tasks=['Add new parameter to API', 'Refactor shared utility']
        ),
        
        # Debugging challenges
        TaskTemplate(
            template_id='debugging_error_trace',
            name='Error Trace Analysis',
            category='debugging',
            difficulty='medium',
            description_template='Users report: "{error_message}". The stack trace points to {file} but the actual bug is elsewhere. Find and fix it.',
            required_skills=['error analysis', 'root cause identification', 'debugging'],
            weakness_targeted='wrong_files',
            example_tasks=['Fix misleading exception', 'Trace null pointer source']
        ),
        TaskTemplate(
            template_id='debugging_silent',
            name='Silent Failure Detection',
            category='debugging',
            difficulty='hard',
            description_template='The {function} sometimes returns incorrect results for edge cases. No error is raised. Add proper handling for {edge_case}.',
            required_skills=['edge case reasoning', 'test analysis', 'defensive coding'],
            weakness_targeted='no_verification',
            example_tasks=['Handle empty input', 'Fix floating point edge case']
        ),
        
        # Verification challenges
        TaskTemplate(
            template_id='verification_tdd',
            name='Test-Driven Fix',
            category='verification',
            difficulty='medium',
            description_template='A failing test in {test_file} reveals a bug. Fix the implementation to make the test pass without modifying the test.',
            required_skills=['test interpretation', 'spec compliance', 'verification'],
            weakness_targeted='no_verification',
            example_tasks=['Fix assertion failure', 'Match expected output']
        ),
        
        # Efficiency challenges
        TaskTemplate(
            template_id='efficiency_timeout',
            name='Step-Limited Fix',
            category='efficiency',
            difficulty='hard',
            description_template='Fix the bug in {module} using at most {max_steps} tool calls. The solution is straightforward if you find the right file quickly.',
            required_skills=['efficient search', 'direct action', 'minimal exploration'],
            weakness_targeted='hit_max_steps',
            example_tasks=['Quick config fix', 'One-line patch']
        ),
    ]
    
    return templates


def generate_tasks_for_model(model: str, weakness_profile: dict, 
                            templates: list[TaskTemplate], n_tasks: int = 5) -> list[TaskSpec]:
    """Generate tasks specifically targeting a model's weaknesses."""
    
    tasks = []
    
    # Get model's top weaknesses
    model_weaknesses = weakness_profile.get('weaknesses', [])
    
    if not model_weaknesses:
        # Model has no clear weaknesses, generate general difficulty progression
        model_weaknesses = [
            {'pattern': 'general', 'rate': 1.0, 'description': 'General capability test'}
        ]
    
    task_counter = 1
    
    for weakness in model_weaknesses[:3]:  # Top 3 weaknesses
        # Find templates targeting this weakness
        matching_templates = [t for t in templates 
                            if t.weakness_targeted == weakness['pattern']]
        
        if not matching_templates:
            # Use a general template
            matching_templates = [t for t in templates if t.difficulty == 'medium']
        
        for template in matching_templates[:2]:  # 2 tasks per weakness
            task = TaskSpec(
                task_id=f"gen_{model.split('-')[0]}_{task_counter:03d}",
                title=f"{template.name} for {model.split('-')[0]}",
                description=template.description_template,
                difficulty=template.difficulty,
                target_weakness=weakness['pattern'],
                target_models=[model],
                required_skills=template.required_skills,
                estimated_steps=8 if template.difficulty == 'medium' else 12,
                file_patterns=['*.py'],
                success_criteria='All tests pass, no regressions',
                rationale=f"Targets {model}'s weakness: {weakness['description']} (occurs in {weakness['rate']:.0%} of failures)"
            )
            tasks.append(task)
            task_counter += 1
            
            if len(tasks) >= n_tasks:
                break
        
        if len(tasks) >= n_tasks:
            break
    
    return tasks


def generate_difficulty_ladder(templates: list[TaskTemplate]) -> list[TaskSpec]:
    """Generate a difficulty progression for curriculum learning."""
    
    ladder = []
    
    difficulties = ['easy', 'medium', 'hard']
    
    for i, diff in enumerate(difficulties):
        matching = [t for t in templates if t.difficulty == diff]
        
        for j, template in enumerate(matching[:2]):
            task = TaskSpec(
                task_id=f"ladder_{diff}_{j+1:02d}",
                title=f"Level {i+1}.{j+1}: {template.name}",
                description=template.description_template,
                difficulty=diff,
                target_weakness=template.weakness_targeted,
                target_models=['all'],
                required_skills=template.required_skills,
                estimated_steps=5 + i*3,
                file_patterns=['*.py'],
                success_criteria='All tests pass',
                rationale=f"Difficulty ladder level {i+1}: {diff} challenge"
            )
            ladder.append(task)
    
    return ladder


def plot_weakness_analysis(weaknesses: dict, output_path: Path) -> Path:
    """Visualize model weaknesses."""
    
    models = list(weaknesses.keys())
    
    # Extract weakness patterns
    all_patterns = set()
    for profile in weaknesses.values():
        for w in profile.get('weaknesses', []):
            all_patterns.add(w['pattern'])
    
    all_patterns = sorted(all_patterns)
    
    if not all_patterns:
        # No weaknesses found, create placeholder
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No significant weakness patterns detected',
               ha='center', va='center', fontsize=14)
        plot_path = output_path / 'weakness_analysis.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        return plot_path
    
    # Create heatmap data
    data = np.zeros((len(models), len(all_patterns)))
    
    for i, model in enumerate(models):
        for w in weaknesses[model].get('weaknesses', []):
            if w['pattern'] in all_patterns:
                j = all_patterns.index(w['pattern'])
                data[i, j] = w['rate']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(data, cmap='Reds', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(all_patterns)))
    ax.set_xticklabels([p.replace('_', '\n') for p in all_patterns], fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m[:25] for m in models], fontsize=10)
    
    ax.set_xlabel('Weakness Pattern', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    ax.set_title('Model Weakness Heatmap (rate in failures)', fontsize=14)
    
    plt.colorbar(im, ax=ax, label='Occurrence Rate')
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(all_patterns)):
            if data[i, j] > 0:
                ax.text(j, i, f'{data[i, j]:.0%}', ha='center', va='center',
                       color='white' if data[i, j] > 0.5 else 'black', fontsize=9)
    
    plt.tight_layout()
    
    plot_path = output_path / 'weakness_analysis.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def plot_task_coverage(tasks: list[TaskSpec], output_path: Path) -> Path:
    """Visualize what the generated tasks cover."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # By difficulty
    ax1 = axes[0]
    difficulties = [t.difficulty for t in tasks]
    diff_counts = pd.Series(difficulties).value_counts()
    colors = {'easy': '#2ecc71', 'medium': '#f39c12', 'hard': '#e74c3c'}
    ax1.bar(diff_counts.index, diff_counts.values, 
           color=[colors.get(d, '#3498db') for d in diff_counts.index],
           edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Difficulty', fontsize=12)
    ax1.set_ylabel('Number of Tasks', fontsize=12)
    ax1.set_title('Tasks by Difficulty', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # By weakness targeted
    ax2 = axes[1]
    weaknesses = [t.target_weakness for t in tasks]
    weak_counts = pd.Series(weaknesses).value_counts()
    ax2.barh(range(len(weak_counts)), weak_counts.values, color='#3498db', 
            edgecolor='black', alpha=0.7)
    ax2.set_yticks(range(len(weak_counts)))
    ax2.set_yticklabels([w.replace('_', ' ')[:20] for w in weak_counts.index], fontsize=9)
    ax2.set_xlabel('Number of Tasks', fontsize=12)
    ax2.set_title('Tasks by Target Weakness', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # By required skills
    ax3 = axes[2]
    all_skills = []
    for t in tasks:
        all_skills.extend(t.required_skills)
    skill_counts = pd.Series(all_skills).value_counts().head(8)
    ax3.barh(range(len(skill_counts)), skill_counts.values, color='#9b59b6',
            edgecolor='black', alpha=0.7)
    ax3.set_yticks(range(len(skill_counts)))
    ax3.set_yticklabels(skill_counts.index, fontsize=9)
    ax3.set_xlabel('Occurrences', fontsize=12)
    ax3.set_title('Most Required Skills', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle(f'Generated Task Coverage ({len(tasks)} tasks)', fontsize=14, y=1.02)
    plt.tight_layout()
    
    plot_path = output_path / 'task_coverage.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def generate_report(weaknesses: dict, tasks: list[TaskSpec], 
                   templates: list[TaskTemplate], output_path: Path) -> Path:
    """Generate markdown report."""
    
    lines = [
        "# Task Generator Report",
        "",
        "## Overview",
        "",
        "This report describes automatically generated task specifications",
        "designed to target specific model weaknesses identified through analysis.",
        "",
        f"**Generated**: {len(tasks)} tasks from {len(templates)} templates",
        "",
        "## Model Weakness Analysis",
        "",
    ]
    
    for model, profile in weaknesses.items():
        lines.append(f"### {model}")
        lines.append("")
        lines.append(f"- Success rate: {profile['success_rate']:.1%}")
        lines.append(f"- Successes: {profile['n_success']}, Failures: {profile['n_failure']}")
        
        if profile['weaknesses']:
            lines.append("")
            lines.append("**Identified Weaknesses:**")
            for w in profile['weaknesses']:
                lines.append(f"- {w['pattern']}: {w['rate']:.0%} of failures ({w['description']})")
        else:
            lines.append("")
            lines.append("*No significant weakness patterns detected*")
        
        lines.append("")
    
    lines.extend([
        "## Task Templates",
        "",
        "| Template | Category | Difficulty | Targets |",
        "|----------|----------|------------|---------|",
    ])
    
    for t in templates:
        lines.append(f"| {t.name} | {t.category} | {t.difficulty} | {t.weakness_targeted} |")
    
    lines.extend([
        "",
        "## Generated Tasks",
        "",
        "### By Target Model",
        "",
    ])
    
    # Group tasks by target
    by_model = defaultdict(list)
    for task in tasks:
        for model in task.target_models:
            by_model[model].append(task)
    
    for model, model_tasks in by_model.items():
        lines.append(f"#### {model}")
        lines.append("")
        lines.append("| Task ID | Title | Difficulty | Weakness |")
        lines.append("|---------|-------|------------|----------|")
        for t in model_tasks[:5]:
            lines.append(f"| {t.task_id} | {t.title[:30]} | {t.difficulty} | {t.target_weakness} |")
        lines.append("")
    
    lines.extend([
        "## Sample Task Specifications",
        "",
    ])
    
    for task in tasks[:3]:
        lines.extend([
            f"### {task.task_id}: {task.title}",
            "",
            f"**Difficulty**: {task.difficulty}",
            f"**Target Weakness**: {task.target_weakness}",
            f"**Required Skills**: {', '.join(task.required_skills)}",
            f"**Estimated Steps**: {task.estimated_steps}",
            "",
            f"**Description**: {task.description}",
            "",
            f"**Rationale**: {task.rationale}",
            "",
        ])
    
    lines.extend([
        "## Usage",
        "",
        "1. Select tasks based on model weaknesses to create targeted training data",
        "2. Use difficulty ladder for curriculum learning",
        "3. Templates can be instantiated with specific repos/files",
        "",
        "## Files Generated",
        "",
        "- `generated_tasks.json`: All task specifications",
        "- `task_templates.json`: Reusable templates",
        "- `weakness_analysis.png`: Model weakness heatmap",
        "- `task_coverage.png`: Task distribution analysis",
        "",
    ])
    
    report_path = output_path / 'task_generator_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    
    return report_path


def main():
    """Run task generator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Task Generator')
    parser.add_argument('--data', type=Path, default=Path('training_data/real_training_data.jsonl'),
                       help='Path to training data JSONL')
    parser.add_argument('--output', type=Path, default=Path('analysis/outputs/task_generator'),
                       help='Output directory')
    parser.add_argument('--tasks-per-model', type=int, default=5,
                       help='Number of tasks to generate per model')
    
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("TASK GENERATOR")
    print("=" * 60)
    print()
    
    # Load data
    print(f"Loading data from {args.data}...")
    df = load_training_data(args.data)
    print(f"Loaded {len(df)} samples from {df['model'].nunique()} models")
    print()
    
    # Analyze weaknesses
    print("Analyzing model weaknesses...")
    weaknesses = analyze_model_weaknesses(df)
    
    for model, profile in weaknesses.items():
        print(f"\n{model}:")
        print(f"  Success rate: {profile['success_rate']:.1%}")
        if profile['weaknesses']:
            for w in profile['weaknesses'][:2]:
                print(f"  Weakness: {w['pattern']} ({w['rate']:.0%})")
    
    # Generate templates
    print("\n\nGenerating task templates...")
    templates = generate_task_templates()
    print(f"Created {len(templates)} templates")
    
    # Save templates
    templates_data = [asdict(t) for t in templates]
    with open(args.output / 'task_templates.json', 'w') as f:
        json.dump(templates_data, f, indent=2)
    
    # Generate tasks for each model
    print("\nGenerating targeted tasks...")
    all_tasks = []
    
    for model in df['model'].unique():
        model_tasks = generate_tasks_for_model(
            model, weaknesses[model], templates, args.tasks_per_model
        )
        all_tasks.extend(model_tasks)
        print(f"  {model}: {len(model_tasks)} tasks")
    
    # Add difficulty ladder
    ladder_tasks = generate_difficulty_ladder(templates)
    all_tasks.extend(ladder_tasks)
    print(f"  Difficulty ladder: {len(ladder_tasks)} tasks")
    
    # Save tasks
    tasks_data = [asdict(t) for t in all_tasks]
    with open(args.output / 'generated_tasks.json', 'w') as f:
        json.dump(tasks_data, f, indent=2)
    
    print(f"\nTotal tasks generated: {len(all_tasks)}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot1 = plot_weakness_analysis(weaknesses, args.output)
    print(f"  Weakness analysis: {plot1}")
    
    plot2 = plot_task_coverage(all_tasks, args.output)
    print(f"  Task coverage: {plot2}")
    
    # Generate report
    report_path = generate_report(weaknesses, all_tasks, templates, args.output)
    print(f"\nReport saved to {report_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nTemplates: {len(templates)}")
    print(f"Tasks generated: {len(all_tasks)}")
    print(f"\nBy difficulty:")
    for diff in ['easy', 'medium', 'hard']:
        count = sum(1 for t in all_tasks if t.difficulty == diff)
        print(f"  {diff}: {count}")
    
    print(f"\nBy target weakness:")
    weakness_counts = defaultdict(int)
    for t in all_tasks:
        weakness_counts[t.target_weakness] += 1
    for w, c in sorted(weakness_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"  {w}: {c}")
    
    print("\n" + "=" * 60)
    print("✓ Task generation complete!")


if __name__ == '__main__':
    main()