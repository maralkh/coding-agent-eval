# Task Generator Report

## Overview

This report describes automatically generated task specifications
designed to target specific model weaknesses identified through analysis.

**Generated**: 15 tasks from 8 templates

## Model Weakness Analysis

> **Note**: Claude Opus 4 data appears incomplete—0% success rate with 100% 'no_changes' 
> suggests the agent may not have been properly configured or runs were interrupted.
> Results for this model should be interpreted with caution.

### claude-opus-4-20250514

- Success rate: 0.0%
- Successes: 0, Failures: 52

**Identified Weaknesses:**
- no_changes: 100% of failures (Fails to make any code changes, gets stuck in exploration)
- low_exploration_efficiency: 98% of failures (Wastes steps on irrelevant exploration)
- no_verification: 100% of failures (Does not verify changes with tests)

### gpt-4o

- Success rate: 26.9%
- Successes: 14, Failures: 38

**Identified Weaknesses:**
- tool_errors: 95% of failures (Frequent tool usage errors, struggles with API)
- no_verification: 53% of failures (Does not verify changes with tests)

### gpt-5.1

- Success rate: 34.6%
- Successes: 18, Failures: 34

**Identified Weaknesses:**
- tool_errors: 97% of failures (Frequent tool usage errors, struggles with API)

### o4-mini

- Success rate: 11.5%
- Successes: 6, Failures: 46

**Identified Weaknesses:**
- tool_errors: 100% of failures (Frequent tool usage errors, struggles with API)
- no_changes: 63% of failures (Fails to make any code changes, gets stuck in exploration)
- hit_max_steps: 54% of failures (Runs out of steps, inefficient exploration)
- low_exploration_efficiency: 52% of failures (Wastes steps on irrelevant exploration)
- no_verification: 61% of failures (Does not verify changes with tests)

## Task Templates

| Template | Category | Difficulty | Targets |
|----------|----------|------------|---------|
| Deep Codebase Navigation | exploration | hard | low_exploration_efficiency |
| Targeted File Discovery | exploration | medium | wrong_files |
| Precise Edit Challenge | implementation | medium | tool_errors |
| Multi-File Coordination | implementation | hard | no_changes |
| Error Trace Analysis | debugging | medium | wrong_files |
| Silent Failure Detection | debugging | hard | no_verification |
| Test-Driven Fix | verification | medium | no_verification |
| Step-Limited Fix | efficiency | hard | hit_max_steps |

## Generated Tasks

### By Target Model

#### claude-opus-4-20250514

| Task ID | Title | Difficulty | Weakness |
|---------|-------|------------|----------|
| gen_claude_001 | Multi-File Coordination for cl | hard | no_changes |
| gen_claude_002 | Deep Codebase Navigation for c | hard | low_exploration_efficiency |
| gen_claude_003 | Silent Failure Detection for c | hard | no_verification |
| gen_claude_004 | Test-Driven Fix for claude | medium | no_verification |

#### gpt-4o

| Task ID | Title | Difficulty | Weakness |
|---------|-------|------------|----------|
| gen_gpt_001 | Precise Edit Challenge for gpt | medium | tool_errors |
| gen_gpt_002 | Silent Failure Detection for g | hard | no_verification |
| gen_gpt_003 | Test-Driven Fix for gpt | medium | no_verification |

#### gpt-5.1

| Task ID | Title | Difficulty | Weakness |
|---------|-------|------------|----------|
| gen_gpt_001 | Precise Edit Challenge for gpt | medium | tool_errors |

#### o4-mini

| Task ID | Title | Difficulty | Weakness |
|---------|-------|------------|----------|
| gen_o4_001 | Precise Edit Challenge for o4 | medium | tool_errors |
| gen_o4_002 | Multi-File Coordination for o4 | hard | no_changes |
| gen_o4_003 | Step-Limited Fix for o4 | hard | hit_max_steps |

#### all

| Task ID | Title | Difficulty | Weakness |
|---------|-------|------------|----------|
| ladder_medium_01 | Level 2.1: Targeted File Disco | medium | wrong_files |
| ladder_medium_02 | Level 2.2: Precise Edit Challe | medium | tool_errors |
| ladder_hard_01 | Level 3.1: Deep Codebase Navig | hard | low_exploration_efficiency |
| ladder_hard_02 | Level 3.2: Multi-File Coordina | hard | no_changes |

## Sample Task Specifications

### gen_claude_001: Multi-File Coordination for claude

**Difficulty**: hard
**Target Weakness**: no_changes
**Required Skills**: multi-file editing, consistency maintenance, API design
**Estimated Steps**: 12

**Description**: Implement {feature} which requires coordinated changes across {n_files} files: {file_list}. Changes must be consistent.

**Rationale**: Targets claude-opus-4-20250514's weakness: Fails to make any code changes, gets stuck in exploration (occurs in 100% of failures)

### gen_claude_002: Deep Codebase Navigation for claude

**Difficulty**: hard
**Target Weakness**: low_exploration_efficiency
**Required Skills**: code navigation, dependency tracing, systematic search
**Estimated Steps**: 12

**Description**: Fix a bug in {module} that requires understanding the interaction between {component1} and {component2}. The bug manifests as {symptom} but the root cause is in a different file.

**Rationale**: Targets claude-opus-4-20250514's weakness: Wastes steps on irrelevant exploration (occurs in 98% of failures)

### gen_claude_003: Silent Failure Detection for claude

**Difficulty**: hard
**Target Weakness**: no_verification
**Required Skills**: edge case reasoning, test analysis, defensive coding
**Estimated Steps**: 12

**Description**: The {function} sometimes returns incorrect results for edge cases. No error is raised. Add proper handling for {edge_case}.

**Rationale**: Targets claude-opus-4-20250514's weakness: Does not verify changes with tests (occurs in 100% of failures)

## Usage

1. Select tasks based on model weaknesses to create targeted training data
2. Use difficulty ladder for curriculum learning
3. Templates can be instantiated with specific repos/files

## Files Generated

- `generated_tasks.json`: All task specifications
- `task_templates.json`: Reusable templates
- `weakness_analysis.png`: Model weakness heatmap
- `task_coverage.png`: Task distribution analysis