"""Metrics for debugging and evaluating agent output quality.

This module provides comprehensive metrics at multiple levels:
- Token level: Usage, cost, efficiency
- Reasoning level: Quality of agent's thinking
- Tool level: How tools are used
- Sequence level: Phases, trajectory, convergence
- Patch level: Quality of the final output
- Failure analysis: Why things went wrong
"""

import re
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path
from typing import Optional


# =============================================================================
# ENUMS
# =============================================================================

class Phase(Enum):
    """Phases of agent execution."""
    EXPLORATION = "exploration"      # Reading, listing, searching
    UNDERSTANDING = "understanding"  # Re-reading, analyzing
    IMPLEMENTATION = "implementation"  # Writing, editing
    VERIFICATION = "verification"    # Testing, checking
    SUBMISSION = "submission"        # Final submission


class FailureMode(Enum):
    """Classification of failure modes."""
    # Exploration failures
    MISSED_RELEVANT_FILE = "missed_relevant_file"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    EXCESSIVE_EXPLORATION = "excessive_exploration"
    
    # Understanding failures
    MISUNDERSTOOD_ISSUE = "misunderstood_issue"
    WRONG_ROOT_CAUSE = "wrong_root_cause"
    
    # Implementation failures
    WRONG_FIX_LOCATION = "wrong_fix_location"
    INCOMPLETE_FIX = "incomplete_fix"
    INTRODUCED_BUG = "introduced_bug"
    OVERWROTE_FILE = "overwrote_file"
    
    # Process failures
    GAVE_UP_EARLY = "gave_up_early"
    STUCK_IN_LOOP = "stuck_in_loop"
    EXCEEDED_BUDGET = "exceeded_budget"
    NO_SUBMISSION = "no_submission"
    
    # Tool failures
    TOOL_MISUSE = "tool_misuse"
    SYNTAX_ERROR = "syntax_error"


# =============================================================================
# TOKEN-LEVEL METRICS
# =============================================================================

@dataclass
class TokenMetrics:
    """Token usage and cost metrics."""
    
    # Totals
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    
    # Per-step breakdown
    tokens_per_step: list = field(default_factory=list)  # [{step, input, output}]
    
    # Cost estimation (USD)
    estimated_cost_usd: float = 0.0
    cost_breakdown: dict = field(default_factory=dict)  # {input_cost, output_cost}
    
    # Efficiency metrics
    tokens_per_tool_call: float = 0.0
    tokens_per_file_read: float = 0.0
    tokens_per_edit: float = 0.0
    
    # Context usage
    max_context_used: int = 0
    context_growth_rate: float = 0.0  # How fast context grows per step
    
    # Waste analysis
    repeated_content_tokens: int = 0  # Tokens spent re-reading same content
    

# =============================================================================
# REASONING-LEVEL METRICS
# =============================================================================

@dataclass
class ReasoningMetrics:
    """Quality of agent's reasoning and explanations."""
    
    # Presence of reasoning
    has_explicit_reasoning: bool = False
    reasoning_segments: list = field(default_factory=list)  # Extracted reasoning text
    total_reasoning_chars: int = 0
    
    # Quality signals
    mentions_issue_keywords: bool = False
    issue_keyword_matches: list = field(default_factory=list)
    mentions_relevant_files: bool = False
    
    # Reasoning patterns
    hypothesizes_before_acting: bool = False  # "I think the bug is..."
    explains_changes: bool = False             # "I'm changing X because..."
    verifies_after_change: bool = False        # "Let me test to confirm..."
    considers_alternatives: bool = False       # "Another approach would be..."
    
    # Reasoning quality score (0-1)
    reasoning_quality_score: float = 0.0
    
    # Problem understanding
    correctly_identified_issue: bool = False
    correctly_identified_location: bool = False
    
    # Explanation at submission
    submission_explanation: str = ""
    explanation_quality_score: float = 0.0


@dataclass
class ToolArgumentMetrics:
    """Quality of tool arguments and usage patterns."""
    
    # Search quality
    search_queries: list = field(default_factory=list)
    unique_search_queries: int = 0
    repeated_searches: int = 0
    search_specificity_score: float = 0.0  # How targeted are searches?
    search_success_rate: float = 0.0       # Found useful results?
    failed_searches: list = field(default_factory=list)
    
    # Read patterns
    files_read: list = field(default_factory=list)
    unique_files_read: int = 0
    repeated_reads: int = 0
    read_relevance_score: float = 0.0  # % of reads that were relevant
    
    # Edit quality
    edits_attempted: int = 0
    edits_successful: int = 0
    edit_precision_score: float = 0.0  # Minimal changes vs rewrites
    
    # Test selection
    tests_run: list = field(default_factory=list)
    ran_relevant_tests: bool = False
    test_coverage: float = 0.0  # % of fail_to_pass tests run


# =============================================================================
# SEQUENCE-LEVEL METRICS
# =============================================================================

@dataclass
class PhaseMetrics:
    """Analysis of execution phases."""
    
    # Phase sequence
    phase_sequence: list = field(default_factory=list)  # [Phase.EXPLORATION, ...]
    phase_labels: list = field(default_factory=list)    # ["exploration", ...]
    
    # Steps per phase
    exploration_steps: int = 0
    understanding_steps: int = 0
    implementation_steps: int = 0
    verification_steps: int = 0
    submission_steps: int = 0
    
    # Phase percentages
    exploration_pct: float = 0.0
    implementation_pct: float = 0.0
    verification_pct: float = 0.0
    
    # Transitions
    phase_transitions: int = 0
    transition_sequence: list = field(default_factory=list)  # [(from, to), ...]
    
    # Quality patterns
    followed_read_before_write: bool = False
    followed_test_after_change: bool = False
    has_verification_phase: bool = False
    
    # Phase timing (if available)
    time_per_phase: dict = field(default_factory=dict)


@dataclass
class ExplorationMetrics:
    """How the agent explored the codebase."""
    
    # Coverage
    files_explored: int = 0
    directories_explored: int = 0
    total_lines_read: int = 0
    
    # Strategy classification
    exploration_strategy: str = ""  # "breadth_first", "depth_first", "targeted", "random"
    
    # Efficiency
    relevant_file_discovery_step: int = -1  # When did it find the bug location?
    wasted_explorations: int = 0            # Files read but not relevant
    exploration_efficiency: float = 0.0      # relevant / total explorations
    
    # Search patterns
    search_to_read_ratio: float = 0.0
    search_refinement_count: int = 0  # Did it refine searches?
    
    # Path through codebase
    exploration_path: list = field(default_factory=list)  # [(action, target), ...]


@dataclass
class TrajectoryMetrics:
    """Comparison of agent trajectory to optimal path."""
    
    # Agent's trajectory
    agent_trajectory: list = field(default_factory=list)  # ["read file.py", "search X", ...]
    trajectory_length: int = 0
    
    # Optimal trajectory (derived from gold patch)
    optimal_trajectory: list = field(default_factory=list)
    optimal_length: int = 0
    
    # Comparison
    trajectory_similarity: float = 0.0  # 0-1 similarity score
    unnecessary_steps: list = field(default_factory=list)  # Steps not needed
    missing_steps: list = field(default_factory=list)      # Steps that should have been taken
    
    # Efficiency
    trajectory_efficiency: float = 0.0  # optimal_length / actual_length
    
    # Decision points
    decision_points: list = field(default_factory=list)  # Key moments analyzed
    good_decisions: int = 0
    suboptimal_decisions: int = 0
    bad_decisions: int = 0


@dataclass
class ConvergenceMetrics:
    """Did the agent make progress toward a solution?"""
    
    # Progress tracking
    progress_curve: list = field(default_factory=list)  # Similarity at each step
    
    # Convergence
    converged: bool = False
    convergence_step: int = -1  # When did it stabilize?
    final_similarity: float = 0.0
    
    # Progress quality
    monotonic_progress: bool = False  # Always improving?
    max_progress: float = 0.0
    progress_volatility: float = 0.0  # Variance in progress
    
    # Regression detection
    had_regression: bool = False  # Did it get worse at any point?
    regression_steps: list = field(default_factory=list)
    
    # Oscillation
    oscillation_count: int = 0  # Back and forth changes


@dataclass
class ErrorRecoveryMetrics:
    """How the agent handled errors and recovered."""
    
    # Tool errors
    tool_errors: list = field(default_factory=list)  # [{step, tool, error, recovered}]
    total_errors: int = 0
    recovered_errors: int = 0
    recovery_rate: float = 0.0
    
    # Error types
    error_types: dict = field(default_factory=dict)  # {type: count}
    
    # Backtracking
    backtrack_count: int = 0
    backtrack_steps: list = field(default_factory=list)
    
    # Repeated actions (potential stuck)
    repeated_actions: list = field(default_factory=list)  # [(action, count)]
    max_repetition: int = 0
    
    # Stuck detection
    stuck_episodes: list = field(default_factory=list)  # [{start_step, duration}]
    total_stuck_steps: int = 0
    max_stuck_duration: int = 0


# =============================================================================
# PATCH-LEVEL METRICS (existing, enhanced)
# =============================================================================

@dataclass
class ToolUsageMetrics:
    """Metrics about how the agent used tools."""
    
    total_calls: int = 0
    calls_by_tool: dict = field(default_factory=dict)
    
    # Key behaviors
    read_relevant_files: bool = False
    used_str_replace: bool = False
    used_write_file: bool = False
    ran_tests: bool = False
    submitted: bool = False
    
    # Errors
    tool_errors: list = field(default_factory=list)
    
    # Sequence
    tool_sequence: list = field(default_factory=list)
    
    # Detailed call log
    tool_calls: list = field(default_factory=list)  # [{step, tool, args, result}]


@dataclass
class PatchQualityMetrics:
    """Metrics about the quality of the generated patch."""
    
    # Basic stats
    lines_added: int = 0
    lines_removed: int = 0
    files_changed: list = field(default_factory=list)
    
    # Comparison to gold
    gold_files_touched: list = field(default_factory=list)
    correct_files_touched: bool = False
    extra_files_touched: list = field(default_factory=list)
    missing_files: list = field(default_factory=list)
    
    # Content similarity
    similarity_score: float = 0.0
    line_level_similarity: float = 0.0
    
    # Size analysis
    patch_too_large: bool = False
    gold_patch_size: int = 0
    agent_patch_size: int = 0
    size_ratio: float = 0.0
    
    # Semantic analysis
    same_fix_location: bool = False
    same_fix_type: bool = False  # Same kind of change (rename, add, delete, modify)


@dataclass
class FailureAnalysis:
    """Comprehensive analysis of why a task might have failed."""
    
    # Completion
    hit_max_steps: bool = False
    agent_submitted: bool = False
    
    # Classified failure modes
    failure_modes: list = field(default_factory=list)  # [FailureMode, ...]
    primary_failure_mode: str = ""
    
    # Specific failures
    no_changes_made: bool = False
    wrong_files_modified: bool = False
    patch_too_large: bool = False
    tool_errors_occurred: bool = False
    model_got_stuck: bool = False
    
    # Details
    failure_reasons: list = field(default_factory=list)
    
    # Counterfactual analysis
    what_went_wrong: str = ""
    suggested_fix: str = ""


# =============================================================================
# COMPLETE METRICS CONTAINER
# =============================================================================

@dataclass
class DebugMetrics:
    """Complete debug metrics for a task."""
    
    task_id: str
    
    # Core metrics (existing)
    tool_usage: ToolUsageMetrics = field(default_factory=ToolUsageMetrics)
    patch_quality: PatchQualityMetrics = field(default_factory=PatchQualityMetrics)
    failure_analysis: FailureAnalysis = field(default_factory=FailureAnalysis)
    
    # New detailed metrics
    token_metrics: TokenMetrics = field(default_factory=TokenMetrics)
    reasoning_metrics: ReasoningMetrics = field(default_factory=ReasoningMetrics)
    tool_argument_metrics: ToolArgumentMetrics = field(default_factory=ToolArgumentMetrics)
    phase_metrics: PhaseMetrics = field(default_factory=PhaseMetrics)
    exploration_metrics: ExplorationMetrics = field(default_factory=ExplorationMetrics)
    trajectory_metrics: TrajectoryMetrics = field(default_factory=TrajectoryMetrics)
    convergence_metrics: ConvergenceMetrics = field(default_factory=ConvergenceMetrics)
    error_recovery_metrics: ErrorRecoveryMetrics = field(default_factory=ErrorRecoveryMetrics)
    
    # Test results
    tests_passed: int = 0
    tests_failed: int = 0
    resolved: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def classify_tool_to_phase(tool_name: str) -> Phase:
    """Map a tool name to an execution phase."""
    exploration_tools = {"list_directory", "read_file", "search_code"}
    implementation_tools = {"write_file", "str_replace_in_file"}
    verification_tools = {"run_tests", "run_command"}
    submission_tools = {"submit_patch"}
    
    if tool_name in exploration_tools:
        return Phase.EXPLORATION
    elif tool_name in implementation_tools:
        return Phase.IMPLEMENTATION
    elif tool_name in verification_tools:
        return Phase.VERIFICATION
    elif tool_name in submission_tools:
        return Phase.SUBMISSION
    else:
        return Phase.EXPLORATION  # Default


def extract_reasoning_from_messages(messages: list[dict]) -> list[str]:
    """Extract reasoning/thinking segments from assistant messages."""
    reasoning_segments = []
    
    # Patterns that indicate reasoning
    reasoning_patterns = [
        r"I think\b.*?[.!]",
        r"I believe\b.*?[.!]",
        r"This suggests\b.*?[.!]",
        r"The issue (?:is|seems|appears)\b.*?[.!]",
        r"Looking at\b.*?[.!]",
        r"Based on\b.*?[.!]",
        r"I need to\b.*?[.!]",
        r"Let me\b.*?[.!]",
        r"First,\b.*?[.!]",
        r"The (?:bug|problem|error)\b.*?[.!]",
    ]
    
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        
        content = msg.get("content", [])
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = " ".join(
                block.get("text", "") 
                for block in content 
                if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            continue
        
        for pattern in reasoning_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            reasoning_segments.extend(matches)
    
    return reasoning_segments


def analyze_tool_usage(messages: list[dict], relevant_files: list[str]) -> ToolUsageMetrics:
    """Analyze tool usage from agent message history."""
    metrics = ToolUsageMetrics()
    
    relevant_files_lower = [f.lower() for f in relevant_files]
    step = 0
    
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
            
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        
        step += 1
        
        for block in content:
            if block.get("type") != "tool_use":
                continue
                
            tool_name = block.get("name", "")
            tool_input = block.get("input", {})
            
            metrics.total_calls += 1
            metrics.calls_by_tool[tool_name] = metrics.calls_by_tool.get(tool_name, 0) + 1
            metrics.tool_sequence.append(tool_name)
            
            # Log detailed call
            metrics.tool_calls.append({
                "step": step,
                "tool": tool_name,
                "args": tool_input,
            })
            
            # Track key behaviors
            if tool_name == "read_file":
                path = tool_input.get("path", "").lower()
                if any(rf in path or path in rf for rf in relevant_files_lower):
                    metrics.read_relevant_files = True
                    
            elif tool_name == "str_replace_in_file":
                metrics.used_str_replace = True
                
            elif tool_name == "write_file":
                metrics.used_write_file = True
                
            elif tool_name == "run_tests":
                metrics.ran_tests = True
                
            elif tool_name == "submit_patch":
                metrics.submitted = True
    
    # Analyze tool results for errors
    for msg in messages:
        if msg.get("role") != "user":
            continue
            
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
            
        for block in content:
            if block.get("type") != "tool_result":
                continue
                
            result = block.get("content", "")
            if isinstance(result, str) and result.startswith("Error:"):
                metrics.tool_errors.append(result)
    
    return metrics


def analyze_phases(messages: list[dict]) -> PhaseMetrics:
    """Analyze execution phases from message history."""
    metrics = PhaseMetrics()
    
    current_phase = None
    
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
            
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        
        for block in content:
            if block.get("type") != "tool_use":
                continue
            
            tool_name = block.get("name", "")
            phase = classify_tool_to_phase(tool_name)
            
            metrics.phase_sequence.append(phase)
            metrics.phase_labels.append(phase.value)
            
            # Count steps per phase
            if phase == Phase.EXPLORATION:
                metrics.exploration_steps += 1
            elif phase == Phase.UNDERSTANDING:
                metrics.understanding_steps += 1
            elif phase == Phase.IMPLEMENTATION:
                metrics.implementation_steps += 1
            elif phase == Phase.VERIFICATION:
                metrics.verification_steps += 1
            elif phase == Phase.SUBMISSION:
                metrics.submission_steps += 1
            
            # Track transitions
            if current_phase is not None and phase != current_phase:
                metrics.phase_transitions += 1
                metrics.transition_sequence.append((current_phase.value, phase.value))
            
            current_phase = phase
    
    # Calculate percentages
    total = len(metrics.phase_sequence)
    if total > 0:
        metrics.exploration_pct = metrics.exploration_steps / total
        metrics.implementation_pct = metrics.implementation_steps / total
        metrics.verification_pct = metrics.verification_steps / total
    
    # Check quality patterns
    if metrics.phase_sequence:
        # Did exploration come before implementation?
        first_impl = next(
            (i for i, p in enumerate(metrics.phase_sequence) if p == Phase.IMPLEMENTATION),
            len(metrics.phase_sequence)
        )
        first_explore = next(
            (i for i, p in enumerate(metrics.phase_sequence) if p == Phase.EXPLORATION),
            len(metrics.phase_sequence)
        )
        metrics.followed_read_before_write = first_explore < first_impl
        
        # Did verification come after implementation?
        last_impl = len(metrics.phase_sequence) - 1 - next(
            (i for i, p in enumerate(reversed(metrics.phase_sequence)) if p == Phase.IMPLEMENTATION),
            len(metrics.phase_sequence)
        )
        has_verify = Phase.VERIFICATION in metrics.phase_sequence
        if has_verify:
            first_verify = next(
                (i for i, p in enumerate(metrics.phase_sequence) if p == Phase.VERIFICATION),
                -1
            )
            metrics.followed_test_after_change = first_verify > last_impl
            metrics.has_verification_phase = True
    
    return metrics


def analyze_reasoning(
    messages: list[dict],
    issue_body: str,
    relevant_files: list[str],
) -> ReasoningMetrics:
    """Analyze quality of agent's reasoning."""
    metrics = ReasoningMetrics()
    
    # Extract reasoning segments
    metrics.reasoning_segments = extract_reasoning_from_messages(messages)
    metrics.has_explicit_reasoning = len(metrics.reasoning_segments) > 0
    metrics.total_reasoning_chars = sum(len(s) for s in metrics.reasoning_segments)
    
    # Extract all assistant text
    all_text = ""
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        if isinstance(content, str):
            all_text += content + " "
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    all_text += block.get("text", "") + " "
    
    all_text_lower = all_text.lower()
    
    # Check for issue keywords
    issue_words = set(re.findall(r'\b\w{4,}\b', issue_body.lower()))
    common_words = {"this", "that", "with", "from", "have", "been", "when", "should", "would", "could"}
    issue_keywords = issue_words - common_words
    
    for keyword in list(issue_keywords)[:20]:  # Check top 20
        if keyword in all_text_lower:
            metrics.issue_keyword_matches.append(keyword)
    
    metrics.mentions_issue_keywords = len(metrics.issue_keyword_matches) > 0
    
    # Check for file mentions
    for f in relevant_files:
        if f.lower() in all_text_lower or Path(f).name.lower() in all_text_lower:
            metrics.mentions_relevant_files = True
            break
    
    # Check reasoning patterns
    metrics.hypothesizes_before_acting = any(
        p in all_text_lower for p in ["i think", "i believe", "the issue is", "the bug is", "the problem is"]
    )
    metrics.explains_changes = any(
        p in all_text_lower for p in ["because", "this fixes", "this should", "changing this"]
    )
    metrics.verifies_after_change = any(
        p in all_text_lower for p in ["let me test", "let me verify", "running tests", "to confirm"]
    )
    metrics.considers_alternatives = any(
        p in all_text_lower for p in ["alternatively", "another approach", "could also", "other option"]
    )
    
    # Calculate reasoning quality score
    score = 0
    if metrics.has_explicit_reasoning:
        score += 0.2
    if metrics.mentions_issue_keywords:
        score += 0.2
    if metrics.mentions_relevant_files:
        score += 0.15
    if metrics.hypothesizes_before_acting:
        score += 0.15
    if metrics.explains_changes:
        score += 0.15
    if metrics.verifies_after_change:
        score += 0.15
    
    metrics.reasoning_quality_score = min(score, 1.0)
    
    # Extract submission explanation
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_use" and block.get("name") == "submit_patch":
                    metrics.submission_explanation = block.get("input", {}).get("explanation", "")
                    break
    
    return metrics


def analyze_tool_arguments(
    messages: list[dict],
    relevant_files: list[str],
    fail_to_pass: list[str],
) -> ToolArgumentMetrics:
    """Analyze quality of tool arguments."""
    metrics = ToolArgumentMetrics()
    
    relevant_files_lower = set(f.lower() for f in relevant_files)
    
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
            
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        
        for block in content:
            if block.get("type") != "tool_use":
                continue
            
            tool_name = block.get("name", "")
            tool_input = block.get("input", {})
            
            if tool_name == "search_code":
                query = tool_input.get("pattern", "")
                metrics.search_queries.append(query)
                
            elif tool_name == "read_file":
                path = tool_input.get("path", "")
                metrics.files_read.append(path)
                
            elif tool_name in ("write_file", "str_replace_in_file"):
                metrics.edits_attempted += 1
                
            elif tool_name == "run_tests":
                test_path = tool_input.get("test_path", "")
                metrics.tests_run.append(test_path)
    
    # Analyze search patterns
    metrics.unique_search_queries = len(set(metrics.search_queries))
    metrics.repeated_searches = len(metrics.search_queries) - metrics.unique_search_queries
    
    # Analyze read patterns
    metrics.unique_files_read = len(set(metrics.files_read))
    metrics.repeated_reads = len(metrics.files_read) - metrics.unique_files_read
    
    # Calculate read relevance
    if metrics.files_read:
        relevant_reads = sum(
            1 for f in metrics.files_read
            if any(rf in f.lower() for rf in relevant_files_lower)
        )
        metrics.read_relevance_score = relevant_reads / len(metrics.files_read)
    
    # Check test coverage
    if fail_to_pass:
        for test_run in metrics.tests_run:
            if any(test in test_run for test in fail_to_pass):
                metrics.ran_relevant_tests = True
                break
        
        covered = sum(1 for t in fail_to_pass if any(t in tr for tr in metrics.tests_run))
        metrics.test_coverage = covered / len(fail_to_pass)
    
    return metrics


def analyze_exploration(
    messages: list[dict],
    relevant_files: list[str],
) -> ExplorationMetrics:
    """Analyze exploration patterns."""
    metrics = ExplorationMetrics()
    
    relevant_files_lower = set(f.lower() for f in relevant_files)
    directories_seen = set()
    files_seen = set()
    step = 0
    
    search_count = 0
    read_count = 0
    
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
            
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        
        step += 1
        
        for block in content:
            if block.get("type") != "tool_use":
                continue
            
            tool_name = block.get("name", "")
            tool_input = block.get("input", {})
            
            if tool_name == "list_directory":
                path = tool_input.get("path", "")
                directories_seen.add(path)
                metrics.exploration_path.append(("list", path))
                
            elif tool_name == "read_file":
                path = tool_input.get("path", "")
                files_seen.add(path)
                read_count += 1
                metrics.exploration_path.append(("read", path))
                
                # Check if this is a relevant file
                if metrics.relevant_file_discovery_step == -1:
                    if any(rf in path.lower() for rf in relevant_files_lower):
                        metrics.relevant_file_discovery_step = step
                
            elif tool_name == "search_code":
                search_count += 1
                pattern = tool_input.get("pattern", "")
                metrics.exploration_path.append(("search", pattern))
    
    metrics.files_explored = len(files_seen)
    metrics.directories_explored = len(directories_seen)
    
    # Calculate efficiency
    relevant_reads = sum(1 for f in files_seen if any(rf in f.lower() for rf in relevant_files_lower))
    metrics.wasted_explorations = len(files_seen) - relevant_reads
    if files_seen:
        metrics.exploration_efficiency = relevant_reads / len(files_seen)
    
    # Search to read ratio
    if read_count > 0:
        metrics.search_to_read_ratio = search_count / read_count
    
    # Classify exploration strategy
    if metrics.relevant_file_discovery_step <= 3:
        metrics.exploration_strategy = "targeted"
    elif metrics.directories_explored > metrics.files_explored:
        metrics.exploration_strategy = "breadth_first"
    elif metrics.wasted_explorations > relevant_reads:
        metrics.exploration_strategy = "random"
    else:
        metrics.exploration_strategy = "depth_first"
    
    return metrics


def analyze_trajectory(
    messages: list[dict],
    gold_patch: str,
    relevant_files: list[str],
) -> TrajectoryMetrics:
    """Analyze agent trajectory compared to optimal."""
    metrics = TrajectoryMetrics()
    
    # Build agent trajectory
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
            
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        
        for block in content:
            if block.get("type") != "tool_use":
                continue
            
            tool_name = block.get("name", "")
            tool_input = block.get("input", {})
            
            # Create readable trajectory step
            if tool_name == "read_file":
                step = f"read {tool_input.get('path', '?')}"
            elif tool_name == "search_code":
                step = f"search '{tool_input.get('pattern', '?')[:30]}'"
            elif tool_name == "str_replace_in_file":
                step = f"edit {tool_input.get('path', '?')}"
            elif tool_name == "write_file":
                step = f"write {tool_input.get('path', '?')}"
            elif tool_name == "run_tests":
                step = f"test {tool_input.get('test_path', '?')}"
            elif tool_name == "submit_patch":
                step = "submit"
            else:
                step = f"{tool_name}"
            
            metrics.agent_trajectory.append(step)
    
    metrics.trajectory_length = len(metrics.agent_trajectory)
    
    # Build optimal trajectory from gold patch
    gold_files = extract_files_from_patch(gold_patch)
    
    optimal = []
    for f in relevant_files:
        optimal.append(f"read {f}")
    for f in gold_files:
        optimal.append(f"edit {f}")
    optimal.append("test")
    optimal.append("submit")
    
    metrics.optimal_trajectory = optimal
    metrics.optimal_length = len(optimal)
    
    # Calculate efficiency
    if metrics.trajectory_length > 0:
        metrics.trajectory_efficiency = metrics.optimal_length / metrics.trajectory_length
    
    # Find unnecessary steps (very simplified)
    optimal_set = set(metrics.optimal_trajectory)
    for step in metrics.agent_trajectory:
        action = step.split()[0] if step else ""
        if action in ("search", "list"):
            # These are exploratory, not strictly necessary
            metrics.unnecessary_steps.append(step)
    
    return metrics


def analyze_convergence(
    messages: list[dict],
    gold_patch: str,
) -> ConvergenceMetrics:
    """Analyze if the agent made progress toward the solution."""
    metrics = ConvergenceMetrics()
    
    gold_changes = extract_change_lines(gold_patch)
    
    # Track cumulative changes at each step
    cumulative_changes = ""
    step = 0
    
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
            
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        
        for block in content:
            if block.get("type") != "tool_use":
                continue
            
            tool_name = block.get("name", "")
            tool_input = block.get("input", {})
            
            step += 1
            
            # Track edits
            if tool_name == "str_replace_in_file":
                old_str = tool_input.get("old_str", "")
                new_str = tool_input.get("new_str", "")
                cumulative_changes += f"-{old_str}\n+{new_str}\n"
            elif tool_name == "write_file":
                content_written = tool_input.get("content", "")
                cumulative_changes = f"+{content_written}\n"
            
            # Calculate similarity at this step
            if cumulative_changes:
                similarity = SequenceMatcher(None, cumulative_changes, gold_changes).ratio()
            else:
                similarity = 0.0
            
            metrics.progress_curve.append(similarity)
    
    # Analyze convergence
    if metrics.progress_curve:
        metrics.final_similarity = metrics.progress_curve[-1]
        metrics.max_progress = max(metrics.progress_curve)
        
        # Check for convergence (similarity stopped changing)
        for i in range(len(metrics.progress_curve) - 1, 0, -1):
            if abs(metrics.progress_curve[i] - metrics.progress_curve[i-1]) > 0.01:
                metrics.convergence_step = i + 1
                break
        
        metrics.converged = metrics.final_similarity > 0.5
        
        # Check for monotonic progress
        metrics.monotonic_progress = all(
            metrics.progress_curve[i] >= metrics.progress_curve[i-1]
            for i in range(1, len(metrics.progress_curve))
        )
        
        # Detect regressions
        for i in range(1, len(metrics.progress_curve)):
            if metrics.progress_curve[i] < metrics.progress_curve[i-1] - 0.05:
                metrics.had_regression = True
                metrics.regression_steps.append(i)
        
        # Calculate volatility
        if len(metrics.progress_curve) > 1:
            diffs = [
                abs(metrics.progress_curve[i] - metrics.progress_curve[i-1])
                for i in range(1, len(metrics.progress_curve))
            ]
            metrics.progress_volatility = sum(diffs) / len(diffs)
    
    return metrics


def analyze_error_recovery(messages: list[dict]) -> ErrorRecoveryMetrics:
    """Analyze how the agent handled errors."""
    metrics = ErrorRecoveryMetrics()
    
    step = 0
    last_actions = []  # Track recent actions for repetition detection
    action_counts = {}
    
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "tool_use":
                        step += 1
                        tool_name = block.get("name", "")
                        tool_input = block.get("input", {})
                        
                        # Track action for repetition
                        action_key = f"{tool_name}:{str(tool_input)[:100]}"
                        action_counts[action_key] = action_counts.get(action_key, 0) + 1
                        last_actions.append((step, action_key))
        
        elif msg.get("role") == "user":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "tool_result":
                        result = block.get("content", "")
                        if isinstance(result, str) and "Error" in result:
                            error_record = {
                                "step": step,
                                "error": result[:200],
                                "recovered": False,
                            }
                            
                            # Check if there's a subsequent successful action
                            # (simplified: just check if more actions follow)
                            if i < len(messages) - 2:
                                error_record["recovered"] = True
                            
                            metrics.tool_errors.append(error_record)
                            
                            # Classify error type
                            if "not found" in result.lower():
                                error_type = "not_found"
                            elif "syntax" in result.lower():
                                error_type = "syntax"
                            elif "permission" in result.lower():
                                error_type = "permission"
                            else:
                                error_type = "other"
                            
                            metrics.error_types[error_type] = metrics.error_types.get(error_type, 0) + 1
    
    metrics.total_errors = len(metrics.tool_errors)
    metrics.recovered_errors = sum(1 for e in metrics.tool_errors if e.get("recovered"))
    if metrics.total_errors > 0:
        metrics.recovery_rate = metrics.recovered_errors / metrics.total_errors
    
    # Detect repetitions
    for action, count in action_counts.items():
        if count > 1:
            metrics.repeated_actions.append((action[:50], count))
    
    if metrics.repeated_actions:
        metrics.max_repetition = max(count for _, count in metrics.repeated_actions)
    
    # Detect stuck episodes (same action 3+ times in a row)
    if last_actions:
        streak_start = 0
        streak_action = last_actions[0][1] if last_actions else None
        streak_length = 1
        
        for i in range(1, len(last_actions)):
            if last_actions[i][1] == streak_action:
                streak_length += 1
            else:
                if streak_length >= 3:
                    metrics.stuck_episodes.append({
                        "start_step": last_actions[streak_start][0],
                        "duration": streak_length,
                    })
                    metrics.total_stuck_steps += streak_length
                    metrics.max_stuck_duration = max(metrics.max_stuck_duration, streak_length)
                
                streak_start = i
                streak_action = last_actions[i][1]
                streak_length = 1
        
        # Check final streak
        if streak_length >= 3:
            metrics.stuck_episodes.append({
                "start_step": last_actions[streak_start][0],
                "duration": streak_length,
            })
    
    return metrics


def classify_failure_modes(
    tool_metrics: ToolUsageMetrics,
    patch_metrics: PatchQualityMetrics,
    phase_metrics: PhaseMetrics,
    reasoning_metrics: ReasoningMetrics,
    exploration_metrics: ExplorationMetrics,
    error_recovery_metrics: ErrorRecoveryMetrics,
    resolved: bool,
    max_steps: int,
    actual_steps: int,
) -> FailureAnalysis:
    """Classify failure modes based on all metrics."""
    analysis = FailureAnalysis()
    
    analysis.hit_max_steps = actual_steps >= max_steps
    analysis.agent_submitted = tool_metrics.submitted
    
    if resolved:
        return analysis  # No failure to analyze
    
    # Check each failure mode
    failure_modes = []
    
    # Exploration failures
    if not tool_metrics.read_relevant_files:
        failure_modes.append(FailureMode.MISSED_RELEVANT_FILE)
        analysis.failure_reasons.append("Did not read the relevant files")
    
    if exploration_metrics.exploration_efficiency < 0.3:
        failure_modes.append(FailureMode.EXCESSIVE_EXPLORATION)
        analysis.failure_reasons.append("Too much unfocused exploration")
    
    # Understanding failures
    if not reasoning_metrics.correctly_identified_issue and reasoning_metrics.reasoning_quality_score < 0.3:
        failure_modes.append(FailureMode.MISUNDERSTOOD_ISSUE)
        analysis.failure_reasons.append("May have misunderstood the issue")
    
    # Implementation failures
    if not patch_metrics.files_changed:
        failure_modes.append(FailureMode.GAVE_UP_EARLY)
        analysis.no_changes_made = True
        analysis.failure_reasons.append("No changes made to any files")
    
    if patch_metrics.extra_files_touched:
        failure_modes.append(FailureMode.WRONG_FIX_LOCATION)
        analysis.wrong_files_modified = True
        analysis.failure_reasons.append(f"Modified wrong files: {patch_metrics.extra_files_touched}")
    
    if patch_metrics.patch_too_large:
        failure_modes.append(FailureMode.OVERWROTE_FILE)
        analysis.patch_too_large = True
        analysis.failure_reasons.append("Patch too large (may have rewritten entire file)")
    
    if tool_metrics.used_write_file and not tool_metrics.used_str_replace:
        analysis.failure_reasons.append("Used write_file instead of str_replace_in_file")
    
    # Process failures
    if not tool_metrics.submitted:
        failure_modes.append(FailureMode.NO_SUBMISSION)
        analysis.failure_reasons.append("Did not call submit_patch")
    
    if error_recovery_metrics.max_stuck_duration >= 3:
        failure_modes.append(FailureMode.STUCK_IN_LOOP)
        analysis.model_got_stuck = True
        analysis.failure_reasons.append(f"Got stuck (repeated same action {error_recovery_metrics.max_stuck_duration} times)")
    
    if analysis.hit_max_steps:
        failure_modes.append(FailureMode.EXCEEDED_BUDGET)
        analysis.failure_reasons.append("Exceeded step budget")
    
    # Tool failures
    if error_recovery_metrics.total_errors > 0:
        analysis.tool_errors_occurred = True
        analysis.failure_reasons.append(f"Tool errors: {error_recovery_metrics.total_errors}")
    
    analysis.failure_modes = [fm.value for fm in failure_modes]
    if failure_modes:
        analysis.primary_failure_mode = failure_modes[0].value
    
    return analysis


def analyze_patch_quality(
    agent_patch: str, 
    gold_patch: str, 
    relevant_files: list[str]
) -> PatchQualityMetrics:
    """Analyze the quality of the generated patch compared to gold."""
    metrics = PatchQualityMetrics()
    
    # Parse patches to get files and changes
    agent_files = extract_files_from_patch(agent_patch)
    gold_files = extract_files_from_patch(gold_patch)
    
    metrics.files_changed = agent_files
    metrics.gold_files_touched = gold_files
    
    # Check file overlap
    agent_set = set(agent_files)
    gold_set = set(gold_files)
    relevant_set = set(relevant_files)
    
    metrics.correct_files_touched = bool(agent_set & gold_set)
    metrics.extra_files_touched = list(agent_set - gold_set - relevant_set)
    metrics.missing_files = list(gold_set - agent_set)
    
    # Count lines
    metrics.lines_added = agent_patch.count("\n+") - agent_patch.count("\n+++")
    metrics.lines_removed = agent_patch.count("\n-") - agent_patch.count("\n---")
    
    # Patch sizes
    metrics.agent_patch_size = len(agent_patch)
    metrics.gold_patch_size = len(gold_patch)
    
    # Size ratio
    if gold_patch:
        metrics.size_ratio = metrics.agent_patch_size / max(metrics.gold_patch_size, 1)
        metrics.patch_too_large = metrics.size_ratio > 10 or (
            metrics.gold_patch_size < 500 and metrics.agent_patch_size > 5000
        )
    
    # Similarity score
    if agent_patch and gold_patch:
        agent_changes = extract_change_lines(agent_patch)
        gold_changes = extract_change_lines(gold_patch)
        metrics.similarity_score = SequenceMatcher(None, agent_changes, gold_changes).ratio()
        
        # Line-level similarity
        agent_lines = set(agent_changes.split('\n'))
        gold_lines = set(gold_changes.split('\n'))
        if gold_lines:
            overlap = agent_lines & gold_lines
            metrics.line_level_similarity = len(overlap) / len(gold_lines)
    
    return metrics


def extract_files_from_patch(patch: str) -> list[str]:
    """Extract file paths from a git diff patch."""
    files = []
    for match in re.finditer(r'(?:diff --git a/|^\+\+\+ b/)([^\s]+)', patch, re.MULTILINE):
        path = match.group(1)
        if path not in files:
            files.append(path)
    return files


def extract_change_lines(patch: str) -> str:
    """Extract just the changed lines from a patch (for comparison)."""
    lines = []
    for line in patch.split('\n'):
        if line.startswith('+') and not line.startswith('+++'):
            lines.append(line)
        elif line.startswith('-') and not line.startswith('---'):
            lines.append(line)
    return '\n'.join(lines)


# =============================================================================
# MAIN COMPUTATION FUNCTION
# =============================================================================

def compute_debug_metrics(
    task_id: str,
    messages: list[dict],
    agent_patch: str,
    gold_patch: str,
    relevant_files: list[str],
    resolved: bool,
    max_steps: int,
    actual_steps: int,
    tests_passed: int = 0,
    tests_failed: int = 0,
    issue_body: str = "",
    fail_to_pass: list[str] = None,
    token_usage: dict = None,
) -> DebugMetrics:
    """Compute all debug metrics for a task."""
    
    fail_to_pass = fail_to_pass or []
    token_usage = token_usage or {}
    
    # Core metrics
    tool_metrics = analyze_tool_usage(messages, relevant_files)
    patch_metrics = analyze_patch_quality(agent_patch, gold_patch, relevant_files)
    
    # New detailed metrics
    phase_metrics = analyze_phases(messages)
    reasoning_metrics = analyze_reasoning(messages, issue_body, relevant_files)
    tool_arg_metrics = analyze_tool_arguments(messages, relevant_files, fail_to_pass)
    exploration_metrics = analyze_exploration(messages, relevant_files)
    trajectory_metrics = analyze_trajectory(messages, gold_patch, relevant_files)
    convergence_metrics = analyze_convergence(messages, gold_patch)
    error_recovery_metrics = analyze_error_recovery(messages)
    
    # Failure analysis (uses all metrics)
    failure_analysis = classify_failure_modes(
        tool_metrics=tool_metrics,
        patch_metrics=patch_metrics,
        phase_metrics=phase_metrics,
        reasoning_metrics=reasoning_metrics,
        exploration_metrics=exploration_metrics,
        error_recovery_metrics=error_recovery_metrics,
        resolved=resolved,
        max_steps=max_steps,
        actual_steps=actual_steps,
    )
    
    # Token metrics
    token_metrics = TokenMetrics(
        total_input_tokens=token_usage.get("input_tokens", 0),
        total_output_tokens=token_usage.get("output_tokens", 0),
        total_tokens=token_usage.get("total_tokens", 0),
        estimated_cost_usd=token_usage.get("estimated_cost", 0.0),
    )
    
    if tool_metrics.total_calls > 0:
        token_metrics.tokens_per_tool_call = token_metrics.total_tokens / tool_metrics.total_calls
    
    return DebugMetrics(
        task_id=task_id,
        tool_usage=tool_metrics,
        patch_quality=patch_metrics,
        failure_analysis=failure_analysis,
        token_metrics=token_metrics,
        reasoning_metrics=reasoning_metrics,
        tool_argument_metrics=tool_arg_metrics,
        phase_metrics=phase_metrics,
        exploration_metrics=exploration_metrics,
        trajectory_metrics=trajectory_metrics,
        convergence_metrics=convergence_metrics,
        error_recovery_metrics=error_recovery_metrics,
        tests_passed=tests_passed,
        tests_failed=tests_failed,
        resolved=resolved,
    )


# =============================================================================
# REPORT FORMATTING
# =============================================================================

def format_debug_report(metrics: DebugMetrics, agent_patch: str = "") -> str:
    """Format debug metrics as a readable report."""
    lines = [
        f"{'='*70}",
        f"DEBUG REPORT: {metrics.task_id}",
        f"{'='*70}",
        "",
        "## RESULT",
        f"  Resolved: {metrics.resolved}",
        f"  Tests: {metrics.tests_passed} passed, {metrics.tests_failed} failed",
        "",
    ]
    
    # Token metrics
    if metrics.token_metrics.total_tokens > 0:
        lines.extend([
            "## TOKEN USAGE",
            f"  Total tokens: {metrics.token_metrics.total_tokens:,}",
            f"  Input: {metrics.token_metrics.total_input_tokens:,}",
            f"  Output: {metrics.token_metrics.total_output_tokens:,}",
            f"  Estimated cost: ${metrics.token_metrics.estimated_cost_usd:.4f}",
            f"  Tokens per tool call: {metrics.token_metrics.tokens_per_tool_call:.0f}",
            "",
        ])
    
    # Reasoning metrics
    lines.extend([
        "## REASONING QUALITY",
        f"  Quality score: {metrics.reasoning_metrics.reasoning_quality_score:.1%}",
        f"  Has explicit reasoning: {metrics.reasoning_metrics.has_explicit_reasoning}",
        f"  Mentions issue keywords: {metrics.reasoning_metrics.mentions_issue_keywords}",
        f"  Mentions relevant files: {metrics.reasoning_metrics.mentions_relevant_files}",
        f"  Hypothesizes before acting: {metrics.reasoning_metrics.hypothesizes_before_acting}",
        f"  Explains changes: {metrics.reasoning_metrics.explains_changes}",
        f"  Verifies after change: {metrics.reasoning_metrics.verifies_after_change}",
    ])
    if metrics.reasoning_metrics.issue_keyword_matches:
        lines.append(f"  Keywords found: {metrics.reasoning_metrics.issue_keyword_matches[:5]}")
    lines.append("")
    
    # Phase metrics
    lines.extend([
        "## EXECUTION PHASES",
        f"  Exploration: {metrics.phase_metrics.exploration_steps} steps ({metrics.phase_metrics.exploration_pct:.0%})",
        f"  Implementation: {metrics.phase_metrics.implementation_steps} steps ({metrics.phase_metrics.implementation_pct:.0%})",
        f"  Verification: {metrics.phase_metrics.verification_steps} steps ({metrics.phase_metrics.verification_pct:.0%})",
        f"  Phase transitions: {metrics.phase_metrics.phase_transitions}",
        f"  Read before write: {metrics.phase_metrics.followed_read_before_write}",
        f"  Test after change: {metrics.phase_metrics.followed_test_after_change}",
        "",
    ])
    
    # Exploration metrics
    lines.extend([
        "## EXPLORATION",
        f"  Strategy: {metrics.exploration_metrics.exploration_strategy}",
        f"  Files explored: {metrics.exploration_metrics.files_explored}",
        f"  Directories explored: {metrics.exploration_metrics.directories_explored}",
        f"  Relevant file found at step: {metrics.exploration_metrics.relevant_file_discovery_step}",
        f"  Exploration efficiency: {metrics.exploration_metrics.exploration_efficiency:.1%}",
        f"  Wasted explorations: {metrics.exploration_metrics.wasted_explorations}",
        "",
    ])
    
    # Trajectory metrics
    lines.extend([
        "## TRAJECTORY",
        f"  Agent trajectory length: {metrics.trajectory_metrics.trajectory_length}",
        f"  Optimal length: {metrics.trajectory_metrics.optimal_length}",
        f"  Efficiency: {metrics.trajectory_metrics.trajectory_efficiency:.1%}",
        f"  Unnecessary steps: {len(metrics.trajectory_metrics.unnecessary_steps)}",
    ])
    if metrics.trajectory_metrics.agent_trajectory:
        lines.append(f"  Path: {' → '.join(metrics.trajectory_metrics.agent_trajectory[:8])}")
        if len(metrics.trajectory_metrics.agent_trajectory) > 8:
            lines.append(f"        ... ({len(metrics.trajectory_metrics.agent_trajectory) - 8} more)")
    lines.append("")
    
    # Convergence metrics
    lines.extend([
        "## CONVERGENCE",
        f"  Final similarity: {metrics.convergence_metrics.final_similarity:.1%}",
        f"  Max progress: {metrics.convergence_metrics.max_progress:.1%}",
        f"  Converged: {metrics.convergence_metrics.converged}",
        f"  Monotonic progress: {metrics.convergence_metrics.monotonic_progress}",
        f"  Had regression: {metrics.convergence_metrics.had_regression}",
        f"  Volatility: {metrics.convergence_metrics.progress_volatility:.3f}",
    ])
    if metrics.convergence_metrics.progress_curve:
        curve_str = " ".join(f"{p:.0%}" for p in metrics.convergence_metrics.progress_curve[-5:])
        lines.append(f"  Progress curve (last 5): {curve_str}")
    lines.append("")
    
    # Tool usage
    lines.extend([
        "## TOOL USAGE",
        f"  Total calls: {metrics.tool_usage.total_calls}",
        f"  Calls by tool: {metrics.tool_usage.calls_by_tool}",
        f"  Read relevant files: {metrics.tool_usage.read_relevant_files}",
        f"  Used str_replace: {metrics.tool_usage.used_str_replace}",
        f"  Ran tests: {metrics.tool_usage.ran_tests}",
        f"  Submitted: {metrics.tool_usage.submitted}",
        f"  Tool errors: {len(metrics.tool_usage.tool_errors)}",
    ])
    if metrics.tool_usage.tool_sequence:
        seq = " → ".join(metrics.tool_usage.tool_sequence[:10])
        lines.append(f"  Sequence: {seq}{'...' if len(metrics.tool_usage.tool_sequence) > 10 else ''}")
    lines.append("")
    
    # Error recovery
    if metrics.error_recovery_metrics.total_errors > 0 or metrics.error_recovery_metrics.stuck_episodes:
        lines.extend([
            "## ERROR RECOVERY",
            f"  Total errors: {metrics.error_recovery_metrics.total_errors}",
            f"  Recovered: {metrics.error_recovery_metrics.recovered_errors}",
            f"  Recovery rate: {metrics.error_recovery_metrics.recovery_rate:.1%}",
            f"  Stuck episodes: {len(metrics.error_recovery_metrics.stuck_episodes)}",
            f"  Max stuck duration: {metrics.error_recovery_metrics.max_stuck_duration}",
            "",
        ])
    
    # Patch quality
    lines.extend([
        "## PATCH QUALITY",
        f"  Files changed: {metrics.patch_quality.files_changed}",
        f"  Gold files: {metrics.patch_quality.gold_files_touched}",
        f"  Correct files touched: {metrics.patch_quality.correct_files_touched}",
        f"  Lines: +{metrics.patch_quality.lines_added} -{metrics.patch_quality.lines_removed}",
        f"  Patch size: {metrics.patch_quality.agent_patch_size} chars (gold: {metrics.patch_quality.gold_patch_size})",
        f"  Similarity to gold: {metrics.patch_quality.similarity_score:.1%}",
        f"  Line-level similarity: {metrics.patch_quality.line_level_similarity:.1%}",
        f"  Patch too large: {metrics.patch_quality.patch_too_large}",
    ])
    if metrics.patch_quality.extra_files_touched:
        lines.append(f"  Extra files: {metrics.patch_quality.extra_files_touched}")
    if metrics.patch_quality.missing_files:
        lines.append(f"  Missing files: {metrics.patch_quality.missing_files}")
    lines.append("")
    
    # Show actual patch
    if agent_patch:
        lines.append("## AGENT'S PATCH")
        patch_lines = agent_patch.strip().split('\n')
        if len(patch_lines) <= 25:
            for line in patch_lines:
                lines.append(f"  {line}")
        else:
            for line in patch_lines[:12]:
                lines.append(f"  {line}")
            lines.append(f"  ... ({len(patch_lines) - 24} lines omitted) ...")
            for line in patch_lines[-12:]:
                lines.append(f"  {line}")
        lines.append("")
    
    # Failure analysis
    lines.extend([
        "## FAILURE ANALYSIS",
        f"  Hit max steps: {metrics.failure_analysis.hit_max_steps}",
        f"  Agent submitted: {metrics.failure_analysis.agent_submitted}",
    ])
    if metrics.failure_analysis.primary_failure_mode:
        lines.append(f"  Primary failure mode: {metrics.failure_analysis.primary_failure_mode}")
    if metrics.failure_analysis.failure_modes:
        lines.append(f"  All failure modes: {metrics.failure_analysis.failure_modes}")
    if metrics.failure_analysis.failure_reasons:
        lines.append("  Issues found:")
        for reason in metrics.failure_analysis.failure_reasons:
            lines.append(f"    - {reason}")
    else:
        lines.append("  No obvious issues found")
    
    lines.extend(["", "=" * 70, ""])
    
    return "\n".join(lines)