"""Metrics for debugging and evaluating agent output quality."""

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path


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


@dataclass
class PatchQualityMetrics:
    """Metrics about the quality of the generated patch."""
    
    # Basic stats
    lines_added: int = 0
    lines_removed: int = 0
    files_changed: list = field(default_factory=list)
    
    # Comparison to gold
    gold_files_touched: list = field(default_factory=list)  # files in gold patch
    correct_files_touched: bool = False  # did we touch the right files?
    extra_files_touched: list = field(default_factory=list)  # files we touched but shouldn't have
    missing_files: list = field(default_factory=list)  # files we should have touched but didn't
    
    # Content similarity
    similarity_score: float = 0.0  # 0-1 score vs gold patch
    
    # Size analysis
    patch_too_large: bool = False  # rewrote whole file?
    gold_patch_size: int = 0
    agent_patch_size: int = 0


@dataclass
class FailureAnalysis:
    """Analysis of why a task might have failed."""
    
    # Completion
    hit_max_steps: bool = False
    agent_submitted: bool = False
    
    # Failure modes
    no_changes_made: bool = False
    wrong_files_modified: bool = False
    patch_too_large: bool = False
    tool_errors_occurred: bool = False
    model_got_stuck: bool = False  # text-only responses
    
    # Details
    failure_reasons: list = field(default_factory=list)


@dataclass 
class DebugMetrics:
    """Complete debug metrics for a task."""
    
    task_id: str
    tool_usage: ToolUsageMetrics
    patch_quality: PatchQualityMetrics
    failure_analysis: FailureAnalysis
    
    # Test results
    tests_passed: int = 0
    tests_failed: int = 0
    resolved: bool = False


def analyze_tool_usage(messages: list[dict], relevant_files: list[str]) -> ToolUsageMetrics:
    """Analyze tool usage from agent message history."""
    metrics = ToolUsageMetrics()
    
    relevant_files_lower = [f.lower() for f in relevant_files]
    
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
            
            metrics.total_calls += 1
            metrics.calls_by_tool[tool_name] = metrics.calls_by_tool.get(tool_name, 0) + 1
            metrics.tool_sequence.append(tool_name)
            
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
    
    # Check if patch is too large (more than 10x gold patch or >5000 chars for small fixes)
    if gold_patch:
        size_ratio = metrics.agent_patch_size / max(metrics.gold_patch_size, 1)
        metrics.patch_too_large = size_ratio > 10 or (metrics.gold_patch_size < 500 and metrics.agent_patch_size > 5000)
    
    # Similarity score
    if agent_patch and gold_patch:
        # Compare the actual changes (not headers)
        agent_changes = extract_change_lines(agent_patch)
        gold_changes = extract_change_lines(gold_patch)
        metrics.similarity_score = SequenceMatcher(None, agent_changes, gold_changes).ratio()
    
    return metrics


def analyze_failure(
    tool_metrics: ToolUsageMetrics,
    patch_metrics: PatchQualityMetrics,
    resolved: bool,
    max_steps: int,
    actual_steps: int,
) -> FailureAnalysis:
    """Analyze why a task might have failed."""
    analysis = FailureAnalysis()
    
    analysis.hit_max_steps = actual_steps >= max_steps
    analysis.agent_submitted = tool_metrics.submitted
    
    # Check failure modes
    if not patch_metrics.files_changed:
        analysis.no_changes_made = True
        analysis.failure_reasons.append("No changes made to any files")
    
    if patch_metrics.extra_files_touched:
        analysis.wrong_files_modified = True
        analysis.failure_reasons.append(f"Modified wrong files: {patch_metrics.extra_files_touched}")
    
    if patch_metrics.patch_too_large:
        analysis.patch_too_large = True
        analysis.failure_reasons.append(f"Patch too large ({patch_metrics.agent_patch_size} chars vs {patch_metrics.gold_patch_size} gold)")
    
    if tool_metrics.tool_errors:
        analysis.tool_errors_occurred = True
        analysis.failure_reasons.append(f"Tool errors: {len(tool_metrics.tool_errors)}")
    
    # Check for stuck model (many text responses, few tool calls)
    if tool_metrics.total_calls < 3 and actual_steps > 5:
        analysis.model_got_stuck = True
        analysis.failure_reasons.append("Model may have gotten stuck (few tool calls)")
    
    # Additional insights
    if not tool_metrics.read_relevant_files:
        analysis.failure_reasons.append("Did not read the relevant files")
    
    if tool_metrics.used_write_file and not tool_metrics.used_str_replace:
        analysis.failure_reasons.append("Used write_file instead of str_replace_in_file")
    
    if not tool_metrics.submitted:
        analysis.failure_reasons.append("Did not call submit_patch")
    
    return analysis


def extract_files_from_patch(patch: str) -> list[str]:
    """Extract file paths from a git diff patch."""
    files = []
    # Match "diff --git a/path/to/file b/path/to/file" or "+++ b/path/to/file"
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
) -> DebugMetrics:
    """Compute all debug metrics for a task."""
    
    tool_metrics = analyze_tool_usage(messages, relevant_files)
    patch_metrics = analyze_patch_quality(agent_patch, gold_patch, relevant_files)
    failure_analysis = analyze_failure(
        tool_metrics, patch_metrics, resolved, max_steps, actual_steps
    )
    
    return DebugMetrics(
        task_id=task_id,
        tool_usage=tool_metrics,
        patch_quality=patch_metrics,
        failure_analysis=failure_analysis,
        tests_passed=tests_passed,
        tests_failed=tests_failed,
        resolved=resolved,
    )


def format_debug_report(metrics: DebugMetrics, agent_patch: str = "") -> str:
    """Format debug metrics as a readable report."""
    lines = [
        f"=== Debug Report: {metrics.task_id} ===",
        "",
        "## Result",
        f"  Resolved: {metrics.resolved}",
        f"  Tests: {metrics.tests_passed} passed, {metrics.tests_failed} failed",
        "",
        "## Tool Usage",
        f"  Total calls: {metrics.tool_usage.total_calls}",
        f"  Calls by tool: {metrics.tool_usage.calls_by_tool}",
        f"  Sequence: {' -> '.join(metrics.tool_usage.tool_sequence[:10])}{'...' if len(metrics.tool_usage.tool_sequence) > 10 else ''}",
        f"  Read relevant files: {metrics.tool_usage.read_relevant_files}",
        f"  Used str_replace: {metrics.tool_usage.used_str_replace}",
        f"  Used write_file: {metrics.tool_usage.used_write_file}",
        f"  Ran tests: {metrics.tool_usage.ran_tests}",
        f"  Submitted: {metrics.tool_usage.submitted}",
        f"  Tool errors: {len(metrics.tool_usage.tool_errors)}",
    ]
    
    if metrics.tool_usage.tool_errors:
        lines.append("  Error details:")
        for err in metrics.tool_usage.tool_errors[:3]:
            lines.append(f"    - {err[:100]}...")
    
    lines.extend([
        "",
        "## Patch Quality",
        f"  Files changed: {metrics.patch_quality.files_changed}",
        f"  Gold files: {metrics.patch_quality.gold_files_touched}",
        f"  Correct files touched: {metrics.patch_quality.correct_files_touched}",
        f"  Lines: +{metrics.patch_quality.lines_added} -{metrics.patch_quality.lines_removed}",
        f"  Patch size: {metrics.patch_quality.agent_patch_size} chars (gold: {metrics.patch_quality.gold_patch_size})",
        f"  Similarity to gold: {metrics.patch_quality.similarity_score:.1%}",
        f"  Patch too large: {metrics.patch_quality.patch_too_large}",
    ])
    
    if metrics.patch_quality.extra_files_touched:
        lines.append(f"  Extra files: {metrics.patch_quality.extra_files_touched}")
    if metrics.patch_quality.missing_files:
        lines.append(f"  Missing files: {metrics.patch_quality.missing_files}")
    
    # Show actual patch if available
    if agent_patch:
        lines.extend([
            "",
            "## Agent's Patch",
        ])
        patch_lines = agent_patch.strip().split('\n')
        if len(patch_lines) <= 30:
            for line in patch_lines:
                lines.append(f"  {line}")
        else:
            for line in patch_lines[:15]:
                lines.append(f"  {line}")
            lines.append(f"  ... ({len(patch_lines) - 30} more lines) ...")
            for line in patch_lines[-15:]:
                lines.append(f"  {line}")
    
    lines.extend([
        "",
        "## Failure Analysis",
        f"  Hit max steps: {metrics.failure_analysis.hit_max_steps}",
        f"  Agent submitted: {metrics.failure_analysis.agent_submitted}",
    ])
    
    if metrics.failure_analysis.failure_reasons:
        lines.append("  Issues found:")
        for reason in metrics.failure_analysis.failure_reasons:
            lines.append(f"    - {reason}")
    else:
        lines.append("  No obvious issues found")
    
    lines.append("")
    return "\n".join(lines)