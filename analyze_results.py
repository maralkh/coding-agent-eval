#!/usr/bin/env python
"""
Analyze benchmark results with unified scoring.

Loads results from benchmark output directory and computes various scores
to compare model and task performance.

Usage:
    # Basic analysis
    python analyze_results.py --results results/benchmark/
    
    # Specific scoring method
    python analyze_results.py --results results/benchmark/ --method hierarchical
    
    # Compare all scoring methods
    python analyze_results.py --results results/benchmark/ --compare-methods
    
    # Output to file
    python analyze_results.py --results results/benchmark/ --output analysis.json
    
    # Filter by model
    python analyze_results.py --results results/benchmark/ --model gpt-4o
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from eval.harness.results import (
    BenchmarkAnalyzer,
    load_benchmark_dir,
    BenchmarkTaskResult,
)
from eval.harness.scoring import (
    UnifiedScorer,
    compute_elo_ratings,
    hierarchical_score,
    weighted_score,
    geometric_score,
)


def print_header(title: str, char: str = "=", width: int = 70):
    """Print a formatted header."""
    print(char * width)
    print(f" {title}")
    print(char * width)


def print_table(headers: list, rows: list, col_widths: list = None):
    """Print a formatted table."""
    if col_widths is None:
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 
                      for i in range(len(headers))]
    
    # Header
    header_str = "".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    print(header_str)
    print("-" * sum(col_widths))
    
    # Rows
    for row in rows:
        row_str = "".join(f"{str(v):<{w}}" for v, w in zip(row, col_widths))
        print(row_str)


def analyze_results(
    results_path: str,
    method: str = "hierarchical",
    model_filter: str = None,
    compare_methods: bool = False,
    output_path: str = None,
    verbose: bool = True,
):
    """
    Analyze benchmark results with unified scoring.
    
    Args:
        results_path: Path to results directory or file
        method: Scoring method to use
        model_filter: Filter results to specific model
        compare_methods: Compare all scoring methods
        output_path: Save analysis to JSON file
        verbose: Print detailed output
    """
    # Load results
    if verbose:
        print(f"\nLoading results from: {results_path}")
    
    try:
        analyzer = BenchmarkAnalyzer(results_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    benchmark = analyzer.benchmark
    
    # Basic info
    if verbose:
        print_header("BENCHMARK ANALYSIS")
        print(f"Timestamp: {benchmark.timestamp}")
        print(f"Tasks: {len(benchmark.tasks)}")
        print(f"Models: {len(benchmark.models)}")
        print(f"Total runs: {sum(len(m.task_results) for m in benchmark.models)}")
        print()
    
    # Build analysis results
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "source": str(results_path),
        "config": {
            "method": method,
            "model_filter": model_filter,
        },
        "summary": {
            "total_tasks": len(benchmark.tasks),
            "total_models": len(benchmark.models),
            "total_runs": sum(len(m.task_results) for m in benchmark.models),
        },
        "models": {},
        "tasks": {},
        "scores": {},
    }
    
    # =========================================================================
    # Model Summary
    # =========================================================================
    if verbose:
        print_header("MODEL SUMMARY", "-")
    
    model_rows = []
    for m in sorted(benchmark.models, key=lambda x: -x.resolve_rate):
        if model_filter and m.model != model_filter:
            continue
        
        resolved = f"{m.resolved_count}/{m.total_tasks} ({m.resolve_rate:.0%})"
        
        model_rows.append([
            m.model[:30],
            m.provider,
            resolved,
            f"{m.avg_similarity:.1%}",
            f"{m.avg_steps:.1f}",
            f"{m.avg_trajectory_efficiency:.1%}",
        ])
        
        analysis["models"][m.model] = {
            "provider": m.provider,
            "total_tasks": m.total_tasks,
            "resolved_count": m.resolved_count,
            "resolve_rate": m.resolve_rate,
            "submit_rate": m.submit_rate,
            "avg_steps": m.avg_steps,
            "avg_similarity": m.avg_similarity,
            "avg_reasoning_score": m.avg_reasoning_score,
            "avg_exploration_efficiency": m.avg_exploration_efficiency,
            "avg_trajectory_efficiency": m.avg_trajectory_efficiency,
            "failure_modes": m.failure_modes,
        }
    
    if verbose and model_rows:
        print_table(
            ["Model", "Provider", "Resolved", "Similarity", "Steps", "Traj Eff"],
            model_rows,
            [32, 12, 14, 12, 8, 10],
        )
        print()
    
    # =========================================================================
    # Unified Scores by Model
    # =========================================================================
    if verbose:
        print_header(f"UNIFIED SCORES ({method.upper()})", "-")
    
    # Get all metrics for reference population (needed for some methods)
    all_metrics = analyzer.get_metrics_dicts()
    
    # Compute scores for each model
    model_scores = {}
    for m in benchmark.models:
        if model_filter and m.model != model_filter:
            continue
        
        task_scores = []
        for tr in m.task_results:
            metrics = tr.to_metrics_dict()
            
            # Compute score based on method
            if method in ["percentile", "topsis", "pareto", "pca"]:
                scorer = UnifiedScorer(method=method, reference_population=all_metrics)
            else:
                scorer = UnifiedScorer(method=method)
            
            score = scorer.score(metrics)
            task_scores.append({
                "task_id": tr.task_id,
                "score": score,
                "resolved": tr.resolved,
                "metrics": metrics,
            })
        
        # Aggregate
        scores = [ts["score"] for ts in task_scores]
        model_scores[m.model] = {
            "mean_score": sum(scores) / len(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "task_scores": task_scores,
        }
    
    # Display
    if verbose:
        score_rows = []
        for model, data in sorted(model_scores.items(), key=lambda x: -x[1]["mean_score"]):
            score_rows.append([
                model[:30],
                f"{data['mean_score']:.3f}",
                f"{data['min_score']:.3f}",
                f"{data['max_score']:.3f}",
            ])
        
        print_table(
            ["Model", "Mean Score", "Min", "Max"],
            score_rows,
            [32, 12, 10, 10],
        )
        print()
    
    analysis["scores"][method] = model_scores
    
    # =========================================================================
    # Compare All Scoring Methods
    # =========================================================================
    if compare_methods:
        if verbose:
            print_header("SCORING METHOD COMPARISON", "-")
        
        methods = ["weighted", "geometric", "hierarchical"]
        if len(all_metrics) >= 3:
            methods.extend(["percentile", "topsis", "pareto"])
        
        comparison = {}
        for m in benchmark.models:
            if model_filter and m.model != model_filter:
                continue
            
            comparison[m.model] = {}
            for meth in methods:
                scores = []
                for tr in m.task_results:
                    metrics = tr.to_metrics_dict()
                    if meth in ["percentile", "topsis", "pareto"]:
                        scorer = UnifiedScorer(method=meth, reference_population=all_metrics)
                    else:
                        scorer = UnifiedScorer(method=meth)
                    scores.append(scorer.score(metrics))
                
                comparison[m.model][meth] = sum(scores) / len(scores) if scores else 0
        
        if verbose:
            # Build table
            headers = ["Model"] + [m.capitalize()[:8] for m in methods]
            rows = []
            for model, scores in sorted(comparison.items(), key=lambda x: -x[1].get("hierarchical", 0)):
                row = [model[:25]] + [f"{scores.get(m, 0):.3f}" for m in methods]
                rows.append(row)
            
            col_widths = [27] + [10] * len(methods)
            print_table(headers, rows, col_widths)
            print()
        
        analysis["method_comparison"] = comparison
    
    # =========================================================================
    # Elo Ratings
    # =========================================================================
    if len(benchmark.models) > 1:
        if verbose:
            print_header("ELO RATINGS", "-")
        
        # Build task_results structure for Elo
        task_results = {}
        for m in benchmark.models:
            if model_filter and m.model != model_filter:
                continue
            task_results[m.model] = {}
            for tr in m.task_results:
                task_results[m.model][tr.task_id] = tr.to_metrics_dict()
        
        if len(task_results) > 1:
            elo_ratings = compute_elo_ratings(task_results)
            
            if verbose:
                elo_rows = []
                for model, rating in sorted(elo_ratings.items(), key=lambda x: -x[1]):
                    elo_rows.append([model[:30], f"{rating:.1f}"])
                
                print_table(["Model", "Elo Rating"], elo_rows, [32, 12])
                print()
            
            analysis["elo_ratings"] = elo_ratings
    
    # =========================================================================
    # Per-Task Analysis
    # =========================================================================
    if verbose:
        print_header("PER-TASK ANALYSIS", "-")
    
    # Aggregate by task
    task_data = defaultdict(list)
    for m in benchmark.models:
        if model_filter and m.model != model_filter:
            continue
        for tr in m.task_results:
            task_data[tr.task_id].append({
                "model": m.model,
                "resolved": tr.resolved,
                "score": model_scores.get(m.model, {}).get("task_scores", [{}])[0].get("score", 0),
                "steps": tr.steps,
            })
    
    # Find easiest/hardest tasks
    task_resolve_rates = {}
    for task_id, runs in task_data.items():
        resolved = sum(1 for r in runs if r["resolved"])
        task_resolve_rates[task_id] = resolved / len(runs) if runs else 0
    
    if verbose and task_resolve_rates:
        # Easiest tasks
        print("\nEasiest Tasks (highest resolve rate):")
        for task_id, rate in sorted(task_resolve_rates.items(), key=lambda x: -x[1])[:5]:
            print(f"  {task_id}: {rate:.0%}")
        
        # Hardest tasks
        print("\nHardest Tasks (lowest resolve rate):")
        for task_id, rate in sorted(task_resolve_rates.items(), key=lambda x: x[1])[:5]:
            print(f"  {task_id}: {rate:.0%}")
        print()
    
    analysis["tasks"] = {
        task_id: {
            "resolve_rate": rate,
            "runs": task_data[task_id],
        }
        for task_id, rate in task_resolve_rates.items()
    }
    
    # =========================================================================
    # Failure Analysis
    # =========================================================================
    if verbose:
        print_header("FAILURE ANALYSIS", "-")
    
    failure_counts = analyzer.failure_analysis(model_filter)
    
    if verbose and failure_counts:
        total_failures = sum(failure_counts.values())
        failure_rows = []
        for mode, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
            pct = count / total_failures * 100 if total_failures > 0 else 0
            failure_rows.append([mode[:35], str(count), f"{pct:.1f}%"])
        
        print_table(["Failure Mode", "Count", "Percentage"], failure_rows, [37, 8, 12])
        print()
    
    analysis["failure_analysis"] = failure_counts
    
    # =========================================================================
    # Score Distribution
    # =========================================================================
    if verbose:
        print_header("SCORE DISTRIBUTION", "-")
        
        all_scores = []
        for model_data in model_scores.values():
            all_scores.extend([ts["score"] for ts in model_data["task_scores"]])
        
        if all_scores:
            import numpy as np
            scores_arr = np.array(all_scores)
            
            print(f"Total runs scored: {len(all_scores)}")
            print(f"Mean score: {np.mean(scores_arr):.3f}")
            print(f"Std dev: {np.std(scores_arr):.3f}")
            print(f"Min: {np.min(scores_arr):.3f}")
            print(f"Max: {np.max(scores_arr):.3f}")
            print(f"Median: {np.median(scores_arr):.3f}")
            
            # Quartiles
            q1, q2, q3 = np.percentile(scores_arr, [25, 50, 75])
            print(f"Quartiles: Q1={q1:.3f}, Q2={q2:.3f}, Q3={q3:.3f}")
            print()
            
            analysis["score_distribution"] = {
                "mean": float(np.mean(scores_arr)),
                "std": float(np.std(scores_arr)),
                "min": float(np.min(scores_arr)),
                "max": float(np.max(scores_arr)),
                "median": float(np.median(scores_arr)),
                "q1": float(q1),
                "q3": float(q3),
            }
    
    # =========================================================================
    # Save Output
    # =========================================================================
    if output_path:
        output_path = Path(output_path)
        with open(output_path, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"Analysis saved to: {output_path}")
    
    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results with unified scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analyze_results.py --results results/benchmark/
    python analyze_results.py --results results/benchmark/ --method geometric
    python analyze_results.py --results results/benchmark/ --compare-methods
    python analyze_results.py --results results/benchmark/ --output analysis.json
    python analyze_results.py --results results/benchmark/ --model gpt-4o

Scoring Methods:
    weighted      - Weighted linear combination (default weights)
    geometric     - Geometric mean (penalizes weak metrics)
    hierarchical  - Two-level category aggregation (recommended)
    percentile    - Percentile rank vs population
    topsis        - Multi-criteria decision analysis
    pareto        - Pareto dominance score
        """,
    )
    
    parser.add_argument(
        "--results", "-r",
        required=True,
        help="Path to results directory or benchmark_result.json",
    )
    parser.add_argument(
        "--method", "-m",
        default="hierarchical",
        choices=["weighted", "geometric", "hierarchical", "percentile", "topsis", "pareto"],
        help="Scoring method (default: hierarchical)",
    )
    parser.add_argument(
        "--model",
        help="Filter to specific model",
    )
    parser.add_argument(
        "--compare-methods", "-c",
        action="store_true",
        help="Compare all scoring methods",
    )
    parser.add_argument(
        "--output", "-o",
        help="Save analysis to JSON file",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output",
    )
    
    args = parser.parse_args()
    
    analyze_results(
        results_path=args.results,
        method=args.method,
        model_filter=args.model,
        compare_methods=args.compare_methods,
        output_path=args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
