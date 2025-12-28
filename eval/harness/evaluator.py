"""Test evaluation and metrics computation."""

import subprocess
import re
from dataclasses import dataclass
from pathlib import Path

from ..task import Task
from .results import TaskResult


@dataclass 
class TestRunResult:
    """Result of running pytest."""
    passed: int
    failed: int
    errors: int
    total: int
    output: str
    success: bool
    test_details: list[dict]  # per-test results


class Evaluator:
    """Evaluates agent patches by running tests."""

    def __init__(self, timeout: int = 300):
        self.timeout = timeout

    def run_pytest(
        self,
        repo_path: Path,
        test_spec: str | list[str],
        timeout: int | None = None,
    ) -> TestRunResult:
        """
        Run pytest on specified tests.
        
        Args:
            repo_path: Path to repository
            test_spec: Test file, directory, or list of test IDs
            timeout: Timeout in seconds
            
        Returns:
            TestRunResult with pass/fail counts
        """
        timeout = timeout or self.timeout
        
        # Build pytest command
        if isinstance(test_spec, list):
            test_args = test_spec
        else:
            test_args = [test_spec]
        
        cmd = [
            "python", "-m", "pytest",
            "-v",
            "--tb=short",
            "-q",
        ] + test_args
        
        try:
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = result.stdout + result.stderr
            success = result.returncode == 0
        except subprocess.TimeoutExpired:
            output = f"Timeout after {timeout}s"
            success = False
        except Exception as e:
            output = f"Error running pytest: {e}"
            success = False
        
        # Parse results
        passed, failed, errors = self._parse_pytest_summary(output)
        test_details = self._parse_test_details(output)
        
        return TestRunResult(
            passed=passed,
            failed=failed,
            errors=errors,
            total=passed + failed + errors,
            output=output,
            success=success,
            test_details=test_details,
        )

    def _parse_pytest_summary(self, output: str) -> tuple[int, int, int]:
        """Parse pytest summary line for pass/fail/error counts."""
        passed = failed = errors = 0
        
        # Look for summary line like "5 passed, 2 failed, 1 error"
        # Or "5 passed in 1.23s"
        patterns = [
            (r"(\d+) passed", "passed"),
            (r"(\d+) failed", "failed"),
            (r"(\d+) errors?", "errors"),
        ]
        
        for pattern, key in patterns:
            match = re.search(pattern, output)
            if match:
                if key == "passed":
                    passed = int(match.group(1))
                elif key == "failed":
                    failed = int(match.group(1))
                elif key == "errors":
                    errors = int(match.group(1))
        
        return passed, failed, errors

    def _parse_test_details(self, output: str) -> list[dict]:
        """Parse individual test results from pytest output."""
        details = []
        
        # Look for lines like:
        # test_file.py::test_name PASSED
        # test_file.py::test_name FAILED
        pattern = r"([\w/]+\.py::\w+)\s+(PASSED|FAILED|ERROR)"
        
        for match in re.finditer(pattern, output):
            test_id = match.group(1)
            status = match.group(2)
            details.append({
                "test": test_id,
                "passed": status == "PASSED",
                "status": status,
            })
        
        return details

    def evaluate(
        self,
        task: Task,
        repo_path: Path,
        agent_patch: str,
        steps: int = 0,
        duration: float = 0.0,
        explanation: str = "",
    ) -> TaskResult:
        """
        Evaluate agent's patch against task requirements.
        
        Args:
            task: The task being evaluated
            repo_path: Path to repository (with agent's changes applied)
            agent_patch: Git diff of agent's changes
            steps: Number of agent steps taken
            duration: Time taken by agent
            explanation: Agent's explanation of the fix
            
        Returns:
            TaskResult with all metrics
        """
        result = TaskResult(
            task_id=task.id,
            agent_patch=agent_patch,
            steps=steps,
            duration=duration,
            explanation=explanation,
        )
        
        # Get diff stats
        from .repo_manager import RepoManager
        repo_mgr = RepoManager()
        stats = repo_mgr.get_diff_stats(repo_path)
        result.diff_size = stats["insertions"] + stats["deletions"]
        result.files_changed = repo_mgr.get_changed_files(repo_path)
        
        # Run fail_to_pass tests
        if task.fail_to_pass:
            f2p_result = self._run_tests_by_name(repo_path, task.fail_to_pass)
            result.fail_to_pass_total = f2p_result.total
            result.fail_to_pass_passed = f2p_result.passed
            result.fail_to_pass_results = f2p_result.test_details
            result.resolved = (f2p_result.failed == 0 and f2p_result.errors == 0 and f2p_result.total > 0)
        else:
            # No specific tests, consider resolved if patch was made
            result.resolved = bool(agent_patch.strip())
        
        # Run pass_to_pass tests
        if task.pass_to_pass:
            p2p_result = self._run_tests_by_name(repo_path, task.pass_to_pass)
            result.pass_to_pass_total = p2p_result.total
            result.pass_to_pass_passed = p2p_result.passed
            result.pass_to_pass_results = p2p_result.test_details
            result.no_regression = (p2p_result.failed == 0 and p2p_result.errors == 0)
        else:
            result.no_regression = True
        
        # Set timestamp
        from datetime import datetime
        result.timestamp = datetime.now().isoformat()
        
        return result

    def _run_tests_by_name(
        self, repo_path: Path, test_names: list[str]
    ) -> TestRunResult:
        """
        Run tests by name/pattern.
        
        Handles both full test IDs (file::test) and just test names.
        """
        # If test names look like full paths, use them directly
        # Otherwise, use -k to match by name
        if any("::" in name for name in test_names):
            return self.run_pytest(repo_path, test_names)
        else:
            # Use -k with OR to match any of the test names
            pattern = " or ".join(test_names)
            return self.run_pytest(repo_path, ["-k", pattern])

    def run_relevant_tests(
        self, repo_path: Path, changed_files: list[str]
    ) -> TestRunResult:
        """
        Run tests relevant to changed files.
        
        Discovers test files based on changed source files.
        """
        test_files = []
        
        for file in changed_files:
            if "test" in file:
                test_files.append(file)
            else:
                # Try to find corresponding test file
                # e.g., sklearn/metrics/pairwise.py -> sklearn/metrics/tests/test_pairwise.py
                path = Path(file)
                test_candidates = [
                    path.parent / "tests" / f"test_{path.name}",
                    path.parent / f"test_{path.name}",
                ]
                for candidate in test_candidates:
                    full_path = repo_path / candidate
                    if full_path.exists():
                        test_files.append(str(candidate))
                        break
        
        if test_files:
            return self.run_pytest(repo_path, test_files)
        else:
            return TestRunResult(
                passed=0, failed=0, errors=0, total=0,
                output="No test files found",
                success=True,
                test_details=[],
            )
