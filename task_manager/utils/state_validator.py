"""
State Validation and Health Check Module

Provides comprehensive state integrity validation and health checks
to detect silent failures, hangs, and data corruption.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import hashlib

from task_manager.models import AgentState, TaskStatus
from task_manager.utils.logger import get_logger

logger = get_logger(__name__)


class StateValidator:
    """Validates agent state integrity and detects anomalies."""

    @staticmethod
    def validate_state_integrity(state: AgentState) -> Tuple[bool, List[str]]:
        """
        Validate that state is internally consistent.

        Args:
            state: Agent state to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check required fields
        required_fields = ["objective", "tasks", "completed_task_ids", "failed_task_ids"]
        for field in required_fields:
            if field not in state:
                errors.append(f"Missing required field: {field}")

        # Check task list consistency
        tasks = state.get("tasks", [])
        completed_ids = set(state.get("completed_task_ids", []))
        failed_ids = set(state.get("failed_task_ids", []))
        task_ids = {t.get("id") for t in tasks if isinstance(t, dict)}

        # Completed/failed IDs should exist in tasks
        invalid_completed = completed_ids - task_ids
        if invalid_completed:
            errors.append(f"Completed task IDs not in task list: {invalid_completed}")

        invalid_failed = failed_ids - task_ids
        if invalid_failed:
            errors.append(f"Failed task IDs not in task list: {invalid_failed}")

        # No overlap between completed and failed
        overlap = completed_ids & failed_ids
        if overlap:
            errors.append(f"Task IDs in both completed and failed: {overlap}")

        # Check task structure
        for task in tasks:
            if isinstance(task, dict):
                task_id = task.get("id", "unknown")

                # Required task fields
                if "description" not in task:
                    errors.append(f"Task {task_id}: missing 'description'")
                if "status" not in task:
                    errors.append(f"Task {task_id}: missing 'status'")

                # Valid status
                valid_statuses = {
                    TaskStatus.PENDING,
                    TaskStatus.ANALYZING,
                    TaskStatus.BROKEN_DOWN,
                    TaskStatus.EXECUTING,
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                }
                if task.get("status") not in valid_statuses:
                    errors.append(f"Task {task_id}: invalid status '{task.get('status')}'")

        # Check blackboard entries
        blackboard = state.get("blackboard", [])
        for i, entry in enumerate(blackboard):
            if not isinstance(entry, dict):
                errors.append(f"Blackboard entry {i}: not a dictionary")
            elif "entry_type" not in entry:
                errors.append(f"Blackboard entry {i}: missing 'entry_type'")

        return len(errors) == 0, errors

    @staticmethod
    def check_task_duplication(state: AgentState) -> List[str]:
        """
        Check for duplicates in completed_task_ids.

        Args:
            state: Agent state

        Returns:
            List of duplicate task IDs
        """
        completed_ids = state.get("completed_task_ids", [])
        seen = set()
        duplicates = []

        for task_id in completed_ids:
            if task_id in seen:
                duplicates.append(task_id)
            seen.add(task_id)

        return duplicates

    @staticmethod
    def validate_active_task(state: AgentState) -> Tuple[bool, Optional[str]]:
        """
        Validate that active_task_id points to a valid task.

        Args:
            state: Agent state

        Returns:
            Tuple of (is_valid, error_message)
        """
        active_task_id = state.get("active_task_id")
        tasks = state.get("tasks", [])

        if not active_task_id:
            return False, "No active_task_id set"

        task_ids = {t.get("id") for t in tasks if isinstance(t, dict)}
        if active_task_id not in task_ids:
            return False, f"Active task {active_task_id} not in tasks list"

        return True, None

    @staticmethod
    def compute_state_hash(state: AgentState) -> str:
        """
        Compute a hash of the current state for change detection.

        Args:
            state: Agent state

        Returns:
            SHA256 hash of state
        """
        import json

        # Extract key fields for hashing
        hashable_state = {
            "num_tasks": len(state.get("tasks", [])),
            "completed_count": len(set(state.get("completed_task_ids", []))),
            "failed_count": len(set(state.get("failed_task_ids", []))),
            "active_task_id": state.get("active_task_id", ""),
            "iteration_count": state.get("iteration_count", 0),
        }

        state_str = json.dumps(hashable_state, sort_keys=True)
        return hashlib.sha256(state_str.encode()).hexdigest()[:16]


class HealthChecker:
    """Monitors agent health and detects hangs/failures."""

    def __init__(self, max_iterations: int = 100, timeout_seconds: float = 600):
        """
        Initialize health checker.

        Args:
            max_iterations: Maximum allowed iterations before hang detection
            timeout_seconds: Timeout for node execution
        """
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.node_start_time: Optional[datetime] = None
        self.node_name: Optional[str] = None
        self.last_state_change: Optional[datetime] = None
        self.last_state_hash: Optional[str] = None

    def start_node_execution(self, node_name: str) -> None:
        """
        Record when a node starts executing.

        Args:
            node_name: Name of the node
        """
        self.node_start_time = datetime.now()
        self.node_name = node_name
        logger.debug(f"[HEALTH] Node execution started: {node_name}")

    def end_node_execution(self) -> float:
        """
        Record when a node finishes executing.

        Returns:
            Execution duration in seconds
        """
        if not self.node_start_time:
            return 0.0

        duration = (datetime.now() - self.node_start_time).total_seconds()
        logger.debug(f"[HEALTH] Node execution completed: {self.node_name} ({duration:.2f}s)")
        return duration

    def check_node_timeout(self) -> Tuple[bool, Optional[str]]:
        """
        Check if current node execution has exceeded timeout.

        Returns:
            Tuple of (is_timeout, message)
        """
        if not self.node_start_time:
            return False, None

        elapsed = (datetime.now() - self.node_start_time).total_seconds()
        if elapsed > self.timeout_seconds:
            message = f"Node {self.node_name} exceeded timeout: {elapsed:.0f}s > {self.timeout_seconds}s"
            logger.warning(f"[HEALTH] ⚠️  {message}")
            return True, message

        return False, None

    def detect_state_hang(self, state: AgentState) -> Tuple[bool, Optional[str]]:
        """
        Detect if state has stopped changing (hang condition).

        Args:
            state: Current agent state

        Returns:
            Tuple of (is_hanging, message)
        """
        current_hash = StateValidator.compute_state_hash(state)

        # First call
        if self.last_state_hash is None:
            self.last_state_hash = current_hash
            self.last_state_change = datetime.now()
            return False, None

        # State changed
        if current_hash != self.last_state_hash:
            self.last_state_hash = current_hash
            self.last_state_change = datetime.now()
            return False, None

        # State unchanged for too long
        if self.last_state_change:
            stalled_duration = (datetime.now() - self.last_state_change).total_seconds()
            if stalled_duration > self.timeout_seconds:
                message = f"State has not changed for {stalled_duration:.0f}s (hang detected)"
                logger.warning(f"[HEALTH] ⚠️  {message}")
                return True, message

        return False, None

    def check_iteration_limit(self, iteration_count: int) -> Tuple[bool, Optional[str]]:
        """
        Check if iteration count exceeds limit.

        Args:
            iteration_count: Current iteration count

        Returns:
            Tuple of (exceeded, message)
        """
        if iteration_count >= self.max_iterations:
            message = f"Iteration limit exceeded: {iteration_count} >= {self.max_iterations}"
            logger.warning(f"[HEALTH] ⚠️  {message}")
            return True, message

        return False, None

    def check_task_progress(self, state: AgentState) -> Tuple[bool, Optional[str]]:
        """
        Check that tasks are making progress.

        Args:
            state: Agent state

        Returns:
            Tuple of (is_progressing, message)
        """
        tasks = state.get("tasks", [])
        completed = len(set(state.get("completed_task_ids", [])))
        failed = len(set(state.get("failed_task_ids", [])))
        pending = len([t for t in tasks if isinstance(t, dict) and t.get("status") == TaskStatus.PENDING])

        total_done = completed + failed

        # No tasks completed or failed
        if total_done == 0 and len(tasks) > 0:
            iteration_count = state.get("iteration_count", 0)
            if iteration_count > 5:  # Allow some iterations for initial setup
                message = f"No task progress after {iteration_count} iterations: {completed} completed, {failed} failed, {pending} pending"
                logger.warning(f"[HEALTH] ⚠️  {message}")
                return False, message

        return True, None

    def get_health_report(self, state: AgentState) -> str:
        """
        Generate a comprehensive health report.

        Args:
            state: Agent state

        Returns:
            Formatted health report
        """
        is_valid, errors = StateValidator.validate_state_integrity(state)
        duplicates = StateValidator.check_task_duplication(state)
        active_valid, active_error = StateValidator.validate_active_task(state)
        state_hash = StateValidator.compute_state_hash(state)

        tasks = state.get("tasks", [])
        completed = len(set(state.get("completed_task_ids", [])))
        failed = len(set(state.get("failed_task_ids", [])))

        report = []
        report.append("\n" + "=" * 80)
        report.append("AGENT HEALTH REPORT")
        report.append("=" * 80)

        report.append(f"\nState Integrity: {'✓ VALID' if is_valid else '✗ INVALID'}")
        if errors:
            for error in errors:
                report.append(f"  • {error}")

        report.append(f"\nActive Task: {'✓ VALID' if active_valid else '✗ INVALID'}")
        if active_error:
            report.append(f"  • {active_error}")

        report.append(f"\nTask Duplication: {'✓ NONE' if not duplicates else '✗ DETECTED'}")
        if duplicates:
            report.append(f"  • Duplicate IDs: {duplicates}")

        report.append(f"\nTask Progress:")
        report.append(f"  • Total Tasks: {len(tasks)}")
        report.append(f"  • Completed: {completed}")
        report.append(f"  • Failed: {failed}")
        report.append(f"  • Pending: {len([t for t in tasks if isinstance(t, dict) and t.get('status') == TaskStatus.PENDING])}")

        report.append(f"\nState Hash: {state_hash}")

        is_timeout, timeout_msg = self.check_node_timeout()
        if is_timeout:
            report.append(f"\n⚠️  TIMEOUT: {timeout_msg}")

        report.append("\n" + "=" * 80)
        return "\n".join(report)
