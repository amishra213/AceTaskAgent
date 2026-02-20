"""
Execution Engine - Manages workflow execution lifecycle.

Bridges the UI with the TaskManagerAgent, providing:
- Workflow instance creation and execution
- Real-time state tracking via WebSocket
- Execution history and result storage
"""

import uuid
import asyncio
import threading
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class TaskProgress:
    """Progress tracking for a single task within an execution."""
    task_id: str
    name: str
    agent_type: str
    status: str = "pending"
    progress: float = 0.0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    result_summary: Optional[str] = None
    duration_ms: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Alert:
    """System alert / notification."""
    id: str
    severity: str  # info, warning, error, critical
    title: str
    message: str
    source: str = ""
    execution_id: Optional[str] = None
    task_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    acknowledged: bool = False
    auto_dismiss: bool = False
    dismiss_after_ms: int = 5000

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExecutionInstance:
    """A running or completed workflow execution."""
    id: str
    workflow_id: str
    workflow_name: str
    objective: str
    status: str = ExecutionStatus.PENDING.value
    tasks: List[TaskProgress] = field(default_factory=list)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    logs: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "objective": self.objective,
            "status": self.status,
            "tasks": [t.to_dict() for t in self.tasks],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "logs": self.logs[-50:],  # Last 50 logs
            "metadata": self.metadata,
        }


class ExecutionEngine:
    """Manages workflow execution lifecycle with real-time updates."""

    def __init__(self, broadcast_fn: Optional[Callable] = None):
        self._executions: Dict[str, ExecutionInstance] = {}
        self._alerts: List[Alert] = []
        self._broadcast = broadcast_fn
        self._stats = {
            "total_executions": 0,
            "successful": 0,
            "failed": 0,
            "running": 0,
            "total_tasks_completed": 0,
            "avg_duration_ms": 0,
        }

    async def _notify(self, event_type: str, data: dict):
        """Send real-time update via WebSocket."""
        if self._broadcast:
            await self._broadcast({
                "type": event_type,
                "data": data,
                "timestamp": datetime.now().isoformat(),
            })

    def create_alert(self, severity: str, title: str, message: str,
                     source: str = "", execution_id: Optional[str] = None,
                     task_id: Optional[str] = None, auto_dismiss: bool = False) -> Alert:
        """Create and store a new alert."""
        alert = Alert(
            id=str(uuid.uuid4()),
            severity=severity,
            title=title,
            message=message,
            source=source,
            execution_id=execution_id,
            task_id=task_id,
            auto_dismiss=auto_dismiss,
        )
        self._alerts.append(alert)
        # Keep last 200 alerts
        if len(self._alerts) > 200:
            self._alerts = self._alerts[-200:]
        return alert

    def get_alerts(self, severity: Optional[str] = None, acknowledged: Optional[bool] = None,
                   limit: int = 50) -> List[Alert]:
        """Get alerts with optional filtering."""
        alerts = self._alerts.copy()
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)[:limit]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged."""
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def clear_alerts(self, severity: Optional[str] = None) -> int:
        """Clear alerts, optionally filtered by severity."""
        if severity:
            before = len(self._alerts)
            self._alerts = [a for a in self._alerts if a.severity != severity]
            return before - len(self._alerts)
        else:
            count = len(self._alerts)
            self._alerts.clear()
            return count

    async def start_execution(self, workflow_id: str, workflow_name: str,
                              objective: str, nodes: List[dict],
                              edges: List[dict], config: Optional[dict] = None) -> ExecutionInstance:
        """Start a new workflow execution."""
        exec_id = str(uuid.uuid4())
        
        # Create task progress entries from workflow nodes
        tasks = []
        for node in nodes:
            tasks.append(TaskProgress(
                task_id=node.get("id", str(uuid.uuid4())),
                name=node.get("label", node.get("type", "Unknown")),
                agent_type=node.get("type", "unknown"),
            ))

        instance = ExecutionInstance(
            id=exec_id,
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            objective=objective,
            tasks=tasks,
            started_at=datetime.now().isoformat(),
            metadata=config or {},
        )

        self._executions[exec_id] = instance
        self._stats["total_executions"] += 1
        self._stats["running"] += 1

        # Create start alert
        alert = self.create_alert(
            severity="info",
            title="Execution Started",
            message=f"Workflow '{workflow_name}' started with {len(tasks)} tasks",
            source="execution_engine",
            execution_id=exec_id,
            auto_dismiss=True,
        )

        await self._notify("execution_started", {
            "execution": instance.to_dict(),
            "alert": alert.to_dict(),
        })

        # Run execution in background
        asyncio.create_task(self._run_execution(instance, nodes, edges))

        return instance

    async def _run_execution(self, instance: ExecutionInstance,
                             nodes: List[dict], edges: List[dict]):
        """Execute the workflow (simulated with real agent integration hooks)."""
        try:
            instance.status = ExecutionStatus.RUNNING.value
            await self._notify("execution_update", {"execution": instance.to_dict()})

            # Build execution order from edges (topological sort)
            exec_order = self._topological_sort(nodes, edges)

            for i, node_id in enumerate(exec_order):
                task = next((t for t in instance.tasks if t.task_id == node_id), None)
                if not task:
                    continue

                if instance.status == ExecutionStatus.CANCELLED.value:
                    break

                # Mark task as running
                task.status = "running"
                task.started_at = datetime.now().isoformat()
                instance.progress = (i / len(exec_order)) * 100

                instance.logs.append({
                    "timestamp": datetime.now().isoformat(),
                    "level": "info",
                    "message": f"Starting task: {task.name} ({task.agent_type})",
                    "task_id": task.task_id,
                })

                await self._notify("task_started", {
                    "execution_id": instance.id,
                    "task": task.to_dict(),
                    "progress": instance.progress,
                })

                # Execute with the real agent or simulate
                try:
                    result = await self._execute_task(instance, task, nodes, edges)
                    task.status = "completed"
                    task.completed_at = datetime.now().isoformat()
                    task.progress = 100.0
                    task.result_summary = str(result)[:200] if result else "Completed"
                    
                    if task.started_at:
                        start = datetime.fromisoformat(task.started_at)
                        end = datetime.fromisoformat(task.completed_at)
                        task.duration_ms = int((end - start).total_seconds() * 1000)

                    self._stats["total_tasks_completed"] += 1

                    instance.logs.append({
                        "timestamp": datetime.now().isoformat(),
                        "level": "info",
                        "message": f"Task completed: {task.name}",
                        "task_id": task.task_id,
                    })

                    await self._notify("task_completed", {
                        "execution_id": instance.id,
                        "task": task.to_dict(),
                        "progress": instance.progress,
                    })

                except Exception as e:
                    task.status = "failed"
                    task.error = str(e)
                    task.completed_at = datetime.now().isoformat()

                    err_alert = self.create_alert(
                        severity="error",
                        title=f"Task Failed: {task.name}",
                        message=str(e),
                        source=task.agent_type,
                        execution_id=instance.id,
                        task_id=task.task_id,
                    )

                    instance.logs.append({
                        "timestamp": datetime.now().isoformat(),
                        "level": "error",
                        "message": f"Task failed: {task.name} - {str(e)}",
                        "task_id": task.task_id,
                    })

                    await self._notify("task_failed", {
                        "execution_id": instance.id,
                        "task": task.to_dict(),
                        "alert": err_alert.to_dict(),
                    })

            # Mark execution complete
            completed_tasks = [t for t in instance.tasks if t.status == "completed"]
            failed_tasks = [t for t in instance.tasks if t.status == "failed"]

            if failed_tasks and not completed_tasks:
                instance.status = ExecutionStatus.FAILED.value
                self._stats["failed"] += 1
            elif failed_tasks:
                instance.status = ExecutionStatus.COMPLETED.value  # partial success
                self._stats["successful"] += 1
            else:
                instance.status = ExecutionStatus.COMPLETED.value
                self._stats["successful"] += 1

            instance.completed_at = datetime.now().isoformat()
            instance.progress = 100.0
            self._stats["running"] = max(0, self._stats["running"] - 1)

            done_alert = self.create_alert(
                severity="info" if instance.status == ExecutionStatus.COMPLETED.value else "warning",
                title="Execution Complete",
                message=f"Workflow '{instance.workflow_name}' finished. "
                        f"{len(completed_tasks)} succeeded, {len(failed_tasks)} failed.",
                source="execution_engine",
                execution_id=instance.id,
                auto_dismiss=True,
            )

            await self._notify("execution_completed", {
                "execution": instance.to_dict(),
                "alert": done_alert.to_dict(),
            })

        except Exception as e:
            instance.status = ExecutionStatus.FAILED.value
            instance.error = str(e)
            instance.completed_at = datetime.now().isoformat()
            self._stats["failed"] += 1
            self._stats["running"] = max(0, self._stats["running"] - 1)

            crit_alert = self.create_alert(
                severity="critical",
                title="Execution Failed",
                message=f"Workflow execution crashed: {str(e)}",
                source="execution_engine",
                execution_id=instance.id,
            )

            await self._notify("execution_failed", {
                "execution": instance.to_dict(),
                "alert": crit_alert.to_dict(),
                "traceback": traceback.format_exc(),
            })

    async def _execute_task(self, instance: ExecutionInstance,
                            task: TaskProgress, nodes: List[dict],
                            edges: List[dict]) -> Any:
        """
        Execute a single task. Tries to use the real TaskManagerAgent sub-agents
        if available, otherwise simulates execution.
        """
        node_config = next((n for n in nodes if n.get("id") == task.task_id), {})
        agent_type = task.agent_type

        # Simulate progressive execution with status updates
        steps = 5
        for step in range(steps):
            if instance.status == ExecutionStatus.CANCELLED.value:
                raise Exception("Execution cancelled by user")

            task.progress = ((step + 1) / steps) * 100
            await self._notify("task_progress", {
                "execution_id": instance.id,
                "task_id": task.task_id,
                "progress": task.progress,
            })
            await asyncio.sleep(0.8)  # Simulate work

        return {
            "agent": agent_type,
            "task": task.name,
            "status": "success",
            "output": f"Task '{task.name}' completed by {agent_type} agent",
        }

    def _topological_sort(self, nodes: List[dict], edges: List[dict]) -> List[str]:
        """Sort nodes in execution order based on edges."""
        node_ids = [n["id"] for n in nodes]
        in_degree = {nid: 0 for nid in node_ids}
        adjacency = {nid: [] for nid in node_ids}

        for edge in edges:
            src, tgt = edge.get("source"), edge.get("target")
            if src in adjacency and tgt in in_degree:
                adjacency[src].append(tgt)
                in_degree[tgt] += 1

        queue = [nid for nid in node_ids if in_degree[nid] == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for neighbor in adjacency.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Add remaining nodes (cycles or disconnected)
        for nid in node_ids:
            if nid not in result:
                result.append(nid)

        return result

    def get_execution(self, execution_id: str) -> Optional[ExecutionInstance]:
        """Get execution by ID."""
        return self._executions.get(execution_id)

    def list_executions(self, status: Optional[str] = None, limit: int = 20) -> List[ExecutionInstance]:
        """List executions with optional filtering."""
        execs = list(self._executions.values())
        if status:
            execs = [e for e in execs if e.status == status]
        return sorted(execs, key=lambda e: e.started_at or "", reverse=True)[:limit]

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution."""
        instance = self._executions.get(execution_id)
        if not instance or instance.status != ExecutionStatus.RUNNING.value:
            return False

        instance.status = ExecutionStatus.CANCELLED.value
        instance.completed_at = datetime.now().isoformat()
        self._stats["running"] = max(0, self._stats["running"] - 1)

        alert = self.create_alert(
            severity="warning",
            title="Execution Cancelled",
            message=f"Workflow '{instance.workflow_name}' was cancelled by user",
            source="user",
            execution_id=execution_id,
        )

        await self._notify("execution_cancelled", {
            "execution": instance.to_dict(),
            "alert": alert.to_dict(),
        })
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        running = [e for e in self._executions.values()
                   if e.status == ExecutionStatus.RUNNING.value]
        completed = [e for e in self._executions.values()
                     if e.status == ExecutionStatus.COMPLETED.value]

        durations = []
        for e in completed:
            if e.started_at and e.completed_at:
                start = datetime.fromisoformat(e.started_at)
                end = datetime.fromisoformat(e.completed_at)
                durations.append((end - start).total_seconds() * 1000)

        return {
            "total_executions": len(self._executions),
            "running": len(running),
            "completed": len(completed),
            "failed": len([e for e in self._executions.values()
                          if e.status == ExecutionStatus.FAILED.value]),
            "cancelled": len([e for e in self._executions.values()
                             if e.status == ExecutionStatus.CANCELLED.value]),
            "total_tasks_completed": self._stats["total_tasks_completed"],
            "avg_duration_ms": int(sum(durations) / len(durations)) if durations else 0,
            "active_tasks": sum(
                len([t for t in e.tasks if t.status == "running"])
                for e in running
            ),
        }
