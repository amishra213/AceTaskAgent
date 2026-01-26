"""
Execution Tracer - Comprehensive logging for state transactions and event flow

This module provides detailed tracing of:
1. State transitions between workflow nodes
2. Data flow and transformations
3. Event publishing and subscription
4. Agent-to-agent data passing
5. Blackboard mutations and validation
6. Message serialization/deserialization
7. Checksum validation for data integrity

Helps identify:
- Silent failures (where no exception occurs but execution stops)
- Data loss between node transitions
- Missed state updates
- Race conditions in event handling
- Deadlocks or infinite waits
"""

import json
import hashlib
import logging
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field, asdict
from functools import wraps
import traceback

from task_manager.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StateSnapshot:
    """Snapshot of agent state at a point in time."""
    timestamp: str
    node_name: str
    phase: str  # 'entry', 'exit', 'error'
    state_keys: List[str]
    state_hash: str
    state_size_bytes: int
    tasks_count: int
    completed_tasks: int
    failed_tasks: int
    pending_tasks: int
    active_task_id: Optional[str]
    iteration_count: int
    blackboard_size: int
    notes: str = ""
    
    def to_dict(self):
        return asdict(self)


@dataclass
class EventTransaction:
    """Record of an event publish/subscribe transaction."""
    timestamp: str
    event_id: str
    event_type: str
    source_agent: str
    target_agent: str
    payload_hash: str
    payload_size_bytes: int
    operation: str  # 'publish', 'subscribe', 'handle'
    status: str  # 'initiated', 'sent', 'received', 'processed', 'failed'
    error: Optional[str] = None
    processing_time_ms: float = 0.0
    
    def to_dict(self):
        return asdict(self)


@dataclass
class DataTransactionAudit:
    """Audit trail for data passed between agents."""
    timestamp: str
    from_agent: str
    to_agent: str
    operation_type: str  # 'input', 'output', 'cache', 'retrieve'
    data_key: str
    data_hash: str
    data_size_bytes: int
    validation_status: str  # 'valid', 'invalid', 'corrupted', 'missing'
    details: str = ""
    
    def to_dict(self):
        return asdict(self)


class ExecutionTracer:
    """Comprehensive execution tracer for debugging workflow issues."""
    
    def __init__(self, enable_detailed_logging: bool = True, max_history: int = 10000):
        """
        Initialize the execution tracer.
        
        Args:
            enable_detailed_logging: Whether to log detailed information
            max_history: Maximum number of records to keep in memory
        """
        self.enable_detailed_logging = enable_detailed_logging
        self.max_history = max_history
        
        # History buffers
        self.state_snapshots: List[StateSnapshot] = []
        self.event_transactions: List[EventTransaction] = []
        self.data_audits: List[DataTransactionAudit] = []
        self.routing_decisions: List[Dict[str, Any]] = []
        self.node_execution_times: Dict[str, List[float]] = {}
        
        logger.info("[TRACER] ExecutionTracer initialized")
    
    def record_state_snapshot(
        self,
        node_name: str,
        state: Dict[str, Any],
        phase: str = "entry",
        notes: str = ""
    ) -> None:
        """
        Record a snapshot of agent state.
        
        Args:
            node_name: Name of the workflow node
            state: Current state dictionary
            phase: 'entry', 'exit', or 'error'
            notes: Additional notes
        """
        if not self.enable_detailed_logging:
            return
        
        try:
            # Calculate state metrics
            state_keys = list(state.keys())
            state_str = json.dumps(state, default=str, indent=0)
            state_hash = self._calculate_hash(state_str)
            state_size = len(state_str.encode('utf-8'))
            
            tasks = state.get('tasks', [])
            completed_ids = set(state.get('completed_task_ids', []))
            failed_ids = set(state.get('failed_task_ids', []))
            pending_count = len(tasks) - len(completed_ids) - len(failed_ids)
            
            snapshot = StateSnapshot(
                timestamp=datetime.now().isoformat(),
                node_name=node_name,
                phase=phase,
                state_keys=state_keys,
                state_hash=state_hash,
                state_size_bytes=state_size,
                tasks_count=len(tasks),
                completed_tasks=len(completed_ids),
                failed_tasks=len(failed_ids),
                pending_tasks=pending_count,
                active_task_id=state.get('active_task_id'),
                iteration_count=state.get('iteration_count', 0),
                blackboard_size=len(state.get('blackboard', [])),
                notes=notes
            )
            
            self.state_snapshots.append(snapshot)
            
            # Trim history if needed
            if len(self.state_snapshots) > self.max_history:
                self.state_snapshots = self.state_snapshots[-self.max_history:]
            
            # Log in concise format
            logger.debug(
                f"[STATE SNAPSHOT] {node_name}:{phase} | "
                f"Keys:{len(state_keys)} Size:{state_size}B Hash:{state_hash[:8]} | "
                f"Tasks:{len(tasks)} Done:{len(completed_ids)} Pending:{pending_count}"
            )
            
        except Exception as e:
            logger.error(f"[TRACER] Error recording state snapshot: {str(e)}")
    
    def record_event_transaction(
        self,
        event_type: str,
        source_agent: str,
        target_agent: str,
        payload: Dict[str, Any],
        operation: str = "publish",
        status: str = "initiated"
    ) -> str:
        """
        Record an event transaction.
        
        Args:
            event_type: Type of event
            source_agent: Agent publishing the event
            target_agent: Agent receiving the event
            payload: Event payload
            operation: 'publish', 'subscribe', or 'handle'
            status: Status of the operation
            
        Returns:
            Event ID for tracking
        """
        event_id = f"evt_{datetime.now().timestamp()}_{source_agent}_{target_agent}"
        
        try:
            payload_str = json.dumps(payload, default=str)
            payload_hash = self._calculate_hash(payload_str)
            payload_size = len(payload_str.encode('utf-8'))
            
            transaction = EventTransaction(
                timestamp=datetime.now().isoformat(),
                event_id=event_id,
                event_type=event_type,
                source_agent=source_agent,
                target_agent=target_agent,
                payload_hash=payload_hash,
                payload_size_bytes=payload_size,
                operation=operation,
                status=status
            )
            
            self.event_transactions.append(transaction)
            
            if len(self.event_transactions) > self.max_history:
                self.event_transactions = self.event_transactions[-self.max_history:]
            
            logger.debug(
                f"[EVENT] {operation.upper()} {source_agent} → {target_agent} | "
                f"Type:{event_type} Size:{payload_size}B Hash:{payload_hash[:8]}"
            )
            
        except Exception as e:
            logger.error(f"[TRACER] Error recording event transaction: {str(e)}")
        
        return event_id
    
    def record_data_transaction(
        self,
        from_agent: str,
        to_agent: str,
        data_key: str,
        data: Any,
        operation_type: str = "output"
    ) -> None:
        """
        Record data passed between agents.
        
        Args:
            from_agent: Source agent
            to_agent: Target agent
            data_key: Key for the data
            data: The actual data
            operation_type: 'input', 'output', 'cache', 'retrieve'
        """
        try:
            data_str = json.dumps(data, default=str)
            data_hash = self._calculate_hash(data_str)
            data_size = len(data_str.encode('utf-8'))
            
            # Validate data integrity
            validation_status = self._validate_data(data)
            
            audit = DataTransactionAudit(
                timestamp=datetime.now().isoformat(),
                from_agent=from_agent,
                to_agent=to_agent,
                operation_type=operation_type,
                data_key=data_key,
                data_hash=data_hash,
                data_size_bytes=data_size,
                validation_status=validation_status
            )
            
            self.data_audits.append(audit)
            
            if len(self.data_audits) > self.max_history:
                self.data_audits = self.data_audits[-self.max_history:]
            
            if validation_status != "valid":
                logger.warning(
                    f"[DATA AUDIT] {from_agent} → {to_agent} | "
                    f"Key:{data_key} Status:{validation_status} Size:{data_size}B"
                )
            else:
                logger.debug(
                    f"[DATA AUDIT] {from_agent} → {to_agent} | "
                    f"Key:{data_key} Size:{data_size}B Hash:{data_hash[:8]}"
                )
            
        except Exception as e:
            logger.error(f"[TRACER] Error recording data transaction: {str(e)}")
    
    def record_routing_decision(
        self,
        current_node: str,
        active_task_id: str,
        decision: str,
        reasoning: str,
        analysis_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a routing decision for debugging.
        
        Args:
            current_node: Current workflow node
            active_task_id: Active task ID
            decision: Routing decision (next node)
            reasoning: Why this decision was made
            analysis_data: Additional analysis data
        """
        try:
            record = {
                "timestamp": datetime.now().isoformat(),
                "current_node": current_node,
                "active_task_id": active_task_id,
                "decision": decision,
                "reasoning": reasoning,
                "analysis_data": analysis_data or {}
            }
            
            self.routing_decisions.append(record)
            
            if len(self.routing_decisions) > self.max_history:
                self.routing_decisions = self.routing_decisions[-self.max_history:]
            
            logger.debug(
                f"[ROUTING] {current_node} + task:{active_task_id} → {decision} | {reasoning}"
            )
            
        except Exception as e:
            logger.error(f"[TRACER] Error recording routing decision: {str(e)}")
    
    def record_node_execution_time(self, node_name: str, duration_ms: float) -> None:
        """
        Record execution time for a workflow node.
        
        Args:
            node_name: Name of the node
            duration_ms: Execution time in milliseconds
        """
        if node_name not in self.node_execution_times:
            self.node_execution_times[node_name] = []
        
        self.node_execution_times[node_name].append(duration_ms)
        
        if len(self.node_execution_times[node_name]) > 100:
            self.node_execution_times[node_name] = self.node_execution_times[node_name][-100:]
        
        logger.debug(f"[TIMING] {node_name} executed in {duration_ms:.2f}ms")
    
    def get_state_transition_report(self) -> str:
        """Generate a report of state transitions."""
        report = ["=" * 80, "STATE TRANSITION REPORT", "=" * 80]
        
        for snapshot in self.state_snapshots[-50:]:  # Last 50
            report.append(
                f"{snapshot.timestamp} | {snapshot.node_name}:{snapshot.phase:5} | "
                f"Tasks:{snapshot.tasks_count:2} Done:{snapshot.completed_tasks:2} "
                f"Pending:{snapshot.pending_tasks:2} Hash:{snapshot.state_hash[:8]}"
            )
        
        return "\n".join(report)
    
    def get_event_transaction_report(self) -> str:
        """Generate a report of event transactions."""
        report = ["=" * 80, "EVENT TRANSACTION REPORT", "=" * 80]
        
        for txn in self.event_transactions[-50:]:  # Last 50
            report.append(
                f"{txn.timestamp} | {txn.operation:9} | "
                f"{txn.source_agent:15} → {txn.target_agent:15} | "
                f"{txn.event_type:20} | {txn.status}"
            )
        
        return "\n".join(report)
    
    def get_routing_decision_report(self) -> str:
        """Generate a report of routing decisions."""
        report = ["=" * 80, "ROUTING DECISION REPORT", "=" * 80]
        
        for decision in self.routing_decisions[-50:]:  # Last 50
            report.append(
                f"{decision['timestamp']} | {decision['current_node']:20} + "
                f"{decision['active_task_id']:10} → {decision['decision']:25} | "
                f"{decision['reasoning']}"
            )
        
        return "\n".join(report)
    
    def get_timing_report(self) -> str:
        """Generate a report of node execution times."""
        report = ["=" * 80, "NODE EXECUTION TIMING REPORT", "=" * 80]
        
        for node_name, times in sorted(self.node_execution_times.items()):
            if times:
                avg_ms = sum(times) / len(times)
                min_ms = min(times)
                max_ms = max(times)
                report.append(
                    f"{node_name:30} | Avg:{avg_ms:7.2f}ms | "
                    f"Min:{min_ms:7.2f}ms | Max:{max_ms:7.2f}ms | Calls:{len(times):3}"
                )
        
        return "\n".join(report)
    
    def get_full_diagnostic_report(self) -> str:
        """Generate a comprehensive diagnostic report."""
        report = [
            "=" * 80,
            "COMPREHENSIVE EXECUTION DIAGNOSTIC REPORT",
            "=" * 80,
            f"Generated: {datetime.now().isoformat()}",
            f"State snapshots: {len(self.state_snapshots)}",
            f"Event transactions: {len(self.event_transactions)}",
            f"Data audits: {len(self.data_audits)}",
            f"Routing decisions: {len(self.routing_decisions)}",
            "",
            self.get_state_transition_report(),
            "",
            self.get_event_transaction_report(),
            "",
            self.get_routing_decision_report(),
            "",
            self.get_timing_report(),
            "=" * 80
        ]
        
        return "\n".join(report)
    
    @staticmethod
    def _calculate_hash(data: str) -> str:
        """Calculate SHA256 hash of data."""
        return hashlib.sha256(data.encode('utf-8')).hexdigest()[:16]
    
    @staticmethod
    def _validate_data(data: Any) -> str:
        """
        Validate data integrity.
        
        Returns:
            'valid', 'invalid', 'corrupted', or 'missing'
        """
        if data is None:
            return "missing"
        
        if isinstance(data, dict) and len(data) == 0:
            return "invalid"
        
        if isinstance(data, list) and len(data) == 0:
            return "invalid"
        
        if isinstance(data, str) and len(data.strip()) == 0:
            return "invalid"
        
        return "valid"


def trace_node_execution(node_name: str, tracer: Optional['ExecutionTracer'] = None):
    """
    Decorator to trace node execution with timing and state snapshots.
    
    Usage:
        @trace_node_execution("my_node", tracer)
        def _my_node(self, state):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, state: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
            import time
            
            t = tracer or getattr(self, 'tracer', None)
            
            if t:
                t.record_state_snapshot(node_name, state, phase="entry")
            
            start_time = time.time()
            
            try:
                result = func(self, state, *args, **kwargs)
                
                if t:
                    t.record_state_snapshot(node_name, result, phase="exit")
                    duration_ms = (time.time() - start_time) * 1000
                    t.record_node_execution_time(node_name, duration_ms)
                
                return result
                
            except Exception as e:
                if t:
                    t.record_state_snapshot(
                        node_name, state, phase="error",
                        notes=f"Exception: {type(e).__name__}: {str(e)}"
                    )
                
                logger.error(f"[TRACE] Error in {node_name}: {str(e)}")
                logger.exception(e)
                raise
        
        return wrapper
    
    return decorator
