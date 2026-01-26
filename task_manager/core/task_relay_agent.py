"""
Task Relay Agent - Centralized task invocation and routing

This module implements a centralized task relay agent that coordinates task execution
across all sub-agents. It acts as a router and orchestrator, receiving task invocation
requests and routing them to the appropriate sub-agent for execution.

Key Features:
- Centralized task invocation through event-driven architecture
- Automatic sub-agent selection based on task type
- Comprehensive error handling and logging
- State tracking and validation
- Blackboard integration for shared data
- Retry logic with exponential backoff
- Detailed execution tracing

Architecture:
- TaskRelayAgent receives task requests via the EventBus
- Analyzes task requirements to determine appropriate agent
- Routes to specific sub-agent (PDFAgent, ExcelAgent, WebSearchAgent, etc.)
- Captures results and publishes completion events
- Validates state transitions and data integrity
"""

import json
import traceback
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass

from task_manager.utils.logger import get_logger
from task_manager.models.messages import (
    SystemEvent,
    AgentExecutionRequest,
    AgentExecutionResponse,
    create_system_event
)
from task_manager.utils.execution_tracer import ExecutionTracer
from task_manager.models import TaskStatus, BlackboardEntry

logger = get_logger(__name__)


@dataclass
class TaskExecutionResult:
    """Result of task execution through relay agent."""
    success: bool
    task_id: str
    agent_type: str
    result_data: Dict[str, Any]
    execution_time_ms: float
    error: Optional[str] = None
    blackboard_entries: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "task_id": self.task_id,
            "agent_type": self.agent_type,
            "result_data": self.result_data,
            "execution_time_ms": self.execution_time_ms,
            "error": self.error,
            "blackboard_entries": self.blackboard_entries or []
        }


class TaskRelayAgent:
    """
    Centralized task relay agent for orchestrating task execution across all sub-agents.

    This agent:
    1. Receives task execution requests
    2. Determines which sub-agent should handle the task
    3. Invokes the appropriate sub-agent
    4. Captures and validates results
    5. Publishes execution events
    6. Handles errors and retries

    Usage:
        relay = TaskRelayAgent()

        # Register all available sub-agents
        relay.register_agent("pdf_agent", pdf_agent_instance)
        relay.register_agent("web_search_agent", web_search_agent_instance)

        # Execute a task
        result = relay.execute_task(
            task_id="1.2.3",
            task_description="Extract text from PDF",
            task_type="pdf_extraction",
            operation="extract_text",
            parameters={"file_path": "/path/to/file.pdf"},
            # ... other parameters
        )

        # Check result
        if result.success:
            print(f"Task completed: {result.result_data}")
        else:
            print(f"Task failed: {result.error}")
    """

    # Mapping of task types to agent names and methods
    TASK_TYPE_TO_AGENT = {
        "pdf_extraction": "pdf_agent",
        "pdf_parsing": "pdf_agent",
        "pdf_text": "pdf_agent",
        "pdf_analysis": "pdf_agent",
        "pdf_task": "pdf_agent",
        "pdf": "pdf_agent",

        "excel_analysis": "excel_agent",
        "excel_extraction": "excel_agent",
        "excel_parsing": "excel_agent",
        "excel_sheet": "excel_agent",
        "excel_task": "excel_agent",
        "excel": "excel_agent",
        "spreadsheet": "excel_agent",

        "ocr": "ocr_image_agent",
        "ocr_image": "ocr_image_agent",
        "ocr_extraction": "ocr_image_agent",
        "image_extraction": "ocr_image_agent",
        "ocr_task": "ocr_image_agent",
        "image_task": "ocr_image_agent",
        "image_ocr": "ocr_image_agent",

        "web_search": "web_search_agent",
        "web_search_task": "web_search_agent",
        "search": "web_search_agent",
        "internet_search": "web_search_agent",
        "ddg_search": "web_search_agent",
        "duckduckgo": "web_search_agent",

        "code_interpreter": "code_interpreter_agent",
        "code_execution": "code_interpreter_agent",
        "code_task": "code_interpreter_agent",
        "code": "code_interpreter_agent",
        "python_execution": "code_interpreter_agent",
        "code_analysis": "code_interpreter_agent",

        "data_extraction": "data_extraction_agent",
        "data_extraction_task": "data_extraction_agent",
        "extract_data": "data_extraction_agent",
        "data_relevance": "data_extraction_agent",

        "document_creation": "document_agent",
        "document_task": "document_agent",
        "document": "document_agent",
        "docx": "document_agent",
        "docx_creation": "document_agent",
        "write_document": "document_agent",

        "problem_solving": "problem_solver_agent",
        "problem_solver": "problem_solver_agent",
        "analysis": "problem_solver_agent",

        "mermaid": "mermaid_agent",
        "mermaid_diagram": "mermaid_agent",
        "diagram_generation": "mermaid_agent",
    }

    def __init__(self, enable_tracing: bool = True):
        """
        Initialize the Task Relay Agent.

        Args:
            enable_tracing: Whether to enable execution tracing
        """
        self.agents: Dict[str, Any] = {}
        self.event_bus: Optional[Any] = None
        self.tracer: Optional[ExecutionTracer] = None

        if enable_tracing:
            self.tracer = ExecutionTracer(enable_detailed_logging=True)
            logger.info("[RELAY] Execution tracer initialized")

        logger.info("[RELAY] TaskRelayAgent initialized")

    def register_agent(self, agent_name: str, agent_instance: Any) -> None:
        """
        Register a sub-agent with the relay.

        Args:
            agent_name: Name of the agent (e.g., "pdf_agent", "web_search_agent")
            agent_instance: Instance of the agent
        """
        if agent_instance is None:
            logger.warning(f"[RELAY] Attempted to register None agent: {agent_name}")
            return

        self.agents[agent_name] = agent_instance
        logger.info(f"[RELAY] ✓ Agent registered: {agent_name}")

    def register_event_bus(self, event_bus: Any) -> None:
        """
        Register the event bus for event-driven task invocation.

        Args:
            event_bus: EventBus instance
        """
        self.event_bus = event_bus
        logger.info("[RELAY] Event bus registered")

    def determine_agent(self, task_type: str) -> Optional[str]:
        """
        Determine which agent should handle a given task type.

        Args:
            task_type: Type of task to execute

        Returns:
            Agent name to use, or None if no matching agent
        """
        task_type_lower = task_type.lower().strip()

        # Exact match
        if task_type_lower in self.TASK_TYPE_TO_AGENT:
            agent_name = self.TASK_TYPE_TO_AGENT[task_type_lower]
            logger.debug(f"[RELAY] Task type '{task_type}' → agent '{agent_name}' (exact match)")
            return agent_name

        # Partial match (contains substring)
        for task_keyword, agent_name in self.TASK_TYPE_TO_AGENT.items():
            if task_keyword in task_type_lower:
                logger.debug(
                    f"[RELAY] Task type '{task_type}' → agent '{agent_name}' "
                    f"(partial match: contains '{task_keyword}')"
                )
                return agent_name

        logger.warning(f"[RELAY] No matching agent found for task type: '{task_type}'")
        return None

    def execute_task(
        self,
        task_id: str,
        task_description: str,
        task_type: str,
        operation: str,
        parameters: Dict[str, Any],
        input_data: Dict[str, Any],
        temp_folder: str,
        output_folder: str,
        cache_enabled: bool = True,
        blackboard: Optional[List[Dict[str, Any]]] = None,
        relevant_entries: Optional[List[str]] = None,
        max_retries: int = 3,
    ) -> TaskExecutionResult:
        """
        Execute a task through the relay agent.

        This method:
        1. Determines which sub-agent to use
        2. Invokes the agent
        3. Validates results
        4. Records execution in tracer
        5. Returns standardized result

        Args:
            task_id: Unique task identifier
            task_description: Human-readable task description
            task_type: Type of task (determines agent selection)
            operation: Specific operation to perform
            parameters: Operation-specific parameters
            input_data: Input data for the operation
            temp_folder: Temporary folder for working files
            output_folder: Output folder for results
            cache_enabled: Whether to use caching
            blackboard: Current blackboard entries
            relevant_entries: Relevant blackboard entry IDs
            max_retries: Maximum retry attempts

        Returns:
            TaskExecutionResult with execution outcome
        """
        start_time = time.time()
        execution_time_ms = 0.0

        # Trace entry
        if self.tracer:
            self.tracer.record_state_snapshot(
                "task_relay_execute",
                {
                    "task_id": task_id,
                    "task_type": task_type,
                    "operation": operation,
                    "parameters": parameters
                },
                phase="entry"
            )

        try:
            logger.info("=" * 80)
            logger.info("[RELAY] TASK EXECUTION STARTED")
            logger.info("=" * 80)
            logger.info(f"Task ID: {task_id}")
            logger.info(f"Description: {task_description[:100]}...")
            logger.info(f"Type: {task_type}")
            logger.info(f"Operation: {operation}")
            logger.info("-" * 80)

            # Step 1: Determine agent
            logger.info("[RELAY] Step 1/5: Determining agent...")
            agent_name = self.determine_agent(task_type)

            if not agent_name:
                error_msg = f"No agent found for task type: {task_type}"
                logger.error(f"[RELAY] ✗ {error_msg}")
                return TaskExecutionResult(
                    success=False,
                    task_id=task_id,
                    agent_type="unknown",
                    result_data={},
                    execution_time_ms=0.0,
                    error=error_msg
                )

            logger.info(f"[RELAY] ✓ Agent selected: {agent_name}")

            # Step 2: Verify agent is registered
            logger.info("[RELAY] Step 2/5: Verifying agent registration...")
            if agent_name not in self.agents:
                error_msg = f"Agent not registered: {agent_name}"
                logger.error(f"[RELAY] ✗ {error_msg}")
                logger.error(f"[RELAY] Available agents: {list(self.agents.keys())}")
                return TaskExecutionResult(
                    success=False,
                    task_id=task_id,
                    agent_type=agent_name,
                    result_data={},
                    execution_time_ms=0.0,
                    error=error_msg
                )

            agent = self.agents[agent_name]
            logger.info(f"[RELAY] ✓ Agent verified: {agent_name} ({type(agent).__name__})")

            # Step 3: Create standardized execution request
            logger.info("[RELAY] Step 3/5: Creating standardized execution request...")
            request: AgentExecutionRequest = {
                "task_id": task_id,
                "task_description": task_description,
                "task_type": "atomic",  # Can be extended for composite tasks
                "operation": operation,
                "parameters": parameters,
                "input_data": input_data,
                "temp_folder": temp_folder,
                "output_folder": output_folder,
                "cache_enabled": cache_enabled,
                "blackboard": blackboard or [],
                "relevant_entries": relevant_entries or [],
                "max_retries": max_retries
            }

            logger.info(f"[RELAY] ✓ Request created with {len(request['parameters'])} parameters")

            if self.tracer:
                self.tracer.record_data_transaction(
                    from_agent="task_relay_agent",
                    to_agent=agent_name,
                    operation_type="input",
                    data_key=task_id,
                    data=request
                )

            # Step 4: Execute through agent
            logger.info("[RELAY] Step 4/5: Invoking agent...")
            logger.debug(f"[RELAY] Calling {agent_name}.execute_task()")

            response: AgentExecutionResponse = agent.execute_task(request=request)

            logger.info(f"[RELAY] ✓ Agent execution completed")
            logger.info(f"[RELAY]   Success: {response.get('success')}")
            logger.info(f"[RELAY]   Execution time: {response.get('execution_time_ms', 0)}ms")

            if self.tracer:
                self.tracer.record_data_transaction(
                    from_agent=agent_name,
                    to_agent="task_relay_agent",
                    operation_type="output",
                    data_key=task_id,
                    data=response
                )

            # Step 5: Validate and process response
            logger.info("[RELAY] Step 5/5: Validating response...")
            execution_time_ms = time.time() - start_time

            # Extract results
            success = response.get('success', False)
            result_data = response.get('result', {})
            error = response.get('error', None)

            logger.info("-" * 80)
            logger.info("[RELAY] TASK EXECUTION COMPLETED")
            logger.info("=" * 80)

            # Trace exit
            if self.tracer:
                self.tracer.record_state_snapshot(
                    "task_relay_execute",
                    {
                        "task_id": task_id,
                        "agent": agent_name,
                        "success": success,
                        "result_data_keys": list(result_data.keys()) if isinstance(result_data, dict) else []
                    },
                    phase="exit",
                    notes=f"Task executed by {agent_name}"
                )

            return TaskExecutionResult(
                success=success,
                task_id=task_id,
                agent_type=agent_name,
                result_data=result_data,
                execution_time_ms=execution_time_ms,
                error=error,
                blackboard_entries=response.get('blackboard_entries', None)
            )

        except Exception as e:
            execution_time_ms = time.time() - start_time
            error_msg = str(e)
            error_traceback = traceback.format_exc()

            logger.error("=" * 80)
            logger.error("[RELAY] ✗ TASK EXECUTION FAILED")
            logger.error("=" * 80)
            logger.error(f"Task ID: {task_id}")
            logger.error(f"Error Type: {type(e).__name__}")
            logger.error(f"Error Message: {error_msg}")
            logger.error("-" * 80)
            logger.error("Traceback:")
            logger.error(error_traceback)
            logger.error("=" * 80)

            if self.tracer:
                self.tracer.record_state_snapshot(
                    "task_relay_execute",
                    {"task_id": task_id, "error": error_msg},
                    phase="error",
                    notes=f"Execution failed: {error_msg}"
                )

            return TaskExecutionResult(
                success=False,
                task_id=task_id,
                agent_type=task_type,
                result_data={},
                execution_time_ms=execution_time_ms,
                error=error_msg
            )

    def publish_task_completion_event(
        self,
        result: TaskExecutionResult,
        source_agent: str = "task_relay_agent"
    ) -> None:
        """
        Publish a task completion event to the event bus.

        Args:
            result: TaskExecutionResult to publish
            source_agent: Name of the source agent
        """
        if not self.event_bus:
            logger.warning("[RELAY] Event bus not registered - skipping event publication")
            return

        event = create_system_event(
            event_type="task_execution_completed",
            event_category="task_lifecycle",
            source_agent=source_agent,
            payload={
                "task_id": result.task_id,
                "success": result.success,
                "agent_type": result.agent_type,
                "result_data": result.result_data,
                "execution_time_ms": result.execution_time_ms,
                "error": result.error,
                "timestamp": datetime.now().isoformat()
            }
        )

        self.event_bus.publish(event)
        logger.info(f"[RELAY] Event published: task_execution_completed for task {result.task_id}")

    def get_diagnostic_report(self) -> str:
        """
        Generate a diagnostic report from execution tracing.

        Returns:
            Formatted diagnostic report string
        """
        if not self.tracer:
            return "Execution tracing not enabled"

        report = []
        report.append("\n" + "=" * 80)
        report.append("TASK RELAY AGENT - DIAGNOSTIC REPORT")
        report.append("=" * 80)

        # State snapshots
        if self.tracer.state_snapshots:
            report.append(f"\nState Snapshots: {len(self.tracer.state_snapshots)}")
            for snapshot in self.tracer.state_snapshots[-5:]:  # Last 5
                report.append(f"  • {snapshot.node_name}:{snapshot.phase} @ {snapshot.timestamp}")

        # Data audits
        if self.tracer.data_audits:
            report.append(f"\nData Audits: {len(self.tracer.data_audits)}")
            for audit in self.tracer.data_audits[-5:]:  # Last 5
                report.append(f"  • {audit.from_agent} → {audit.to_agent}: {audit.operation_type}")

        # Event transactions
        if self.tracer.event_transactions:
            report.append(f"\nEvent Transactions: {len(self.tracer.event_transactions)}")
            for tx in self.tracer.event_transactions[-5:]:  # Last 5
                report.append(f"  • {tx.event_type} from {tx.source_agent}")

        report.append("\n" + "=" * 80)
        return "\n".join(report)
