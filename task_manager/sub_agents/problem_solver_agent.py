"""
Problem Solver Sub-Agent for intelligent error resolution and human input interpretation.

Migration Status: Week 7 Day 2 - Dual Format Support

This agent acts as an intelligent intermediary that:
1. Analyzes errors and asks LLM for solutions
2. Interprets and formats human input for programmatic use
3. Provides context-aware suggestions for task recovery
4. Transforms natural language inputs into structured parameters

Capabilities:
- Error diagnosis and solution generation
- Human input parsing and formatting
- Parameter extraction from natural language
- Context-aware retry suggestions
- Multi-format output generation (JSON, dict, structured data)

Operations:
- diagnose_error: Diagnose an error and identify its category
- get_solution: Get LLM-generated solution for an error
- interpret_human_input: Interpret natural language input
- format_for_agent: Format data for consumption by specific agent
- generate_retry_parameters: Generate modified parameters for retry
- create_task_output: Create task output from human input
"""

from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path
from datetime import datetime
import json
import re
import time

# Import standardized schemas and utilities (Week 1-2 implementation)
from task_manager.models import (
    AgentExecutionRequest,
    AgentExecutionResponse,
    create_system_event,
    create_error_response
)
from task_manager.utils import (
    auto_convert_response,
    validate_agent_execution_response,
    exception_to_error_response,
    InvalidParameterError,
    AgentExecutionError,
    wrap_exception
)
from task_manager.core.event_bus import get_event_bus

from task_manager.utils.logger import get_logger

logger = get_logger(__name__)


class ProblemSolverAgent:
    """
    Sub-agent for intelligent problem solving and input interpretation.
    
    This agent serves as a universal helper that can:
    - Diagnose errors and suggest solutions via LLM
    - Parse human natural language input into structured formats
    - Generate retry parameters for failed tasks
    - Format data for consumption by other agents
    """
    
    def __init__(self, llm_client=None, config=None):
        """
        Initialize Problem Solver Agent with dual-format support.
        
        Args:
            llm_client: LangChain LLM client for generating solutions
            config: Optional configuration object
        """
        self.agent_name = "problem_solver_agent"
        self.llm_client = llm_client
        self.config = config or {}
        
        # Initialize event bus for event-driven workflows
        self.event_bus = get_event_bus()
        
        # Supported operations for execute_task routing
        self.supported_operations = [
            "diagnose_error",
            "get_solution", 
            "interpret_human_input",
            "format_for_agent",
            "generate_retry_parameters",
            "create_task_output"
        ]
        
        # Supported output formats
        self.output_formats = [
            "json",
            "dict", 
            "excel_params",
            "web_search_params",
            "file_operation_params",
            "task_context",
            "structured_text"
        ]
        
        # Common error patterns and their solution templates
        self.error_patterns = {
            "file_not_found": {
                "patterns": ["file not found", "no such file", "path does not exist", "filenotfounderror"],
                "solution_prompt": "The file was not found. Suggest alternative file paths or ways to create the missing file."
            },
            "permission_denied": {
                "patterns": ["permission denied", "access denied", "unauthorized"],
                "solution_prompt": "Access was denied. Suggest ways to handle permission issues or alternative approaches."
            },
            "invalid_path": {
                "patterns": ["invalid path", "empty output path", "path provided"],
                "solution_prompt": "The path provided was invalid or empty. Suggest how to construct a valid path."
            },
            "sheet_not_found": {
                "patterns": ["sheet not found", "worksheet not found", "no sheet named"],
                "solution_prompt": "The Excel sheet was not found. Suggest sheet name alternatives or how to create the sheet."
            },
            "api_error": {
                "patterns": ["api error", "rate limit", "429", "503", "timeout"],
                "solution_prompt": "An API error occurred. Suggest retry strategies or alternative approaches."
            },
            "parse_error": {
                "patterns": ["parse error", "json decode", "invalid syntax", "unexpected token"],
                "solution_prompt": "A parsing error occurred. Suggest how to fix the format or handle the malformed data."
            },
            "missing_dependency": {
                "patterns": ["modulenotfounderror", "import error", "no module named"],
                "solution_prompt": "A required module is missing. Suggest how to install or work around the dependency."
            },
            "data_validation": {
                "patterns": ["validation error", "invalid data", "type error", "value error"],
                "solution_prompt": "Data validation failed. Suggest how to fix or transform the data."
            }
        }
        
        logger.info(f"Problem Solver Agent initialized with dual-format support")
    
    
    def set_llm_client(self, llm_client):
        """
        Set or update the LLM client.
        
        Args:
            llm_client: LangChain LLM client
        """
        self.llm_client = llm_client
        logger.info("LLM client set for Problem Solver Agent")
    
    
    def execute_task(
        self,
        *args: Any,
        **kwargs: Any
    ) -> Union[Dict[str, Any], AgentExecutionResponse]:
        """
        Execute a problem solving operation with dual-format support.
        
        Supports three calling conventions:
        1. Legacy positional: execute_task(operation, parameters)
        2. Legacy dict: execute_task({'operation': ..., 'parameters': ...})
        3. Standardized: execute_task(AgentExecutionRequest)
        
        Operations:
        - diagnose_error: Diagnose an error and identify its category
        - get_solution: Get LLM-generated solution for an error
        - interpret_human_input: Interpret natural language input
        - format_for_agent: Format data for consumption by specific agent
        - generate_retry_parameters: Generate modified parameters for retry
        - create_task_output: Create task output from human input
        
        Args:
            *args: Variable positional arguments
            **kwargs: Variable keyword arguments
        
        Returns:
            Legacy dict OR AgentExecutionResponse based on input format
        """
        start_time = time.time()
        return_legacy = True
        operation = None
        parameters = None
        task_dict = None
        
        # Detect calling convention
        # Positional arguments (operation, parameters)
        if len(args) == 2:
            operation, parameters = args
            task_dict = {"operation": operation, "parameters": parameters}
            return_legacy = True
            logger.debug("Legacy positional call")
        
        # Single dict argument
        elif len(args) == 1 and isinstance(args[0], dict):
            task_dict = args[0]
            # Check if standardized request (has task_id and task_description)
            if "task_id" in task_dict and "task_description" in task_dict:
                return_legacy = False
                logger.debug(f"Standardized request call: task_id={task_dict.get('task_id')}")
            else:
                return_legacy = True
                logger.debug("Legacy dict call")
            operation = task_dict.get("operation")
            parameters = task_dict.get("parameters", {})
        
        # Keyword arguments (operation=..., parameters=...)
        elif "operation" in kwargs:
            operation = kwargs.get("operation")
            parameters = kwargs.get("parameters", {})
            task_dict = {"operation": operation, "parameters": parameters}
            return_legacy = True
            logger.debug("Legacy keyword call")
        
        else:
            raise InvalidParameterError(
                parameter_name="task",
                message="Invalid call to execute_task. Use one of:\n"
                "  - execute_task(operation, parameters)\n"
                "  - execute_task({'operation': ..., 'parameters': ...})\n"
                "  - execute_task(AgentExecutionRequest)"
            )
        
        try:
            task_id = task_dict.get("task_id", f"problem_solver_{int(time.time())}")  # type: ignore
            
            # Ensure parameters is not None
            if parameters is None:
                parameters = {}
            
            # Ensure operation is not None
            if operation is None:
                operation = "unknown"
            
            logger.info(f"Executing problem solver operation: {operation} (task_id={task_id})")
            
            # Execute the operation using existing methods
            if operation == "diagnose_error":
                result = self.diagnose_error(
                    error_message=parameters.get('error_message', ''),
                    task_context=parameters.get('task_context'),
                    agent_type=parameters.get('agent_type')
                )
            
            elif operation == "get_solution":
                result = self.get_solution(
                    error_message=parameters.get('error_message', ''),
                    task_context=parameters.get('task_context'),
                    agent_type=parameters.get('agent_type'),
                    available_data=parameters.get('available_data')
                )
            
            elif operation == "interpret_human_input":
                result = self.interpret_human_input(
                    human_input=parameters.get('human_input', ''),
                    target_format=parameters.get('target_format', 'json'),
                    task_context=parameters.get('task_context'),
                    expected_fields=parameters.get('expected_fields')
                )
            
            elif operation == "format_for_agent":
                result = self.format_for_agent(
                    data=parameters.get('data'),
                    agent_type=parameters.get('agent_type', 'generic'),
                    operation=parameters.get('target_operation')
                )
            
            elif operation == "generate_retry_parameters":
                result = self.generate_retry_parameters(
                    failed_task=parameters.get('failed_task', {}),
                    error_info=parameters.get('error_info', {}),
                    human_context=parameters.get('human_context')
                )
            
            elif operation == "create_task_output":
                result = self.create_task_output_from_human_input(
                    human_input=parameters.get('human_input', ''),
                    task_description=parameters.get('task_description', ''),
                    expected_output_type=parameters.get('expected_output_type')
                )
            
            else:
                result = {
                    "success": False,
                    "error": f"Unknown operation: {operation}",
                    "supported_operations": self.supported_operations
                }
            
            # Convert legacy result to standardized response
            standard_response = self._convert_to_standard_response(
                result,
                operation,
                task_id,
                start_time
            )
            
            # Publish completion event for event-driven workflows
            self._publish_completion_event(task_id, operation, standard_response)
            
            # Return in requested format
            if return_legacy:
                # Convert back to legacy format for backward compatibility
                return self._convert_to_legacy_response(standard_response)
            else:
                return standard_response
        
        except Exception as e:
            logger.error(f"Error executing ProblemSolver task: {e}", exc_info=True)
            
            # Create standardized error response
            error = exception_to_error_response(
                e,
                source=self.agent_name,
                task_id=task_dict.get("task_id", "unknown") if task_dict else "unknown"
            )
            
            error_response: AgentExecutionResponse = {
                "status": "failure",
                "success": False,
                "result": {},
                "artifacts": [],
                "execution_time_ms": int((time.time() - start_time) * 1000),
                "timestamp": datetime.now().isoformat(),
                "agent_name": self.agent_name,
                "operation": operation or "unknown",
                "blackboard_entries": [],
                "warnings": []
            }
            # Add error field separately to handle TypedDict
            error_response["error"] = error  # type: ignore
            
            if return_legacy:
                return self._convert_to_legacy_response(error_response)
            else:
                return error_response
    
    def _convert_to_standard_response(
        self,
        legacy_result: Dict[str, Any],
        operation: str,
        task_id: str,
        start_time: float
    ) -> AgentExecutionResponse:
        """Convert legacy result dict to standardized AgentExecutionResponse."""
        success = legacy_result.get("success", True)  # Default to True if not specified
        
        # Handle different result structures based on operation
        # Most operations return success implicitly if no error
        if "error" in legacy_result:
            success = False
        
        # Create blackboard entries for sharing data
        blackboard_entries = []
        
        # For solutions, share the solution data
        if operation == "get_solution" and success:
            blackboard_entries.append({
                "key": f"solution_{task_id}",
                "value": legacy_result,
                "scope": "workflow",
                "ttl_seconds": 3600
            })
        
        # For interpreted input, share the parsed data
        if operation == "interpret_human_input" and success and "parsed_data" in legacy_result:
            blackboard_entries.append({
                "key": f"interpreted_input_{task_id}",
                "value": legacy_result.get("parsed_data", {}),
                "scope": "workflow",
                "ttl_seconds": 3600
            })
        
        # For retry parameters, share them
        if operation == "generate_retry_parameters" and success:
            blackboard_entries.append({
                "key": f"retry_params_{task_id}",
                "value": legacy_result.get("modified_parameters", {}),
                "scope": "workflow",
                "ttl_seconds": 1800
            })
        
        # Build standardized response
        response: AgentExecutionResponse = {
            "status": "success" if success else "failure",
            "success": success,
            "result": legacy_result,
            "artifacts": [],  # Problem solver doesn't create file artifacts
            "execution_time_ms": int((time.time() - start_time) * 1000),
            "timestamp": datetime.now().isoformat(),
            "agent_name": self.agent_name,
            "operation": operation,
            "blackboard_entries": blackboard_entries,
            "warnings": []
        }
        
        # Add error field if present (handle TypedDict)
        if not success and "error" in legacy_result:
            response["error"] = create_error_response(  # type: ignore
                error_code="PROBLEM_SOLVER_001",
                error_type="execution_error",
                message=str(legacy_result.get("error", "Unknown error")),
                source=self.agent_name
            )
        
        return response
    
    def _convert_to_legacy_response(self, standard_response: AgentExecutionResponse) -> Dict[str, Any]:
        """Convert standardized response back to legacy format for backward compatibility."""
        legacy = standard_response["result"].copy() if isinstance(standard_response["result"], dict) else {}
        
        # Ensure success field is present
        if "success" not in legacy:
            legacy["success"] = standard_response["success"]
        
        # Add error if present (use .get() for NotRequired field)
        error = standard_response.get("error")  # type: ignore
        if error and "error" not in legacy:
            legacy["error"] = error["message"]  # type: ignore
        
        return legacy
    
    def _publish_completion_event(
        self,
        task_id: str,
        operation: str,
        response: AgentExecutionResponse
    ):
        """Publish task completion event for event-driven workflows."""
        try:
            # Choose event type based on operation
            event_type_map = {
                "diagnose_error": "error_diagnosed",
                "get_solution": "solution_generated",
                "interpret_human_input": "human_input_interpreted",
                "format_for_agent": "data_formatted",
                "generate_retry_parameters": "retry_params_generated",
                "create_task_output": "task_output_created"
            }
            
            event_type = event_type_map.get(operation, "problem_solver_completed")
            
            event = create_system_event(
                event_type=event_type,
                event_category="task_lifecycle",
                source_agent=self.agent_name,
                payload={
                    "task_id": task_id,
                    "operation": operation,
                    "success": response["success"],
                    "blackboard_keys": [entry["key"] for entry in response["blackboard_entries"]]
                }
            )
            self.event_bus.publish(event)
            logger.debug(f"Published completion event for task {task_id}")
        except Exception as e:
            logger.warning(f"Failed to publish completion event: {e}")
    
    
    def diagnose_error(
        self,
        error_message: str,
        task_context: Optional[Dict[str, Any]] = None,
        agent_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Diagnose an error and identify its category.
        
        Args:
            error_message: The error message to diagnose
            task_context: Optional context about the task that failed
            agent_type: Type of agent that encountered the error
            
        Returns:
            Diagnosis result with error category and initial suggestions
        """
        logger.info("=" * 80)
        logger.info("[PROBLEM SOLVER] Diagnosing error")
        logger.info("=" * 80)
        logger.info(f"[DIAGNOSE] Error message: {error_message[:200]}...")
        logger.info(f"[DIAGNOSE] Agent type: {agent_type}")
        
        error_lower = error_message.lower()
        
        # Identify error category
        error_category = "unknown"
        solution_prompt = "An unknown error occurred. Analyze the error and suggest solutions."
        
        for category, info in self.error_patterns.items():
            for pattern in info["patterns"]:
                if pattern in error_lower:
                    error_category = category
                    solution_prompt = info["solution_prompt"]
                    logger.info(f"[DIAGNOSE] Matched error category: {category}")
                    break
            if error_category != "unknown":
                break
        
        diagnosis = {
            "error_message": error_message,
            "error_category": error_category,
            "solution_prompt": solution_prompt,
            "agent_type": agent_type,
            "task_context": task_context,
            "diagnosed_at": datetime.now().isoformat()
        }
        
        logger.info(f"[DIAGNOSE] Category identified: {error_category}")
        logger.info("=" * 80)
        
        return diagnosis
    
    
    def get_solution(
        self,
        error_message: str,
        task_context: Optional[Dict[str, Any]] = None,
        agent_type: Optional[str] = None,
        available_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get an LLM-generated solution for an error.
        
        Args:
            error_message: The error message to solve
            task_context: Context about the failed task
            agent_type: Type of agent that failed
            available_data: Data available for constructing a solution
            
        Returns:
            Solution dictionary with suggested actions and parameters
        """
        logger.info("=" * 80)
        logger.info("[PROBLEM SOLVER] Generating solution via LLM")
        logger.info("=" * 80)
        
        if not self.llm_client:
            logger.warning("[SOLUTION] No LLM client available, returning template solution")
            return self._generate_template_solution(error_message, task_context, agent_type)
        
        # First diagnose the error
        diagnosis = self.diagnose_error(error_message, task_context, agent_type)
        
        # Build the solution prompt
        prompt = self._build_solution_prompt(diagnosis, available_data)
        
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            
            logger.info("[SOLUTION] Sending request to LLM...")
            
            system_prompt = """You are an expert problem solver for an AI agent system. 
Your task is to analyze errors and provide actionable solutions.

You must respond with ONLY valid JSON in this exact format:
{
  "solution_type": "retry|modify_params|skip|manual_input|alternative_approach",
  "explanation": "Brief explanation of what went wrong",
  "suggested_action": "Description of the recommended action",
  "modified_parameters": {
    // Any modified parameters that should be used for retry
  },
  "alternative_approaches": [
    "List of alternative approaches if the primary solution fails"
  ],
  "requires_human_input": true or false,
  "human_input_prompt": "If human input is needed, what to ask",
  "confidence": 0.0 to 1.0
}"""
            
            response = self.llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ])
            
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            logger.info("[SOLUTION] LLM response received")
            logger.debug(f"[SOLUTION] Response: {response_content[:500]}...")
            
            # Parse the solution
            solution = self._parse_solution_response(response_content)
            solution["diagnosis"] = diagnosis
            solution["generated_at"] = datetime.now().isoformat()
            
            logger.info(f"[SOLUTION] Solution type: {solution.get('solution_type')}")
            logger.info(f"[SOLUTION] Confidence: {solution.get('confidence')}")
            logger.info("=" * 80)
            
            return solution
            
        except Exception as e:
            logger.error(f"[SOLUTION] Error getting LLM solution: {str(e)}")
            return self._generate_template_solution(error_message, task_context, agent_type)
    
    
    def interpret_human_input(
        self,
        human_input: str,
        target_format: str,
        task_context: Optional[Dict[str, Any]] = None,
        expected_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Interpret and format human natural language input for programmatic use.
        
        Args:
            human_input: Raw human input text
            target_format: Desired output format (json, excel_params, etc.)
            task_context: Context about the task requiring this input
            expected_fields: List of expected fields to extract
            
        Returns:
            Formatted output suitable for agent consumption
        """
        logger.info("=" * 80)
        logger.info("[PROBLEM SOLVER] Interpreting human input")
        logger.info("=" * 80)
        logger.info(f"[INTERPRET] Input length: {len(human_input)} chars")
        logger.info(f"[INTERPRET] Target format: {target_format}")
        logger.info(f"[INTERPRET] Expected fields: {expected_fields}")
        
        if not self.llm_client:
            logger.warning("[INTERPRET] No LLM client, using rule-based parsing")
            return self._rule_based_parse(human_input, target_format, expected_fields)
        
        # Build interpretation prompt
        prompt = self._build_interpretation_prompt(
            human_input, target_format, task_context, expected_fields
        )
        
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            
            system_prompt = f"""You are an expert at interpreting natural language and converting it to structured data.
Your task is to extract information from human input and format it for use by AI agents.

Target output format: {target_format}

You must respond with ONLY valid JSON in this format:
{{
  "success": true or false,
  "parsed_data": {{
    // Extracted and formatted data matching the target format
  }},
  "confidence": 0.0 to 1.0,
  "missing_fields": ["list of any fields that couldn't be extracted"],
  "assumptions_made": ["list of any assumptions made during parsing"],
  "original_input": "the original input for reference"
}}"""
            
            logger.info("[INTERPRET] Sending request to LLM...")
            
            response = self.llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ])
            
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            logger.info("[INTERPRET] LLM response received")
            
            # Parse the interpretation
            result = self._parse_interpretation_response(response_content, target_format)
            result["target_format"] = target_format
            result["interpreted_at"] = datetime.now().isoformat()
            
            logger.info(f"[INTERPRET] Success: {result.get('success')}")
            logger.info(f"[INTERPRET] Confidence: {result.get('confidence')}")
            if result.get('missing_fields'):
                logger.warning(f"[INTERPRET] Missing fields: {result.get('missing_fields')}")
            logger.info("=" * 80)
            
            return result
            
        except Exception as e:
            logger.error(f"[INTERPRET] Error during interpretation: {str(e)}")
            return self._rule_based_parse(human_input, target_format, expected_fields)
    
    
    def format_for_agent(
        self,
        data: Any,
        agent_type: str,
        operation: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format data for consumption by a specific agent type.
        
        Args:
            data: Raw data to format
            agent_type: Target agent type (excel, web_search, pdf, etc.)
            operation: Optional specific operation
            
        Returns:
            Formatted parameters ready for agent consumption
        """
        logger.info(f"[FORMAT] Formatting data for {agent_type} agent")
        logger.info(f"[FORMAT] Operation: {operation}")
        
        formatters = {
            "excel": self._format_for_excel,
            "excel_agent": self._format_for_excel,
            "web_search": self._format_for_web_search,
            "web_search_agent": self._format_for_web_search,
            "pdf": self._format_for_pdf,
            "pdf_agent": self._format_for_pdf,
            "ocr": self._format_for_ocr,
            "ocr_agent": self._format_for_ocr,
            "code_interpreter": self._format_for_code_interpreter,
            "code_interpreter_agent": self._format_for_code_interpreter,
        }
        
        formatter = formatters.get(agent_type.lower())
        
        if formatter:
            result = formatter(data, operation)
            logger.info(f"[FORMAT] Formatted successfully for {agent_type}")
            return result
        else:
            logger.warning(f"[FORMAT] No specific formatter for {agent_type}, returning generic format")
            return self._format_generic(data, operation)
    
    
    def generate_retry_parameters(
        self,
        failed_task: Dict[str, Any],
        error_info: Dict[str, Any],
        human_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate modified parameters for retrying a failed task.
        
        Args:
            failed_task: The task that failed
            error_info: Information about the error
            human_context: Optional human-provided context for retry
            
        Returns:
            Modified parameters for retry attempt
        """
        logger.info("=" * 80)
        logger.info("[PROBLEM SOLVER] Generating retry parameters")
        logger.info("=" * 80)
        logger.info(f"[RETRY] Task ID: {failed_task.get('id')}")
        logger.info(f"[RETRY] Error: {str(error_info.get('error', ''))[:100]}")
        
        # Get solution first
        solution = self.get_solution(
            error_message=str(error_info.get('error', 'Unknown error')),
            task_context=failed_task,
            agent_type=error_info.get('agent_type'),
            available_data=error_info.get('available_data')
        )
        
        # Extract retry parameters
        retry_params = {
            "original_task": failed_task,
            "modified_parameters": solution.get('modified_parameters', {}),
            "retry_strategy": solution.get('solution_type', 'retry'),
            "human_context": human_context,
            "solution_explanation": solution.get('explanation'),
            "confidence": solution.get('confidence', 0.5),
            "generated_at": datetime.now().isoformat()
        }
        
        # If human context provided, interpret it
        if human_context:
            logger.info("[RETRY] Interpreting human context for retry...")
            agent_type = error_info.get('agent_type', 'generic')
            interpreted = self.interpret_human_input(
                human_context,
                target_format=f"{agent_type}_params",
                task_context=failed_task
            )
            
            if interpreted.get('success'):
                retry_params["modified_parameters"].update(
                    interpreted.get('parsed_data', {})
                )
                logger.info("[RETRY] Human context integrated into retry parameters")
        
        logger.info(f"[RETRY] Strategy: {retry_params['retry_strategy']}")
        logger.info(f"[RETRY] Confidence: {retry_params['confidence']}")
        logger.info("=" * 80)
        
        return retry_params
    
    
    def create_task_output_from_human_input(
        self,
        human_input: str,
        task_description: str,
        expected_output_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a complete task output from human input.
        
        This is used when a human provides the result for a failed task,
        formatting it properly so dependent tasks can use it.
        
        Args:
            human_input: The human-provided output
            task_description: Description of the task
            expected_output_type: Expected type of output
            
        Returns:
            Properly formatted task output
        """
        logger.info("=" * 80)
        logger.info("[PROBLEM SOLVER] Creating task output from human input")
        logger.info("=" * 80)
        logger.info(f"[TASK OUTPUT] Task: {task_description[:100]}...")
        logger.info(f"[TASK OUTPUT] Expected type: {expected_output_type}")
        
        if not self.llm_client:
            return self._create_simple_task_output(human_input, task_description)
        
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            
            system_prompt = """You are formatting human-provided task output for use by an AI agent system.
Analyze the task description and human input to create a properly structured output.

Respond with ONLY valid JSON:
{
  "success": true,
  "output": "formatted output string or object",
  "output_type": "text|json|data|file_reference|structured",
  "summary": "brief summary of what was provided",
  "key_data_points": ["list of important data points extracted"],
  "usable_by_agents": ["list of agent types that can use this output"],
  "metadata": {
    "human_provided": true,
    "confidence": 0.0 to 1.0,
    "completeness": "complete|partial|minimal"
  }
}"""
            
            prompt = f"""Task Description: {task_description}

Human-Provided Output:
{human_input}

Expected Output Type: {expected_output_type or 'auto-detect'}

Format this human input as a proper task output that other agents can consume."""
            
            response = self.llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ])
            
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            result = self._parse_json_response(response_content)
            result["original_input"] = human_input
            result["created_at"] = datetime.now().isoformat()
            
            logger.info(f"[TASK OUTPUT] Output type: {result.get('output_type')}")
            logger.info(f"[TASK OUTPUT] Usable by: {result.get('usable_by_agents')}")
            logger.info("=" * 80)
            
            return result
            
        except Exception as e:
            logger.error(f"[TASK OUTPUT] Error: {str(e)}")
            return self._create_simple_task_output(human_input, task_description)
    
    
    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================
    
    def _build_solution_prompt(
        self,
        diagnosis: Dict[str, Any],
        available_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for solution generation."""
        
        prompt_parts = [
            f"Error Category: {diagnosis['error_category']}",
            f"Error Message: {diagnosis['error_message']}",
            f"\n{diagnosis['solution_prompt']}"
        ]
        
        if diagnosis.get('agent_type'):
            prompt_parts.append(f"\nAgent Type: {diagnosis['agent_type']}")
        
        if diagnosis.get('task_context'):
            ctx = diagnosis['task_context']
            prompt_parts.append(f"\nTask Context:")
            prompt_parts.append(f"  - Description: {ctx.get('description', 'N/A')}")
            prompt_parts.append(f"  - Parameters: {json.dumps(ctx.get('parameters', {}), indent=2)}")
        
        if available_data:
            prompt_parts.append(f"\nAvailable Data for Solution:")
            prompt_parts.append(json.dumps(available_data, indent=2, default=str))
        
        prompt_parts.append("\nProvide a practical solution that can be implemented programmatically.")
        
        return "\n".join(prompt_parts)
    
    
    def _build_interpretation_prompt(
        self,
        human_input: str,
        target_format: str,
        task_context: Optional[Dict[str, Any]] = None,
        expected_fields: Optional[List[str]] = None
    ) -> str:
        """Build prompt for human input interpretation."""
        
        prompt_parts = [
            f"Human Input to Interpret:",
            f'"""{human_input}"""',
            f"\nTarget Format: {target_format}"
        ]
        
        if expected_fields:
            prompt_parts.append(f"\nExpected Fields: {', '.join(expected_fields)}")
        
        if task_context:
            prompt_parts.append(f"\nTask Context:")
            prompt_parts.append(f"  - Description: {task_context.get('description', 'N/A')}")
            if task_context.get('agent_type'):
                prompt_parts.append(f"  - Agent Type: {task_context.get('agent_type')}")
        
        # Add format-specific guidance
        format_guidance = self._get_format_guidance(target_format)
        if format_guidance:
            prompt_parts.append(f"\nFormat Guidance: {format_guidance}")
        
        prompt_parts.append("\nExtract and format the information from the human input.")
        
        return "\n".join(prompt_parts)
    
    
    def _get_format_guidance(self, target_format: str) -> str:
        """Get format-specific guidance for interpretation."""
        
        guidance = {
            "excel_params": "Extract: file_path, sheet_name, data (as rows), headers, formatting options",
            "web_search_params": "Extract: search query, number of results, date range, source preferences",
            "file_operation_params": "Extract: file_path, operation type, content, encoding",
            "json": "Convert to valid JSON structure preserving all information",
            "task_context": "Extract: objective, constraints, expected output, dependencies",
            "structured_text": "Organize into sections with headers and bullet points"
        }
        
        return guidance.get(target_format, "")
    
    
    def _parse_solution_response(self, response_content: str) -> Dict[str, Any]:
        """Parse LLM solution response into structured format."""
        
        try:
            return self._parse_json_response(response_content)
        except Exception as e:
            logger.warning(f"[PARSE] Failed to parse solution response: {str(e)}")
            return {
                "solution_type": "manual_input",
                "explanation": "Could not parse LLM response",
                "suggested_action": response_content[:500] if response_content else "No response",
                "requires_human_input": True,
                "confidence": 0.3
            }
    
    
    def _parse_interpretation_response(
        self,
        response_content: str,
        target_format: str
    ) -> Dict[str, Any]:
        """Parse LLM interpretation response."""
        
        try:
            return self._parse_json_response(response_content)
        except Exception as e:
            logger.warning(f"[PARSE] Failed to parse interpretation response: {str(e)}")
            return {
                "success": False,
                "parsed_data": {},
                "confidence": 0.0,
                "error": str(e),
                "raw_response": response_content[:500] if response_content else ""
            }
    
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling markdown formatting."""
        
        content = content.strip()
        
        # Remove markdown code blocks if present
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()
        elif content.startswith("```"):
            content = content.split("```")[1].split("```")[0].strip()
        
        return json.loads(content)
    
    
    def _generate_template_solution(
        self,
        error_message: str,
        task_context: Optional[Dict[str, Any]] = None,
        agent_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a template-based solution when LLM is unavailable."""
        
        diagnosis = self.diagnose_error(error_message, task_context, agent_type)
        
        # Template solutions based on error category
        template_solutions = {
            "file_not_found": {
                "solution_type": "modify_params",
                "explanation": "The specified file could not be found",
                "suggested_action": "Verify the file path or create the file first",
                "modified_parameters": {"create_if_missing": True},
                "requires_human_input": True,
                "human_input_prompt": "Please provide the correct file path",
                "confidence": 0.6
            },
            "invalid_path": {
                "solution_type": "modify_params",
                "explanation": "The path provided was invalid or empty",
                "suggested_action": "Provide a valid absolute or relative path",
                "modified_parameters": {},
                "requires_human_input": True,
                "human_input_prompt": "Please provide a valid file path",
                "confidence": 0.5
            },
            "sheet_not_found": {
                "solution_type": "modify_params",
                "explanation": "The Excel sheet name was not found",
                "suggested_action": "Use a different sheet name or create the sheet",
                "modified_parameters": {"create_sheet_if_missing": True},
                "requires_human_input": True,
                "human_input_prompt": "Please provide the correct sheet name",
                "confidence": 0.6
            },
            "api_error": {
                "solution_type": "retry",
                "explanation": "An API error occurred, possibly temporary",
                "suggested_action": "Wait and retry the request",
                "modified_parameters": {"retry_delay": 5},
                "requires_human_input": False,
                "confidence": 0.7
            },
            "unknown": {
                "solution_type": "manual_input",
                "explanation": "An unexpected error occurred",
                "suggested_action": "Review the error and provide manual input",
                "modified_parameters": {},
                "requires_human_input": True,
                "human_input_prompt": "Please review the error and provide guidance",
                "confidence": 0.3
            }
        }
        
        solution = template_solutions.get(
            diagnosis['error_category'],
            template_solutions['unknown']
        )
        solution["diagnosis"] = diagnosis
        solution["template_based"] = True
        
        return solution
    
    
    def _rule_based_parse(
        self,
        human_input: str,
        target_format: str,
        expected_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Rule-based parsing when LLM is unavailable."""
        
        result = {
            "success": True,
            "parsed_data": {},
            "confidence": 0.5,
            "rule_based": True,
            "original_input": human_input
        }
        
        # Basic parsing based on format
        if target_format in ["excel_params", "excel"]:
            result["parsed_data"] = self._parse_excel_params(human_input)
        elif target_format in ["web_search_params", "web_search"]:
            result["parsed_data"] = self._parse_search_params(human_input)
        elif target_format == "json":
            try:
                result["parsed_data"] = json.loads(human_input)
            except:
                result["parsed_data"] = {"raw_text": human_input}
        else:
            # Generic extraction
            result["parsed_data"] = self._extract_key_values(human_input)
        
        return result
    
    
    def _parse_excel_params(self, text: str) -> Dict[str, Any]:
        """Extract Excel-related parameters from text."""
        
        params = {}
        
        # Look for file path patterns
        path_match = re.search(r'(?:path|file|save to|in)\s*[:\s]*([^\n,]+\.xlsx?)', text, re.IGNORECASE)
        if path_match:
            params["file_path"] = path_match.group(1).strip()
        
        # Look for sheet name
        sheet_match = re.search(r'(?:sheet|tab|worksheet)\s*[:\s]*["\']?([^"\'\n,]+)["\']?', text, re.IGNORECASE)
        if sheet_match:
            params["sheet_name"] = sheet_match.group(1).strip()
        
        # Look for data/content
        if "headers" in text.lower():
            headers_match = re.search(r'headers?\s*[:\s]*\[([^\]]+)\]', text, re.IGNORECASE)
            if headers_match:
                headers = [h.strip().strip('"\'') for h in headers_match.group(1).split(',')]
                params["headers"] = headers
        
        # Default output folder
        if "file_path" not in params:
            params["folder_path"] = "output_folder"
            params["file_name"] = "output"
        
        return params
    
    
    def _parse_search_params(self, text: str) -> Dict[str, Any]:
        """Extract web search parameters from text."""
        
        params = {
            "query": text.strip(),
            "num_results": 5
        }
        
        # Look for specific number of results
        num_match = re.search(r'(?:top|first|get)\s*(\d+)', text, re.IGNORECASE)
        if num_match:
            params["num_results"] = int(num_match.group(1))
        
        # Extract the main search query
        query_match = re.search(r'(?:search for|find|look up|query)\s*[:\s]*["\']?([^"\'\n]+)["\']?', text, re.IGNORECASE)
        if query_match:
            params["query"] = query_match.group(1).strip()
        
        return params
    
    
    def _extract_key_values(self, text: str) -> Dict[str, Any]:
        """Extract key-value pairs from text."""
        
        data = {}
        
        # Look for "key: value" patterns
        kv_pattern = re.compile(r'(\w+)\s*[:\=]\s*([^\n]+)')
        for match in kv_pattern.finditer(text):
            key = match.group(1).lower().replace(' ', '_')
            value = match.group(2).strip()
            
            # Try to parse as JSON/number
            try:
                value = json.loads(value)
            except:
                try:
                    value = float(value) if '.' in value else int(value)
                except:
                    pass
            
            data[key] = value
        
        if not data:
            data["content"] = text
        
        return data
    
    
    def _create_simple_task_output(
        self,
        human_input: str,
        task_description: str
    ) -> Dict[str, Any]:
        """Create simple task output without LLM."""
        
        return {
            "success": True,
            "output": human_input,
            "output_type": "text",
            "summary": f"Human-provided output for: {task_description[:50]}...",
            "key_data_points": [human_input[:100]],
            "metadata": {
                "human_provided": True,
                "confidence": 0.7,
                "completeness": "unknown"
            },
            "original_input": human_input,
            "created_at": datetime.now().isoformat()
        }
    
    
    # ========================================================================
    # AGENT-SPECIFIC FORMATTERS
    # ========================================================================
    
    def _format_for_excel(self, data: Any, operation: Optional[str] = None) -> Dict[str, Any]:
        """Format data for Excel agent consumption."""
        
        params = {
            "operation": operation or "create",
            "auto_format": True
        }
        
        if isinstance(data, dict):
            params.update(data)
        elif isinstance(data, str):
            parsed = self._parse_excel_params(data)
            params.update(parsed)
        elif isinstance(data, list):
            # Assume it's data rows
            params["data"] = data
            if data and isinstance(data[0], list):
                params["headers"] = data[0]
        
        # Ensure required fields
        if "file_path" not in params and "output_path" not in params:
            if "folder_path" in params and "file_name" in params:
                from pathlib import Path
                params["output_path"] = str(
                    Path(params["folder_path"]) / f"{params['file_name']}.xlsx"
                )
        
        return params
    
    
    def _format_for_web_search(self, data: Any, operation: Optional[str] = None) -> Dict[str, Any]:
        """Format data for Web Search agent consumption."""
        
        params = {
            "operation": operation or "search",
            "num_results": 5
        }
        
        if isinstance(data, dict):
            params.update(data)
        elif isinstance(data, str):
            parsed = self._parse_search_params(data)
            params.update(parsed)
        
        return params
    
    
    def _format_for_pdf(self, data: Any, operation: Optional[str] = None) -> Dict[str, Any]:
        """Format data for PDF agent consumption."""
        
        params = {
            "operation": operation or "read"
        }
        
        if isinstance(data, dict):
            params.update(data)
        elif isinstance(data, str):
            # Check if it looks like a file path
            if data.endswith('.pdf') or '/' in data or '\\' in data:
                params["file_path"] = data
            else:
                params["content"] = data
        
        return params
    
    
    def _format_for_ocr(self, data: Any, operation: Optional[str] = None) -> Dict[str, Any]:
        """Format data for OCR agent consumption."""
        
        params: Dict[str, Any] = {
            "operation": operation or "extract_text"
        }
        
        if isinstance(data, dict):
            params.update(data)
        elif isinstance(data, str):
            params["file_path"] = data
        elif isinstance(data, list):
            params["file_paths"] = data
        
        return params
    
    
    def _format_for_code_interpreter(self, data: Any, operation: Optional[str] = None) -> Dict[str, Any]:
        """Format data for Code Interpreter agent consumption."""
        
        params: Dict[str, Any] = {
            "operation": operation or "execute"
        }
        
        if isinstance(data, dict):
            params.update(data)
        elif isinstance(data, str):
            # Check if it's code or a request
            if any(keyword in data for keyword in ['def ', 'import ', 'class ', '=']):
                params["code"] = data
            else:
                params["request"] = data
        
        return params
    
    
    def _format_generic(self, data: Any, operation: Optional[str] = None) -> Dict[str, Any]:
        """Generic data formatting."""
        
        params: Dict[str, Any] = {
            "operation": operation or "process"
        }
        
        if isinstance(data, dict):
            params.update(data)
        elif isinstance(data, str):
            params["input"] = data
        elif isinstance(data, list):
            params["data"] = data
        else:
            params["data"] = str(data)
        
        return params
    
    
    # ========================================================================
    # CONVENIENCE METHODS
    # ========================================================================
    
    def quick_solve(self, error: str, context: str = "") -> str:
        """
        Quick one-liner solution for an error.
        
        Args:
            error: Error message
            context: Optional context
            
        Returns:
            Solution string
        """
        solution = self.get_solution(error, {"context": context} if context else None)
        return solution.get('suggested_action', 'No solution found')
    
    
    def quick_format(self, human_input: str, agent: str) -> Dict[str, Any]:
        """
        Quickly format human input for an agent.
        
        Args:
            human_input: Human input text
            agent: Target agent name
            
        Returns:
            Formatted parameters
        """
        interpreted = self.interpret_human_input(human_input, f"{agent}_params")
        return self.format_for_agent(interpreted.get('parsed_data', {}), agent)
