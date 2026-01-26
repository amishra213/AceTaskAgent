"""
Mermaid Diagram Sub-Agent for creating flow diagrams and process visualizations.

Capabilities:
- Create flowcharts for process flows
- Create sequence diagrams
- Create class diagrams
- Create state diagrams
- Create Gantt charts
- Generate Mermaid markdown syntax
- Save diagrams as .mmd or .md files

Migration Status: Week 7 Day 1 - Dual Format Support
- Supports both legacy dict and standardized AgentExecutionRequest/Response
- Maintains 100% backward compatibility
- Publishes SystemEvent on completion for event-driven workflows
"""

from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from datetime import datetime
import json
import time

from task_manager.utils.logger import get_logger

# Import standardized schemas and utilities
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
    wrap_exception
)
from task_manager.core.event_bus import get_event_bus

logger = get_logger(__name__)


class MermaidAgent:
    """
    Sub-agent for creating Mermaid diagrams and visualizations.
    
    This agent handles all Mermaid diagram tasks:
    - Creating flowcharts for process flows
    - Creating sequence diagrams
    - Creating class diagrams
    - Creating state diagrams
    - Creating Gantt charts
    - Generating proper Mermaid syntax
    """
    
    def __init__(self):
        """Initialize Mermaid Agent with dual-format support."""
        self.agent_name = "mermaid_agent"
        self.supported_operations = [
            "create_flowchart",
            "create_sequence",
            "create_class_diagram",
            "create_state_diagram",
            "create_gantt",
            "create_custom"
        ]
        
        # Initialize event bus for event-driven workflows
        self.event_bus = get_event_bus()
        
        logger.info("Mermaid Agent initialized with dual-format support")
    
    
    def _determine_output_path(
        self,
        parameters: Dict[str, Any],
        default_extension: str,
        operation: str
    ) -> str:
        """
        Determine output path from various parameter combinations.
        
        Supports multiple ways to specify output location:
        1. Direct: output_path or file_path
        2. Combined: folder_path + file_name
        3. Template-based: template_name (uses output_folder)
        4. Title-based: title (uses output_folder)
        
        Args:
            parameters: Task parameters dictionary
            default_extension: File extension to use (e.g., 'md', 'mmd')
            operation: Operation name for logging
        
        Returns:
            Resolved output path as string
        """
        output_path_raw = parameters.get('output_path')
        file_path_raw = parameters.get('file_path')
        folder_path = parameters.get('folder_path')
        file_name = parameters.get('file_name')
        template_name = parameters.get('template_name')
        title = parameters.get('title')
        
        logger.info(f"[MERMAID AGENT] Raw output_path: '{output_path_raw}'")
        logger.info(f"[MERMAID AGENT] Raw file_path: '{file_path_raw}'")
        logger.info(f"[MERMAID AGENT] Folder path: '{folder_path}'")
        logger.info(f"[MERMAID AGENT] File name: '{file_name}'")
        logger.info(f"[MERMAID AGENT] Template name: '{template_name}'")
        logger.info(f"[MERMAID AGENT] Title: '{title}'")
        
        # Build output path from various sources
        output_path = output_path_raw or file_path_raw
        
        if not output_path and folder_path and file_name:
            # Construct the path from folder and filename
            # Add extension if not present
            if not file_name.endswith(f'.{default_extension}'):
                file_name = f"{file_name}.{default_extension}"
            output_path = str(Path(folder_path) / file_name)
            logger.info(f"[MERMAID AGENT] Constructed output_path from folder_path + file_name: '{output_path}'")
        
        elif not output_path and folder_path and template_name:
            # Use template name with folder
            safe_name = template_name.replace(' ', '_').replace('Template', '').strip('_')
            output_path = str(Path(folder_path) / f"{safe_name}.{default_extension}")
            logger.info(f"[MERMAID AGENT] Constructed output_path from folder_path + template_name: '{output_path}'")
        
        elif not output_path and folder_path and title:
            # Use title with folder
            safe_name = title.replace(' ', '_')
            output_path = str(Path(folder_path) / f"{safe_name}.{default_extension}")
            logger.info(f"[MERMAID AGENT] Constructed output_path from folder_path + title: '{output_path}'")
        
        elif not output_path and template_name:
            # Use template name in output_folder
            safe_name = template_name.replace(' ', '_').replace('Template', '').strip('_')
            output_path = str(Path('output_folder') / f"{safe_name}.{default_extension}")
            logger.warning(f"[MERMAID AGENT] No folder specified, using output_folder: '{output_path}'")
        
        elif not output_path and title:
            # Use title in output_folder
            safe_name = title.replace(' ', '_')
            output_path = str(Path('output_folder') / f"{safe_name}.{default_extension}")
            logger.warning(f"[MERMAID AGENT] No folder specified, using output_folder: '{output_path}'")
        
        elif not output_path:
            # Last resort: use operation name with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = str(Path('output_folder') / f"{operation}_{timestamp}.{default_extension}")
            logger.warning(f"[MERMAID AGENT] No path info provided, using timestamped name: '{output_path}'")
        
        logger.info(f"[MERMAID AGENT] Final output_path for {operation}: '{output_path}'")
        return output_path
    
    
    def create_flowchart(
        self,
        output_path: str,
        title: str,
        nodes: List[Dict[str, Any]],
        connections: List[Dict[str, Any]],
        direction: str = "TD"
    ) -> Dict[str, Any]:
        """
        Create a Mermaid flowchart diagram.
        
        Args:
            output_path: Path where the diagram file will be created
            title: Title of the flowchart
            nodes: List of nodes [{'id': 'A', 'text': 'Start', 'shape': 'rounded'}]
                   Shapes: 'rectangle', 'rounded', 'stadium', 'subroutine', 'cylinder', 
                          'circle', 'asymmetric', 'rhombus', 'hexagon', 'trapezoid'
            connections: List of connections [{'from': 'A', 'to': 'B', 'text': 'Yes'}]
            direction: Flow direction ('TD'=top-down, 'LR'=left-right, 'BT'=bottom-top, 'RL'=right-left)
        
        Returns:
            Dictionary with operation result
        """
        try:
            output_path_obj = Path(output_path)
            
            # Ensure parent directory exists
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Creating Mermaid flowchart: {output_path}")
            
            # Build Mermaid syntax
            mermaid_code = [f"flowchart {direction}"]
            
            # Add title as a comment
            if title:
                mermaid_code.insert(0, f"---")
                mermaid_code.insert(1, f"title: {title}")
                mermaid_code.insert(2, f"---")
                mermaid_code.insert(3, "")
            
            # Add nodes with shapes
            shape_syntax = {
                'rectangle': ('[', ']'),
                'rounded': ('([', '])'),
                'stadium': ('([', '])'),
                'subroutine': ('[[', ']]'),
                'cylinder': ('[(', ')]'),
                'circle': ('((', '))'),
                'asymmetric': ('>', ']'),
                'rhombus': ('{', '}'),
                'hexagon': ('{{', '}}'),
                'trapezoid': ('[/', '\\]')
            }
            
            for node in nodes:
                node_id = node.get('id', '')
                node_text = node.get('text', node_id)
                node_shape = node.get('shape', 'rectangle')
                
                shape_start, shape_end = shape_syntax.get(node_shape, ('[', ']'))
                mermaid_code.append(f"    {node_id}{shape_start}\"{node_text}\"{shape_end}")
            
            # Add connections
            for conn in connections:
                from_node = conn.get('from', '')
                to_node = conn.get('to', '')
                conn_text = conn.get('text', '')
                conn_type = conn.get('type', 'arrow')  # arrow, dotted, thick
                
                if conn_type == 'dotted':
                    arrow = '-.->|' if conn_text else '-..->'
                elif conn_type == 'thick':
                    arrow = '==>|' if conn_text else '==>'
                else:  # arrow
                    arrow = '-->|' if conn_text else '-->'
                
                if conn_text:
                    mermaid_code.append(f"    {from_node} {arrow}\"{conn_text}\"{to_node}")
                else:
                    mermaid_code.append(f"    {from_node} {arrow} {to_node}")
            
            # Join and save
            mermaid_text = "\n".join(mermaid_code)
            
            # Determine file format
            if output_path_obj.suffix.lower() == '.md':
                content = f"# {title}\n\n```mermaid\n{mermaid_text}\n```\n"
            else:
                content = mermaid_text
            
            with open(output_path_obj, 'w', encoding='utf-8') as f:
                f.write(content)
            
            file_size = output_path_obj.stat().st_size
            
            logger.info(f"Flowchart created successfully: {output_path} ({file_size} bytes)")
            
            return {
                "success": True,
                "file": str(output_path),
                "output_path": str(output_path),
                "diagram_type": "flowchart",
                "size_bytes": file_size,
                "nodes_count": len(nodes),
                "connections_count": len(connections),
                "direction": direction,
                "mermaid_code": mermaid_text,
                "created_at": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error creating flowchart: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file": str(output_path)
            }
    
    
    def create_sequence(
        self,
        output_path: str,
        title: str,
        participants: List[str],
        messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a Mermaid sequence diagram.
        
        Args:
            output_path: Path where the diagram file will be created
            title: Title of the sequence diagram
            participants: List of participant names
            messages: List of messages [{'from': 'A', 'to': 'B', 'text': 'Hello', 'type': 'sync'}]
                     Types: 'sync', 'async', 'return', 'note'
        
        Returns:
            Dictionary with operation result
        """
        try:
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Creating Mermaid sequence diagram: {output_path}")
            
            mermaid_code = ["sequenceDiagram"]
            
            if title:
                mermaid_code.insert(0, f"---")
                mermaid_code.insert(1, f"title: {title}")
                mermaid_code.insert(2, f"---")
                mermaid_code.insert(3, "")
            
            # Add participants
            for participant in participants:
                mermaid_code.append(f"    participant {participant}")
            
            # Add messages
            for msg in messages:
                msg_from = msg.get('from', '')
                msg_to = msg.get('to', '')
                msg_text = msg.get('text', '')
                msg_type = msg.get('type', 'sync')
                
                if msg_type == 'note':
                    note_pos = msg.get('position', 'right of')
                    mermaid_code.append(f"    Note {note_pos} {msg_to}: {msg_text}")
                elif msg_type == 'async':
                    mermaid_code.append(f"    {msg_from}--){msg_to}: {msg_text}")
                elif msg_type == 'return':
                    mermaid_code.append(f"    {msg_from}-->>+{msg_to}: {msg_text}")
                else:  # sync
                    mermaid_code.append(f"    {msg_from}->>{msg_to}: {msg_text}")
            
            mermaid_text = "\n".join(mermaid_code)
            
            if output_path_obj.suffix.lower() == '.md':
                content = f"# {title}\n\n```mermaid\n{mermaid_text}\n```\n"
            else:
                content = mermaid_text
            
            with open(output_path_obj, 'w', encoding='utf-8') as f:
                f.write(content)
            
            file_size = output_path_obj.stat().st_size
            
            logger.info(f"Sequence diagram created successfully: {output_path}")
            
            return {
                "success": True,
                "file": str(output_path),
                "output_path": str(output_path),
                "diagram_type": "sequence",
                "size_bytes": file_size,
                "participants_count": len(participants),
                "messages_count": len(messages),
                "mermaid_code": mermaid_text,
                "created_at": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error creating sequence diagram: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file": str(output_path)
            }
    
    
    def create_state_diagram(
        self,
        output_path: str,
        title: str,
        states: List[Dict[str, Any]],
        transitions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a Mermaid state diagram.
        
        Args:
            output_path: Path where the diagram file will be created
            title: Title of the state diagram
            states: List of states [{'id': 'Idle', 'description': 'System idle'}]
            transitions: List of transitions [{'from': 'Idle', 'to': 'Active', 'event': 'start'}]
        
        Returns:
            Dictionary with operation result
        """
        try:
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Creating Mermaid state diagram: {output_path}")
            
            mermaid_code = ["stateDiagram-v2"]
            
            if title:
                mermaid_code.insert(0, f"---")
                mermaid_code.insert(1, f"title: {title}")
                mermaid_code.insert(2, f"---")
                mermaid_code.insert(3, "")
            
            # Add states with descriptions
            for state in states:
                state_id = state.get('id', '')
                state_desc = state.get('description', '')
                if state_desc:
                    mermaid_code.append(f"    {state_id}: {state_desc}")
            
            # Add transitions
            for trans in transitions:
                from_state = trans.get('from', '')
                to_state = trans.get('to', '')
                event = trans.get('event', '')
                
                if from_state == '[*]' or to_state == '[*]':
                    # Start or end state
                    mermaid_code.append(f"    {from_state} --> {to_state}")
                elif event:
                    mermaid_code.append(f"    {from_state} --> {to_state}: {event}")
                else:
                    mermaid_code.append(f"    {from_state} --> {to_state}")
            
            mermaid_text = "\n".join(mermaid_code)
            
            if output_path_obj.suffix.lower() == '.md':
                content = f"# {title}\n\n```mermaid\n{mermaid_text}\n```\n"
            else:
                content = mermaid_text
            
            with open(output_path_obj, 'w', encoding='utf-8') as f:
                f.write(content)
            
            file_size = output_path_obj.stat().st_size
            
            logger.info(f"State diagram created successfully: {output_path}")
            
            return {
                "success": True,
                "file": str(output_path),
                "output_path": str(output_path),
                "diagram_type": "state",
                "size_bytes": file_size,
                "states_count": len(states),
                "transitions_count": len(transitions),
                "mermaid_code": mermaid_text,
                "created_at": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error creating state diagram: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file": str(output_path)
            }
    
    
    def create_gantt(
        self,
        output_path: str,
        title: str,
        sections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a Mermaid Gantt chart.
        
        Args:
            output_path: Path where the diagram file will be created
            title: Title of the Gantt chart
            sections: List of sections with tasks
                     [{'name': 'Phase 1', 'tasks': [{'name': 'Task 1', 'status': 'done', 'start': '2024-01-01', 'duration': '5d'}]}]
        
        Returns:
            Dictionary with operation result
        """
        try:
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Creating Mermaid Gantt chart: {output_path}")
            
            mermaid_code = ["gantt"]
            
            if title:
                mermaid_code.append(f"    title {title}")
            
            mermaid_code.append(f"    dateFormat YYYY-MM-DD")
            
            task_count = 0
            for section in sections:
                section_name = section.get('name', 'Section')
                mermaid_code.append(f"    section {section_name}")
                
                tasks = section.get('tasks', [])
                for task in tasks:
                    task_name = task.get('name', 'Task')
                    task_status = task.get('status', 'active')  # done, active, crit
                    task_start = task.get('start', '')
                    task_duration = task.get('duration', '1d')
                    task_after = task.get('after', '')
                    
                    # Build task line
                    status_tag = f"{task_status}, " if task_status else ""
                    
                    if task_after:
                        mermaid_code.append(f"    {task_name}    :{status_tag}after {task_after}, {task_duration}")
                    elif task_start:
                        mermaid_code.append(f"    {task_name}    :{status_tag}{task_start}, {task_duration}")
                    else:
                        mermaid_code.append(f"    {task_name}    :{status_tag}{task_duration}")
                    
                    task_count += 1
            
            mermaid_text = "\n".join(mermaid_code)
            
            if output_path_obj.suffix.lower() == '.md':
                content = f"# {title}\n\n```mermaid\n{mermaid_text}\n```\n"
            else:
                content = mermaid_text
            
            with open(output_path_obj, 'w', encoding='utf-8') as f:
                f.write(content)
            
            file_size = output_path_obj.stat().st_size
            
            logger.info(f"Gantt chart created successfully: {output_path}")
            
            return {
                "success": True,
                "file": str(output_path),
                "output_path": str(output_path),
                "diagram_type": "gantt",
                "size_bytes": file_size,
                "sections_count": len(sections),
                "tasks_count": task_count,
                "mermaid_code": mermaid_text,
                "created_at": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error creating Gantt chart: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file": str(output_path)
            }
    
    
    def create_custom(
        self,
        output_path: str,
        title: Optional[str],
        mermaid_code: str
    ) -> Dict[str, Any]:
        """
        Create a custom Mermaid diagram from provided code.
        
        Args:
            output_path: Path where the diagram file will be created
            title: Optional title
            mermaid_code: Raw Mermaid syntax code
        
        Returns:
            Dictionary with operation result
        """
        try:
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Creating custom Mermaid diagram: {output_path}")
            
            if output_path_obj.suffix.lower() == '.md':
                content = f"# {title}\n\n```mermaid\n{mermaid_code}\n```\n" if title else f"```mermaid\n{mermaid_code}\n```\n"
            else:
                content = mermaid_code
            
            with open(output_path_obj, 'w', encoding='utf-8') as f:
                f.write(content)
            
            file_size = output_path_obj.stat().st_size
            
            logger.info(f"Custom diagram created successfully: {output_path}")
            
            return {
                "success": True,
                "file": str(output_path),
                "output_path": str(output_path),
                "diagram_type": "custom",
                "size_bytes": file_size,
                "mermaid_code": mermaid_code,
                "created_at": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error creating custom diagram: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file": str(output_path)
            }
    
    
    def execute_task(
        self,
        *args: Any,
        **kwargs: Any
    ) -> Union[Dict[str, Any], AgentExecutionResponse]:
        """
        Execute a Mermaid diagram operation with dual-format support.
        
        Supports three calling conventions:
        1. Legacy positional: execute_task(operation, parameters)
        2. Legacy dict: execute_task({'operation': ..., 'parameters': ...})
        3. Standardized: execute_task(AgentExecutionRequest)
        
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
        if len(args) == 2:
            operation, parameters = args
            task_dict = {"operation": operation, "parameters": parameters}
            return_legacy = True
            logger.debug("Legacy positional call")
        
        elif len(args) == 1 and isinstance(args[0], dict):
            task_dict = args[0]
            if "task_id" in task_dict and "task_description" in task_dict:
                return_legacy = False
                logger.debug(f"Standardized request call: task_id={task_dict.get('task_id')}")
            else:
                return_legacy = True
                logger.debug("Legacy dict call")
            operation = task_dict.get("operation")
            parameters = task_dict.get("parameters", {})
        
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
            task_id = task_dict.get("task_id", f"mermaid_{int(time.time())}")
            
            if parameters is None:
                parameters = {}
            
            if operation is None:
                operation = "unknown"
            
            logger.info("=" * 80)
            logger.info(f"[MERMAID AGENT] Starting execution")
            logger.info("=" * 80)
            logger.info(f"[MERMAID AGENT] Operation: {operation}")
            logger.info(f"[MERMAID AGENT] Task ID: {task_id}")
            logger.info(f"[MERMAID AGENT] Return format: {'legacy' if return_legacy else 'standardized'}")
            logger.info(f"[MERMAID AGENT] Parameters: {json.dumps(parameters, indent=2, default=str)}")
            
            # Execute the operation
            if operation == "create_flowchart":
                output_path = self._determine_output_path(parameters, 'md', operation)
                result = self.create_flowchart(
                    output_path=output_path,
                    title=parameters.get('title', ''),
                    nodes=parameters.get('nodes', []),
                    connections=parameters.get('connections', []),
                    direction=parameters.get('direction', 'TD')
                )
            
            elif operation == "create_sequence":
                output_path = self._determine_output_path(parameters, 'md', operation)
                result = self.create_sequence(
                    output_path=output_path,
                    title=parameters.get('title', ''),
                    participants=parameters.get('participants', []),
                    messages=parameters.get('messages', [])
                )
            
            elif operation == "create_state_diagram":
                output_path = self._determine_output_path(parameters, 'md', operation)
                result = self.create_state_diagram(
                    output_path=output_path,
                    title=parameters.get('title', ''),
                    states=parameters.get('states', []),
                    transitions=parameters.get('transitions', [])
                )
            
            elif operation == "create_gantt":
                output_path = self._determine_output_path(parameters, 'md', operation)
                result = self.create_gantt(
                    output_path=output_path,
                    title=parameters.get('title', ''),
                    sections=parameters.get('sections', [])
                )
            
            elif operation == "create_custom":
                output_path = self._determine_output_path(parameters, 'md', operation)
                result = self.create_custom(
                    output_path=output_path,
                    title=parameters.get('title'),
                    mermaid_code=parameters.get('mermaid_code', '')
                )
            
            else:
                result = {
                    "success": False,
                    "error": f"Unsupported operation: {operation}",
                    "supported_operations": self.supported_operations
                }
            
            logger.info(f"[MERMAID AGENT] Operation result: success={result.get('success')}")
            
            # Convert to standardized format if needed
            if return_legacy:
                response = result
            else:
                response = self._convert_to_standard_response(result, operation, task_id, start_time)
                self._publish_completion_event(task_id, operation, response)
            
            return response
        
        except Exception as e:
            logger.error(f"Error in execute_task: {str(e)}", exc_info=True)
            
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
        success = legacy_result.get("success", False)
        
        # Extract artifacts from result
        artifacts = []
        output_file = legacy_result.get("file") or legacy_result.get("output_path")
        if output_file and success:
            file_path = Path(output_file)
            if file_path.exists():
                file_ext = file_path.suffix.lower()
                artifacts.append({
                    "type": file_ext[1:] if file_ext else "mmd",
                    "path": str(output_file),
                    "size_bytes": file_path.stat().st_size,
                    "description": f"Mermaid diagram from {operation} operation"
                })
        
        # Create blackboard entries for sharing data
        blackboard_entries = []
        if success and "mermaid_code" in legacy_result:
            blackboard_entries.append({
                "key": f"mermaid_code_{task_id}",
                "value": legacy_result.get("mermaid_code", ""),
                "scope": "workflow",
                "ttl_seconds": 3600
            })
        
        # Build standardized response
        response: AgentExecutionResponse = {
            "status": "success" if success else "failure",
            "success": success,
            "result": {
                k: v for k, v in legacy_result.items()
                if k not in ["success", "file", "output_path"]
            },
            "artifacts": artifacts,
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
                error_code="MERMAID_001",
                error_type="execution_error",
                message=legacy_result.get("error", "Unknown error"),
                source=self.agent_name
            )
        
        return response
    
    
    def _convert_to_legacy_response(self, standard_response: AgentExecutionResponse) -> Dict[str, Any]:
        """Convert standardized response back to legacy format for backward compatibility."""
        legacy = {
            "success": standard_response["success"],
            "result": standard_response["result"]
        }
        
        if standard_response["artifacts"]:
            legacy["file"] = standard_response["artifacts"][0]["path"]
            legacy["output_path"] = standard_response["artifacts"][0]["path"]
        
        # Add error if present (use .get() for NotRequired field)
        error = standard_response.get("error")  # type: ignore
        if error:
            legacy["error"] = error["message"]  # type: ignore
        
        if isinstance(standard_response["result"], dict):
            for key, value in standard_response["result"].items():
                if key not in legacy:
                    legacy[key] = value
        
        return legacy
    
    
    def _publish_completion_event(
        self,
        task_id: str,
        operation: str,
        response: AgentExecutionResponse
    ):
        """Publish task completion event for event-driven workflows."""
        try:
            event = create_system_event(
                event_type="mermaid_diagram_created",
                event_category="task_lifecycle",
                source_agent=self.agent_name,
                payload={
                    "task_id": task_id,
                    "operation": operation,
                    "success": response["success"],
                    "artifacts": response["artifacts"],
                    "blackboard_keys": [entry["key"] for entry in response["blackboard_entries"]]
                }
            )
            self.event_bus.publish(event)
            logger.debug(f"Published completion event for task {task_id}")
        except Exception as e:
            logger.warning(f"Failed to publish completion event: {e}")
