"""
Code Interpreter Sub-Agent for executing Python code and data analysis.

Capabilities:
- Parse natural language analysis requests
- Generate Python code using Pandas, Numpy, Matplotlib
- Execute code in isolated subprocess
- Capture output and generated charts/images
- Post results to blackboard for workflow integration

Migration Status:
- Week 8 Day 1: ✅ COMPLETED - Legacy support removed, standardized-only interface
"""

from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import subprocess
import tempfile
import json
import re
import os
import time

from task_manager.utils.logger import get_logger
from task_manager.models.messages import (
    AgentExecutionRequest,
    AgentExecutionResponse,
    create_system_event,
    create_error_response
)
from task_manager.utils import (
    auto_convert_response,
    validate_agent_execution_response,
    exception_to_error_response
)
from task_manager.core.event_bus import get_event_bus

logger = get_logger(__name__)


class CodeInterpreterAgent:
    """
    Sub-agent for executing Python code and data analysis.
    
    This agent handles:
    - Parsing natural language analysis requests
    - Generating Python code with LLM
    - Executing code in isolated subprocess
    - Capturing output and generated visualizations
    - Posting results to blackboard
    """
    
    def __init__(self, llm=None):
        """
        Initialize Code Interpreter Agent.
        
        Args:
            llm: Optional LLM instance for code generation
        """
        self.agent_name = "code_interpreter_agent"
        self.event_bus = get_event_bus()
        self.supported_operations = [
            "execute_analysis",
            "generate_code",
            "execute_code",
            "analyze_data"
        ]
        self.llm = llm
        logger.info(f"[{self.agent_name}] Initialized with standardized-only interface")
        self._check_dependencies()
    
    
    def _check_dependencies(self):
        """Check if required data analysis libraries are installed."""
        self.available_libraries = []
        
        # Check for Pandas
        try:
            import pandas
            self.available_libraries.append("pandas")
            logger.debug("Pandas available")
        except ImportError:
            logger.warning("Pandas not available - install with: pip install pandas")
        
        # Check for Numpy
        try:
            import numpy
            self.available_libraries.append("numpy")
            logger.debug("Numpy available")
        except ImportError:
            logger.warning("Numpy not available - install with: pip install numpy")
        
        # Check for Matplotlib
        try:
            import matplotlib
            self.available_libraries.append("matplotlib")
            logger.debug("Matplotlib available")
        except ImportError:
            logger.warning("Matplotlib not available - install with: pip install matplotlib")
        
        # Check for Seaborn (optional)
        try:
            import seaborn
            self.available_libraries.append("seaborn")
            logger.debug("Seaborn available")
        except ImportError:
            logger.debug("Seaborn not available")
        
        # Check for Scikit-learn (optional)
        try:
            import sklearn as _sklearn  # type: ignore # noqa: F401
            self.available_libraries.append("scikit-learn")
            logger.debug("Scikit-learn available")
        except ImportError:
            logger.debug("Scikit-learn not available")
    
    
    def execute_analysis(
        self,
        request: str,
        data_context: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a natural language analysis request by generating and running Python code.
        
        Args:
            request: Natural language description of the analysis to perform
            data_context: Optional context with data to analyze (e.g., file paths, data structures)
            output_dir: Directory for output files/charts (default: temp directory)
        
        Returns:
            Dictionary with execution results, output, and generated files
        """
        try:
            if not self.llm:
                return {
                    "success": False,
                    "error": "LLM not provided - code generation not available",
                    "request": request
                }
            
            # Create output directory
            if output_dir is None:
                output_dir = tempfile.mkdtemp(prefix="code_interpreter_")
            else:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Executing analysis request: {request[:100]}...")
            
            # Step 1: Generate Python code from natural language request
            code = self._generate_code(request, data_context, output_dir)
            
            if not code.get("success"):
                return code
            
            generated_code = code.get("code", "")
            
            # Step 2: Execute the generated code
            execution_result = self._execute_code(
                generated_code,
                data_context=data_context,
                output_dir=output_dir
            )
            
            if not execution_result.get("success"):
                return execution_result
            
            # Step 3: Collect generated files (charts, images)
            generated_files = self._collect_generated_files(output_dir)
            
            # Step 4: Compile results
            result = {
                "success": True,
                "request": request,
                "generated_code": generated_code,
                "output": execution_result.get("output", ""),
                "error": execution_result.get("error"),
                "execution_time": execution_result.get("execution_time", 0),
                "generated_files": generated_files,
                "output_directory": output_dir,
                "timestamp": datetime.now().isoformat(),
                "libraries_used": self.available_libraries
            }
            
            logger.info(f"Analysis completed with {len(generated_files)} generated files")
            return result
        
        except Exception as e:
            logger.error(f"Error in execute_analysis: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "request": request
            }
    
    
    def _generate_code(
        self,
        request: str,
        data_context: Optional[Dict[str, Any]] = None,
        output_dir: str = ""
    ) -> Dict[str, Any]:
        """
        Generate Python code from natural language request using LLM.
        
        Args:
            request: Natural language description
            data_context: Optional context about available data
            output_dir: Directory for output files
        
        Returns:
            Dictionary with generated code or error
        """
        try:
            # Build prompt for code generation
            prompt = self._build_code_generation_prompt(request, data_context, output_dir)
            
            # Invoke LLM to generate code
            from langchain_core.messages import HumanMessage, SystemMessage
            
            system_message = SystemMessage(content="""You are a Python code generation expert. 
Generate high-quality, executable Python code that accomplishes the requested analysis.
The code should:
1. Use pandas, numpy, and matplotlib for data analysis and visualization
2. Handle errors gracefully
3. Save any plots/charts to files in the specified output directory
4. Print summary statistics and results
5. Be self-contained and executable
Return ONLY the Python code, no explanations.""")
            
            human_message = HumanMessage(content=prompt)
            
            # Invoke LLM to generate code
            if not self.llm:
                return {
                    "success": False,
                    "error": "LLM not initialized"
                }
            
            response = self.llm.invoke([system_message, human_message])
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract code from response (handle markdown code blocks)
            code = self._extract_code_from_response(response_text)
            
            if not code:
                return {
                    "success": False,
                    "error": "Failed to extract code from LLM response",
                    "response": response_text[:500]
                }
            
            logger.info("Code generated successfully from LLM")
            return {
                "success": True,
                "code": code
            }
        
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            return {
                "success": False,
                "error": f"Code generation failed: {str(e)}"
            }
    
    
    def _execute_code(
        self,
        code: str,
        data_context: Optional[Dict[str, Any]] = None,
        output_dir: str = ""
    ) -> Dict[str, Any]:
        """
        Execute Python code in isolated subprocess.
        
        Args:
            code: Python code to execute
            data_context: Optional context/data to pass to code
            output_dir: Directory for output files
        
        Returns:
            Dictionary with execution results
        """
        try:
            import time
            start_time = time.time()
            
            # Create a temporary Python script
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False,
                dir=output_dir
            ) as f:
                script_path = f.name
                
                # Add setup code for data context if provided
                setup_code = self._generate_setup_code(data_context, output_dir)
                f.write(setup_code)
                f.write("\n\n")
                f.write(code)
            
            try:
                # Execute code in subprocess
                result = subprocess.run(
                    ["python", script_path],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                execution_time = time.time() - start_time
                
                if result.returncode == 0:
                    logger.info(f"Code executed successfully in {execution_time:.2f}s")
                    return {
                        "success": True,
                        "output": result.stdout,
                        "error": None,
                        "execution_time": execution_time
                    }
                else:
                    logger.error(f"Code execution failed: {result.stderr}")
                    return {
                        "success": False,
                        "output": result.stdout,
                        "error": result.stderr,
                        "execution_time": execution_time
                    }
            
            finally:
                # Cleanup temporary script
                try:
                    os.unlink(script_path)
                except:
                    pass
        
        except subprocess.TimeoutExpired:
            logger.error("Code execution timed out (5 minutes)")
            return {
                "success": False,
                "error": "Code execution timed out after 5 minutes",
                "output": ""
            }
        
        except Exception as e:
            logger.error(f"Error executing code: {str(e)}")
            return {
                "success": False,
                "error": f"Execution failed: {str(e)}",
                "output": ""
            }
    
    
    def _generate_setup_code(
        self,
        data_context: Optional[Dict[str, Any]] = None,
        output_dir: str = ""
    ) -> str:
        """Generate setup code to initialize context and output directory."""
        setup_code = f"""
import os
import sys
import json

# Setup
OUTPUT_DIR = r'{output_dir}'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Import common libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use non-interactive backend

# Configure matplotlib for file output
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100
"""
        
        # Add data context setup if provided
        if data_context:
            setup_code += "\n# Data context\n"
            setup_code += f"DATA_CONTEXT = {json.dumps(data_context, default=str)}\n"
        
        return setup_code
    
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response, handling markdown code blocks."""
        # Try to extract from markdown code block
        code_block_pattern = r"```(?:python)?\s*\n(.*?)\n```"
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        
        if matches:
            # Return the first code block found
            return matches[0].strip()
        
        # If no code block found, check if the entire response is code
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            # Skip explanatory text
            if line.strip().startswith(('#', 'python', '```', '```python')):
                continue
            if line.strip() and (line[0] not in (' ', '\t') or in_code):
                code_lines.append(line)
                in_code = True
        
        # If we found code lines, return them
        if code_lines:
            code = '\n'.join(code_lines).strip()
            if code and not code.startswith(('Here', 'This', 'The', 'I', 'Sure')):
                return code
        
        return response.strip()
    
    
    def _collect_generated_files(self, output_dir: str) -> Dict[str, Any]:
        """
        Collect all generated files from output directory.
        
        Returns:
            Dictionary mapping file type to list of file paths
        """
        generated_files = {
            "images": [],
            "charts": [],
            "data": [],
            "other": []
        }
        
        try:
            output_path = Path(output_dir)
            
            for file_path in output_path.iterdir():
                if file_path.is_file():
                    suffix = file_path.suffix.lower()
                    rel_path = str(file_path)
                    
                    if suffix in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                        generated_files["images"].append(rel_path)
                    elif suffix in ['.pdf', '.svg']:
                        generated_files["charts"].append(rel_path)
                    elif suffix in ['.csv', '.json', '.xlsx']:
                        generated_files["data"].append(rel_path)
                    else:
                        generated_files["other"].append(rel_path)
            
            logger.info(f"Collected {sum(len(v) for v in generated_files.values())} generated files")
        
        except Exception as e:
            logger.error(f"Error collecting files: {str(e)}")
        
        return generated_files
    
    
    def _build_code_generation_prompt(
        self,
        request: str,
        data_context: Optional[Dict[str, Any]] = None,
        output_dir: str = ""
    ) -> str:
        """Build detailed prompt for code generation."""
        prompt = f"""Generate Python code to accomplish the following analysis task:

REQUEST: {request}

REQUIREMENTS:
1. Use pandas, numpy, and matplotlib for data analysis and visualization
2. Save all plots/charts to OUTPUT_DIR using plt.savefig()
3. Print informative output about the analysis
4. Handle missing data and edge cases
5. Include proper error handling

AVAILABLE CONTEXT:
- OUTPUT_DIR: {output_dir}
- Available libraries: {', '.join(self.available_libraries)}
"""
        
        if data_context:
            prompt += f"\nDATA CONTEXT:\n{json.dumps(data_context, indent=2, default=str)}\n"
        
        prompt += """
Generate ONLY executable Python code. The code will be saved to a file and executed directly.
Do not include any markdown formatting or explanations - just the raw Python code."""
        
        return prompt.strip()
    
    
    def execute_task(self, request: AgentExecutionRequest) -> AgentExecutionResponse:
        """
        Execute a code interpreter operation using standardized interface.
        
        Args:
            request: AgentExecutionRequest with operation and parameters
        
        Returns:
            AgentExecutionResponse: Standardized response
        """
        start_time = time.time()
        
        # DEBUG LOGGING
        logger.debug("=" * 80)
        logger.debug(f"[{self.agent_name}] execute_task() called")
        logger.debug(f"[{self.agent_name}] request type: {type(request)}")
        logger.debug(f"[{self.agent_name}] request keys: {list(request.keys()) if isinstance(request, dict) else 'N/A'}")
        logger.debug("=" * 80)
        
        try:
            # Extract request parameters
            operation = request.get('operation', '')
            parameters = request.get('parameters', {})
            task_id = request.get('task_id', f"code_interpreter_{int(start_time * 1000)}")
            
            logger.info(f"[{self.agent_name}] Executing operation: {operation} for task_id: {task_id}")
            logger.debug(f"[{self.agent_name}] Parameters: {list(parameters.keys())}")
            
            # Route to operation handlers
            if operation == "execute_analysis":
                logger.debug(f"[{self.agent_name}] Routing to execute_analysis operation")
                legacy_result = self.execute_analysis(
                    request=parameters.get('request', ''),
                    data_context=parameters.get('data_context'),
                    output_dir=parameters.get('output_dir')
                )
            
            elif operation == "generate_code":
                logger.debug(f"[{self.agent_name}] Routing to generate_code operation")
                legacy_result = self._generate_code(
                    request=parameters.get('request', ''),
                    data_context=parameters.get('data_context'),
                    output_dir=parameters.get('output_dir', '')
                )
            
            elif operation == "execute_code":
                logger.debug(f"[{self.agent_name}] Routing to execute_code operation")
                legacy_result = self._execute_code(
                    code=parameters.get('code', ''),
                    data_context=parameters.get('data_context'),
                    output_dir=parameters.get('output_dir', '')
                )
            
            elif operation == "analyze_data":
                # Alias for execute_analysis
                logger.debug(f"[{self.agent_name}] Routing to analyze_data (alias for execute_analysis)")
                legacy_result = self.execute_analysis(
                    request=parameters.get('request', ''),
                    data_context=parameters.get('data_context'),
                    output_dir=parameters.get('output_dir')
                )
            
            else:
                logger.error(f"[{self.agent_name}] Unknown operation: {operation}")
                error_response = create_error_response(
                    error_code="UNKNOWN_OPERATION",
                    error_type="validation_error",
                    message=f"Unknown operation: {operation}",
                    source=self.agent_name,
                    details={"supported_operations": self.supported_operations}
                )
                return self._error_response_to_agent_response(error_response)
            
            # Convert to standardized format
            standardized_response = self._convert_to_standard_response(
                legacy_result, operation, task_id, start_time, parameters
            )
            
            # Publish completion event
            self._publish_completion_event(standardized_response, operation)
            
            return standardized_response
        
        except Exception as e:
            logger.error(f"[{self.agent_name}] Error in execute_task: {str(e)}")
            error_response = exception_to_error_response(
                e,
                source=self.agent_name,
                operation=operation if 'operation' in locals() else "unknown"
            )
            return self._error_response_to_agent_response(error_response)
    
    
    def _convert_to_standard_response(
        self,
        legacy_result: Dict[str, Any],
        operation: str,
        task_id: str,
        start_time: float,
        parameters: Dict[str, Any]
    ) -> AgentExecutionResponse:
        """Convert legacy result to standardized AgentExecutionResponse."""
        execution_time_ms = int((time.time() - start_time) * 1000)
        success = legacy_result.get("success", False)
        
        # Build standardized response
        response: AgentExecutionResponse = {
            "status": "success" if success else "failure",
            "success": success,
            "result": legacy_result if success else {},
            "artifacts": [],
            "execution_time_ms": execution_time_ms,
            "timestamp": datetime.now().isoformat(),
            "agent_name": self.agent_name,
            "operation": operation,
            "blackboard_entries": [],
            "warnings": []
        }
        
        # Add error information if failed
        if not success:
            error_msg = legacy_result.get("error", "Unknown error")
            response["error"] = str(error_msg)
        
        # Add blackboard entries for successful operations
        if success:
            blackboard_entries = []
            
            if operation in ["execute_analysis", "analyze_data"]:
                # Store analysis results
                blackboard_entries.append({
                    "key": f"analysis_result_{task_id}",
                    "value": {
                        "generated_code": legacy_result.get("generated_code"),
                        "output": legacy_result.get("output"),
                        "generated_files": legacy_result.get("generated_files", {}),
                        "output_directory": legacy_result.get("output_directory"),
                        "execution_time": legacy_result.get("execution_time")
                    },
                    "scope": "workflow",
                    "ttl": 3600
                })
            
            elif operation == "generate_code":
                # Store generated code
                blackboard_entries.append({
                    "key": f"generated_code_{task_id}",
                    "value": {
                        "code": legacy_result.get("code"),
                        "request": parameters.get("request")
                    },
                    "scope": "workflow",
                    "ttl": 1800
                })
            
            elif operation == "execute_code":
                # Store execution results
                blackboard_entries.append({
                    "key": f"code_execution_{task_id}",
                    "value": {
                        "output": legacy_result.get("output"),
                        "execution_time": legacy_result.get("execution_time"),
                        "error": legacy_result.get("error")
                    },
                    "scope": "workflow",
                    "ttl": 1800
                })
            
            if blackboard_entries:
                response["blackboard_entries"] = blackboard_entries
        
        return response
    
    
    def _publish_completion_event(self, response: AgentExecutionResponse, operation: str):
        """Publish completion event to EventBus."""
        try:
            event_type_map = {
                "execute_analysis": "code_analysis_completed",
                "generate_code": "code_generated",
                "execute_code": "code_executed",
                "analyze_data": "data_analyzed"
            }
            
            event_type = event_type_map.get(operation, "code_interpreter_completed")
            
            event = create_system_event(
                event_type=event_type,
                event_category="agent_execution",
                source_agent=self.agent_name,
                payload={
                    "status": response["status"],
                    "operation": operation,
                    "execution_time_ms": response["execution_time_ms"]
                }
            )
            
            self.event_bus.publish(event)
            logger.debug(f"Published {event_type} event")
        
        except Exception as e:
            logger.error(f"Error publishing completion event: {str(e)}")
    
    
    def _error_response_to_agent_response(self, error_response: Any) -> AgentExecutionResponse:
        """Convert ErrorResponse to AgentExecutionResponse."""
        return {
            "status": "failure",
            "success": False,
            "result": {},
            "artifacts": [],
            "execution_time_ms": 0,
            "timestamp": datetime.now().isoformat(),
            "agent_name": self.agent_name,
            "operation": "unknown",
            "blackboard_entries": [],
            "warnings": [],
            "error": error_response.get("message", "Unknown error") if isinstance(error_response, dict) else str(error_response)
        }
