"""
Code Interpreter Sub-Agent for executing Python code and data analysis.

Capabilities:
- Parse natural language analysis requests
- Generate Python code using Pandas, Numpy, Matplotlib
- Execute code in isolated subprocess
- Capture output and generated charts/images
- Post results to blackboard for workflow integration
"""

from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import subprocess
import tempfile
import json
import re
import os

from task_manager.utils.logger import get_logger

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
        self.supported_operations = [
            "execute_analysis",
            "generate_code",
            "execute_code",
            "analyze_data"
        ]
        self.llm = llm
        logger.info("Code Interpreter Agent initialized")
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
    
    
    def execute_task(
        self,
        operation: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a code interpreter operation based on operation type.
        
        Args:
            operation: Type of operation (execute_analysis, generate_code, execute_code, analyze_data)
            parameters: Operation parameters
        
        Returns:
            Result dictionary
        """
        logger.info(f"Executing Code Interpreter operation: {operation}")
        
        if operation == "execute_analysis":
            return self.execute_analysis(
                request=parameters.get('request', ''),
                data_context=parameters.get('data_context'),
                output_dir=parameters.get('output_dir')
            )
        
        elif operation == "generate_code":
            return self._generate_code(
                request=parameters.get('request', ''),
                data_context=parameters.get('data_context'),
                output_dir=parameters.get('output_dir', '')
            )
        
        elif operation == "execute_code":
            return self._execute_code(
                code=parameters.get('code', ''),
                data_context=parameters.get('data_context'),
                output_dir=parameters.get('output_dir', '')
            )
        
        elif operation == "analyze_data":
            # Alias for execute_analysis with more specific focus on data analysis
            return self.execute_analysis(
                request=parameters.get('request', ''),
                data_context=parameters.get('data_context'),
                output_dir=parameters.get('output_dir')
            )
        
        else:
            return {
                "success": False,
                "error": f"Unknown operation: {operation}",
                "supported_operations": self.supported_operations
            }
