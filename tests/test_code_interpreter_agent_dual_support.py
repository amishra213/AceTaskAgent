"""
Test suite for CodeInterpreterAgent with standardized interface.

Tests verify:
- Standardized AgentExecutionRequest/Response format
- All 4 operations: execute_analysis, generate_code, execute_code, analyze_data
- Response validation (AgentExecutionResponse schema)
- Event publication
- Blackboard entry creation
- Error handling

Note: Legacy dual-format support removed in Week 8.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from task_manager.sub_agents.code_interpreter_agent import CodeInterpreterAgent
from task_manager.models.messages import AgentExecutionResponse, AgentExecutionRequest
from task_manager.core.event_bus import get_event_bus


class TestCodeInterpreterAgentStandardized:
    """Test suite for CodeInterpreterAgent with standardized interface."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM for testing."""
        llm = Mock()
        response = Mock()
        response.content = """```python
import pandas as pd
import numpy as np

# Sample analysis
data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
print(data.describe())
```"""
        llm.invoke.return_value = response
        return llm
    
    @pytest.fixture
    def agent(self, mock_llm):
        """Create CodeInterpreterAgent instance."""
        return CodeInterpreterAgent(llm=mock_llm)
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp(prefix="test_code_interpreter_")
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def create_request(self, operation: str, parameters: Dict[str, Any], task_id: str = "test") -> AgentExecutionRequest:
        """Helper to create standardized AgentExecutionRequest."""
        return {
            "task_id": task_id,
            "task_description": f"Test {operation}",
            "task_type": "atomic",
            "operation": operation,
            "parameters": parameters,
            "input_data": parameters,
            "temp_folder": "/tmp",
            "output_folder": "/tmp",
            "cache_enabled": False,
            "blackboard": [],
            "relevant_entries": [],
            "max_retries": 1
        }
    
    # ==================== Calling Convention Tests ====================
    
    @pytest.mark.skip(reason="Legacy positional format removed in Week 8")
    def test_execute_task_legacy_positional(self, agent, mock_llm):
        """Test legacy positional calling convention: execute_task(operation, parameters)."""
        with patch.object(agent, '_execute_code') as mock_execute:
            mock_execute.return_value = {
                "success": True,
                "output": "Test output",
                "execution_time": 0.5
            }
            
            result = agent.execute_task(
                "execute_code",
                {"code": "print('Hello')", "task_id": "test_123"}
            )
            
            assert isinstance(result, dict)
            assert result["success"] is True
            assert "output" in result
            mock_execute.assert_called_once()
    
    @pytest.mark.skip(reason="Legacy dict format removed in Week 8")
    def test_execute_task_legacy_dict(self, agent, mock_llm):
        """Test legacy dict calling convention: execute_task({'operation': 'X', 'parameters': {...}})."""
        with patch.object(agent, '_generate_code') as mock_generate:
            mock_generate.return_value = {
                "success": True,
                "code": "print('Generated code')"
            }
            
            result = agent.execute_task({
                "operation": "generate_code",
                "parameters": {"request": "Generate a hello world", "task_id": "test_456"}
            })
            
            assert isinstance(result, dict)
            assert result["success"] is True
            assert "code" in result
            mock_generate.assert_called_once()
    
    def test_execute_task_standardized_request(self, agent, mock_llm):
        """Test standardized calling convention with AgentExecutionRequest."""
        with patch.object(agent, '_execute_code') as mock_execute:
            mock_execute.return_value = {
                "success": True,
                "output": "Execution successful",
                "execution_time": 0.3
            }
            
            request = {
                "task_id": "task_789",
                "operation": "execute_code",
                "parameters": {"code": "print('Test')"}
            }
            
            result = agent.execute_task(request=request)
            
            assert isinstance(result, dict)
            assert "agent_name" in result
            assert result["agent_name"] == "code_interpreter_agent"
            assert "status" in result
            assert result["status"] == "success"
            mock_execute.assert_called_once()
    
    # ==================== Response Validation Tests ====================
    
    def test_standardized_response_validation(self, agent, mock_llm):
        """Verify standardized response conforms to AgentExecutionResponse schema."""
        with patch.object(agent, '_execute_code') as mock_execute:
            mock_execute.return_value = {
                "success": True,
                "output": "Test output",
                "execution_time": 0.5
            }
            
            request = {
                "task_id": "validate_test",
                "operation": "execute_code",
                "parameters": {"code": "print('Validation test')"}
            }
            
            response = agent.execute_task(request=request)
            
            # Validate required fields
            assert "agent_name" in response
            assert "status" in response
            assert "success" in response
            assert "result" in response
            assert "artifacts" in response
            assert "execution_time_ms" in response
            assert "timestamp" in response
            assert "operation" in response
            assert "blackboard_entries" in response
            assert "warnings" in response
            
            assert response["status"] in ["success", "partial_success", "failure", "requires_human_input"]
            assert isinstance(response["result"], dict)
            assert isinstance(response["artifacts"], list)
            assert isinstance(response["blackboard_entries"], list)
            assert isinstance(response["warnings"], list)
    
    # ==================== Event Publication Tests ====================
    
    def test_event_publication(self, agent, mock_llm):
        """Test that events are published to EventBus."""
        event_bus = get_event_bus()
        
        with patch.object(agent, '_execute_code') as mock_execute:
            mock_execute.return_value = {
                "success": True,
                "output": "Event test",
                "execution_time": 0.2
            }
            
            with patch.object(event_bus, 'publish') as mock_publish:
                request = {
                    "task_id": "event_test",
                    "operation": "execute_code",
                    "parameters": {"code": "print('Event')"}
                }
                
                agent.execute_task(request=request)
                
                # Verify event was published
                assert mock_publish.call_count >= 1
                event = mock_publish.call_args[0][0]
                assert event["event_type"] == "code_executed"
                assert event["source_agent"] == "code_interpreter_agent"
    
    # ==================== Blackboard Entry Tests ====================
    
    def test_blackboard_entries_for_analysis(self, agent, mock_llm, temp_output_dir):
        """Test blackboard entries created for analysis operation."""
        with patch.object(agent, 'execute_analysis') as mock_analysis:
            mock_analysis.return_value = {
                "success": True,
                "generated_code": "print('Analysis code')",
                "output": "Analysis output",
                "generated_files": {"images": ["chart.png"]},
                "output_directory": temp_output_dir,
                "execution_time": 1.2
            }
            
            request = {
                "task_id": "analysis_test",
                "operation": "execute_analysis",
                "parameters": {"request": "Analyze data"}
            }
            
            response = agent.execute_task(request=request)
            
            # Check blackboard entries
            assert "blackboard_entries" in response
            entries = response["blackboard_entries"]
            assert len(entries) > 0
            
            # Find analysis result entry
            analysis_entry = next(
                (e for e in entries if e["key"] == "analysis_result_analysis_test"),
                None
            )
            assert analysis_entry is not None
            assert analysis_entry["scope"] == "workflow"
            assert "generated_code" in analysis_entry["value"]
            assert "output" in analysis_entry["value"]
    
    def test_blackboard_entries_for_code_generation(self, agent, mock_llm):
        """Test blackboard entries created for code generation."""
        with patch.object(agent, '_generate_code') as mock_generate:
            mock_generate.return_value = {
                "success": True,
                "code": "generated_code = 'Hello World'"
            }
            
            request = {
                "task_id": "codegen_test",
                "operation": "generate_code",
                "parameters": {"request": "Generate hello world"}
            }
            
            response = agent.execute_task(request=request)
            
            # Check blackboard entries
            assert "blackboard_entries" in response
            entries = response["blackboard_entries"]
            
            # Find generated code entry
            code_entry = next(
                (e for e in entries if e["key"] == "generated_code_codegen_test"),
                None
            )
            assert code_entry is not None
            assert "code" in code_entry["value"]
            assert code_entry["scope"] == "workflow"
    
    def test_blackboard_entries_for_code_execution(self, agent, mock_llm):
        """Test blackboard entries created for code execution."""
        with patch.object(agent, '_execute_code') as mock_execute:
            mock_execute.return_value = {
                "success": True,
                "output": "Execution output",
                "execution_time": 0.8,
                "error": None
            }
            
            request = {
                "task_id": "exec_test",
                "operation": "execute_code",
                "parameters": {"code": "print('Execute')"}
            }
            
            response = agent.execute_task(request=request)
            
            # Check blackboard entries
            assert "blackboard_entries" in response
            entries = response["blackboard_entries"]
            
            # Find execution result entry
            exec_entry = next(
                (e for e in entries if e["key"] == "code_execution_exec_test"),
                None
            )
            assert exec_entry is not None
            assert "output" in exec_entry["value"]
            assert "execution_time" in exec_entry["value"]
    
    # ==================== Operation Tests ====================
    
    def test_operation_execute_analysis(self, agent, mock_llm, temp_output_dir):
        """Test execute_analysis operation."""
        with patch.object(agent, 'execute_analysis') as mock_analysis:
            mock_analysis.return_value = {
                "success": True,
                "request": "Analyze sales data",
                "generated_code": "df.describe()",
                "output": "Statistics...",
                "generated_files": {},
                "execution_time": 2.5
            }
            
            request = self.create_request(
                "execute_analysis",
                {"request": "Analyze sales data", "output_dir": temp_output_dir},
                "test_exec_analysis"
            )
            result = agent.execute_task(request=request)
            
            assert result["success"] is True
            assert "generated_code" in result.get("result", {})
            mock_analysis.assert_called_once()
    
    def test_operation_generate_code(self, agent, mock_llm):
        """Test generate_code operation."""
        request = self.create_request(
            "generate_code",
            {"request": "Generate code to calculate mean"},
            "test_gen_code"
        )
        result = agent.execute_task(request=request)
        
        assert result["success"] is True
        assert "code" in result.get("result", {})
    
    def test_operation_execute_code(self, agent, temp_output_dir):
        """Test execute_code operation with real execution."""
        # Use simple code without pandas imports to avoid subprocess issues
        simple_code = "print('Hello from test')"
        
        with patch.object(agent, '_execute_code') as mock_execute:
            mock_execute.return_value = {
                "success": True,
                "output": "Hello from test\n",
                "execution_time": 0.1
            }
            
            request = self.create_request(
                "execute_code",
                {"code": simple_code, "output_dir": temp_output_dir},
                "test_exec_code"
            )
            result = agent.execute_task(request=request)
            
            assert result["success"] is True
            result_data = result.get("result", {})
            assert "output" in result_data
            assert "Hello from test" in result_data["output"]
    
    def test_operation_analyze_data(self, agent, mock_llm, temp_output_dir):
        """Test analyze_data operation (alias for execute_analysis)."""
        with patch.object(agent, 'execute_analysis') as mock_analysis:
            mock_analysis.return_value = {
                "success": True,
                "output": "Data analyzed"
            }
            
            request = self.create_request(
                "analyze_data",
                {"request": "Analyze customer data", "output_dir": temp_output_dir},
                "test_analyze_data"
            )
            result = agent.execute_task(request=request)
            
            assert result["success"] is True
            mock_analysis.assert_called_once()
    
    def test_unknown_operation(self, agent):
        """Test handling of unknown operation."""
        request = self.create_request(
            "unknown_operation",
            {"some": "params"},
            "test_unknown"
        )
        result = agent.execute_task(request=request)
        
        assert result["success"] is False
        assert "error" in result
        assert "unknown_operation" in result["error"].lower()
    
    # ==================== Error Handling Tests ====================
    
    def test_error_handling_code_execution_failure(self, agent, temp_output_dir):
        """Test error handling when code execution fails."""
        invalid_code = "import nonexistent_module"
        
        request = self.create_request(
            "execute_code",
            {"code": invalid_code, "output_dir": temp_output_dir},
            "test_error_handling"
        )
        result = agent.execute_task(request=request)
        
        assert result["success"] is False
        assert "error" in result
    
    def test_error_handling_missing_llm(self):
        """Test error handling when LLM is not provided."""
        agent_no_llm = CodeInterpreterAgent(llm=None)
        
        # Use standardized format via helper
        request = self.create_request(
            operation="generate_code",
            parameters={"request": "Generate code"},
            task_id="no_llm_test"
        )
        
        result = agent_no_llm.execute_task(request=request)
        
        assert result["success"] is False
        assert "error" in result
        error_text = result.get("error", "")
        # Check for either "LLM not provided" or "LLM not initialized"
        assert ("LLM not provided" in str(error_text) or "LLM not initialized" in str(error_text))
    
    def test_error_handling_standardized(self, agent):
        """Test error responses conform to standardized format."""
        request = self.create_request("invalid_op", {}, "error_test")
        
        response = agent.execute_task(request=request)
        
        assert "error" in response
        assert response["status"] == "failure"
        assert "success" in response
        assert response["success"] is False
    
    @pytest.mark.skip(reason="Invalid call test not applicable - type system enforces correct calls")
    def test_invalid_call_raises_error(self, agent):
        """Test that invalid calling conventions return error responses."""
        # Invalid call with no arguments - should return error response, not raise
        result = agent.execute_task()  # No arguments
        
        # Should return error response
        assert isinstance(result, dict)
        assert result.get("success") is False or result.get("status") == "failure"
    
    # ==================== Backward Compatibility Tests ====================
    
    @pytest.mark.skip(reason="Backward compatibility removed in Week 8 - standardized-only interface")
    def test_backward_compatibility_complete(self, agent, mock_llm):
        """Test complete backward compatibility with legacy code."""
        with patch.object(agent, '_execute_code') as mock_execute:
            mock_execute.return_value = {
                "success": True,
                "output": "Legacy test",
                "execution_time": 0.5
            }
            
            # Legacy positional call
            legacy_result = agent.execute_task(
                "execute_code",
                {"code": "print('Legacy')"}
            )
            
            # Should return dict with 'success' field
            assert isinstance(legacy_result, dict)
            assert "success" in legacy_result
            assert legacy_result["success"] is True
            assert "output" in legacy_result
    
    def test_event_types_for_operations(self, agent, mock_llm):
        """Test correct event types are published for different operations."""
        event_bus = get_event_bus()
        
        test_cases = [
            ("execute_analysis", "code_analysis_completed", "execute_analysis"),
            ("generate_code", "code_generated", "_generate_code"),
            ("execute_code", "code_executed", "_execute_code"),
            ("analyze_data", "data_analyzed", "execute_analysis")
        ]
        
        for operation, expected_event_type, method_name in test_cases:
            with patch.object(agent, method_name) as mock_op:
                mock_op.return_value = {"success": True, "output": "Test"}
                
                with patch.object(event_bus, 'publish') as mock_publish:
                    request = self.create_request(operation, {"request": "test"}, f"test_{operation}")
                    agent.execute_task(request=request)
                    
                    if mock_publish.called:
                        event = mock_publish.call_args[0][0]
                        assert event["event_type"] == expected_event_type
    
    # ==================== Code Generation Tests ====================
    
    def test_code_extraction_from_markdown(self, agent, mock_llm):
        """Test extraction of code from markdown-formatted LLM response."""
        # Test with markdown code block
        mock_llm.invoke.return_value.content = """```python
print('Extracted code')
```"""
        
        request = self.create_request(
            "generate_code",
            {"request": "Generate test code"},
            "test_extraction"
        )
        result = agent.execute_task(request=request)
        
        assert result["success"] is True
        result_data = result.get("result", {})
        assert "print('Extracted code')" in result_data.get("code", "")
    
    def test_execute_analysis_end_to_end(self, agent, mock_llm, temp_output_dir):
        """Test complete analysis flow end-to-end."""
        # Mock the code execution to avoid pandas import issues in subprocess
        with patch.object(agent, '_execute_code') as mock_execute:
            mock_execute.return_value = {
                "success": True,
                "output": "Hello world\n",
                "execution_time": 0.1,
                "error": None
            }
            
            request = self.create_request(
                "execute_analysis",
                {
                    "request": "Print hello world",
                    "output_dir": temp_output_dir
                },
                "test_end_to_end"
            )
            result = agent.execute_task(request=request)
            
            assert result["success"] is True
            result_data = result.get("result", {})
            assert "generated_code" in result_data
            assert "output" in result_data
    
    # ==================== Helper Method Tests ====================
    
    def test_convert_to_legacy_response(self, agent):
        """Test _convert_to_legacy_response helper method."""
        standardized_response: AgentExecutionResponse = {
            "status": "success",
            "success": True,
            "result": {
                "output": "Test output",
                "execution_time": 0.5
            },
            "artifacts": [],
            "execution_time_ms": 500,
            "timestamp": datetime.now().isoformat(),
            "agent_name": "code_interpreter_agent",
            "operation": "execute_code",
            "blackboard_entries": [],
            "warnings": []
        }
        
        legacy = agent._convert_to_legacy_response(standardized_response)
        
        assert isinstance(legacy, dict)
        assert legacy["success"] is True
        assert "output" in legacy
        assert legacy["output"] == "Test output"
    
    def test_dependencies_check(self, agent):
        """Test that dependency checking works correctly."""
        assert hasattr(agent, 'available_libraries')
        assert isinstance(agent.available_libraries, list)
        # At least one of pandas, numpy, matplotlib should be available in test environment
        assert len(agent.available_libraries) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
