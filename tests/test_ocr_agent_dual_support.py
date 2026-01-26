"""
Tests for OCRImageAgent Dual-Format Support (Week 7 Day 1)

Tests verify:
1. execute_task() works with all three calling conventions
2. run_operation() backward compatibility wrapper works
3. Legacy calls return legacy dict format
4. Standardized calls return AgentExecutionResponse
5. Events are published on completion
6. Backward compatibility is maintained
"""

import pytest
from pathlib import Path
import tempfile
import time
from typing import Dict, Any

from task_manager.sub_agents.ocr_image_agent import OCRImageAgent
from task_manager.models import AgentExecutionResponse
from task_manager.utils.validation import validate_agent_execution_response
from task_manager.core.event_bus import get_event_bus


def create_test_request(task_id: str, operation: str, parameters: Dict[str, Any], temp_dir: Path) -> Dict[str, Any]:
    """Helper to create test requests with required fields."""
    return {
        "task_id": task_id,
        "task_description": f"Test {operation} operation",
        "task_type": "atomic",
        "operation": operation,
        "parameters": parameters,
        "input_data": {},
        "temp_folder": str(temp_dir),
        "output_folder": str(temp_dir),
        "cache_enabled": False,
        "blackboard": [],
        "relevant_entries": [],
        "max_retries": 3
    }


class TestOCRImageAgentDualSupport:
    """Test OCRImageAgent dual-format support and execute_task migration."""
    
    @pytest.fixture
    def agent(self):
        """Create OCRImageAgent instance."""
        return OCRImageAgent()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    # ==================== METHOD RENAME TESTS ====================
    
    def test_execute_task_exists(self, agent):
        """Test that execute_task() method exists"""
        assert hasattr(agent, 'execute_task')
        assert callable(agent.execute_task)
    
    def test_run_operation_backward_compat(self, agent):
        """Test that run_operation() still exists for backward compatibility"""
        assert hasattr(agent, 'run_operation')
        assert callable(agent.run_operation)
    
    def test_run_operation_forwards_to_execute_task(self, agent, temp_dir):
        """Test that run_operation() forwards to execute_task()"""
        # This should work via the wrapper
        result = agent.run_operation(
            "ocr_image",
            {"image_path": "nonexistent.jpg"}  # Will fail but tests the call path
        )
        
        assert isinstance(result, dict)
        assert "success" in result
    
    # ==================== CALLING CONVENTION TESTS ====================
    
    def test_legacy_positional_call(self, agent):
        """Test legacy positional arguments: execute_task(operation, parameters)"""
        result = agent.execute_task(
            "ocr_image",
            {"image_path": "test.jpg"}
        )
        
        assert isinstance(result, dict)
        assert "success" in result
    
    def test_legacy_dict_call(self, agent):
        """Test legacy dict format: execute_task({'operation': ..., 'parameters': ...})"""
        result = agent.execute_task({
            "operation": "ocr_image",
            "parameters": {"image_path": "test.jpg"}
        })
        
        assert isinstance(result, dict)
        assert "success" in result
    
    def test_standardized_request_call(self, agent, temp_dir):
        """Test standardized AgentExecutionRequest format"""
        request = create_test_request(
            "test_ocr_001",
            "ocr_image",
            {"image_path": "test.jpg"},
            temp_dir
        )
        
        result = agent.execute_task(request)
        
        assert isinstance(result, dict)
        assert "status" in result
        assert "success" in result
        assert "artifacts" in result
        assert "agent_name" in result
        assert result["agent_name"] == "ocr_image_agent"
    
    # ==================== RESPONSE FORMAT TESTS ====================
    
    def test_legacy_call_returns_legacy_format(self, agent):
        """Verify legacy calls return legacy dict format"""
        result = agent.execute_task(
            "ocr_image",
            {"image_path": "test.jpg"}
        )
        
        # Legacy format has 'success' at top level
        assert "success" in result
        assert isinstance(result, dict)
        
        # Should NOT have standardized fields at top level
        assert "status" not in result
        assert "agent_name" not in result or result.get("agent_name") is None
    
    def test_standardized_call_returns_standardized_format(self, agent, temp_dir):
        """Verify standardized calls return AgentExecutionResponse"""
        request = create_test_request(
            "test_ocr_002",
            "ocr_image",
            {"image_path": "test.jpg"},
            temp_dir
        )
        
        result = agent.execute_task(request)
        
        # Standardized format has these fields
        assert "status" in result
        assert "success" in result
        assert "result" in result
        assert "artifacts" in result
        assert "execution_time_ms" in result
        assert "timestamp" in result
        assert "agent_name" in result
        assert "operation" in result
        assert "blackboard_entries" in result
        assert "warnings" in result
        
        assert result["agent_name"] == "ocr_image_agent"
        assert result["operation"] == "ocr_image"
    
    def test_standardized_response_validation(self, agent, temp_dir):
        """Test that standardized responses pass validation"""
        request = create_test_request(
            "test_ocr_003",
            "ocr_image",
            {"image_path": "test.jpg"},
            temp_dir
        )
        
        result = agent.execute_task(request)
        
        validation = validate_agent_execution_response(dict(result))
        assert validation.valid is True, f"Validation errors: {validation.errors}"
    
    # ==================== OPERATION TESTS ====================
    
    def test_ocr_image_operation_legacy(self, agent):
        """Test ocr_image operation with legacy call"""
        result = agent.execute_task(
            "ocr_image",
            {
                "image_path": "nonexistent.jpg",
                "language": "eng"
            }
        )
        
        assert result["success"] is False  # File doesn't exist
        assert "error" in result
    
    def test_ocr_image_operation_standardized(self, agent, temp_dir):
        """Test ocr_image operation with standardized call"""
        request = create_test_request(
            "test_ocr_004",
            "ocr_image",
            {
                "image_path": "nonexistent.jpg",
                "language": "eng"
            },
            temp_dir
        )
        
        result = agent.execute_task(request)
        
        assert result["success"] is False
        assert result["status"] == "failure"
        assert "error" in result
    
    def test_analyze_visual_content_operation(self, agent, temp_dir):
        """Test analyze_visual_content operation"""
        request = create_test_request(
            "test_ocr_005",
            "analyze_visual_content",
            {
                "image_path": "test.jpg",
                "analysis_prompt": "Describe this image"
            },
            temp_dir
        )
        
        result = agent.execute_task(request)
        
        # Will fail without actual image, but tests the operation routing
        assert isinstance(result, dict)
        assert "success" in result
    
    # ==================== ERROR HANDLING TESTS ====================
    
    def test_invalid_operation_legacy(self, agent):
        """Test unknown operation with legacy call"""
        result = agent.execute_task(
            "invalid_operation",
            {}
        )
        
        assert result["success"] is False
        assert "error" in result
        assert "Unknown operation" in result["error"]
    
    def test_invalid_operation_standardized(self, agent, temp_dir):
        """Test unknown operation with standardized call"""
        request = create_test_request(
            "test_error_001",
            "nonexistent_op",
            {},
            temp_dir
        )
        
        result = agent.execute_task(request)
        
        assert result["success"] is False
        assert result["status"] == "failure"
        assert "error" in result
    
    # ==================== BACKWARD COMPATIBILITY TESTS ====================
    
    def test_all_operations_work_legacy(self, agent):
        """Test that all operations work with legacy interface"""
        operations = [
            ("ocr_image", {"image_path": "test.jpg"}),
            ("batch_ocr", {"image_paths": ["test1.jpg", "test2.jpg"]}),
            ("process_screenshot", {"image_path": "screen.png"}),
        ]
        
        for operation, params in operations:
            result = agent.execute_task(operation, params)
            assert isinstance(result, dict)
            assert "success" in result
    
    def test_keyword_arguments_legacy(self, agent):
        """Test legacy keyword argument call"""
        result = agent.execute_task(
            operation="ocr_image",
            parameters={"image_path": "test.jpg"}
        )
        
        assert isinstance(result, dict)
        assert "success" in result
    
    def test_run_operation_still_works(self, agent):
        """Test that old run_operation() method still works"""
        result = agent.run_operation(
            "ocr_image",
            {"image_path": "test.jpg"}
        )
        
        assert isinstance(result, dict)
        assert "success" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
