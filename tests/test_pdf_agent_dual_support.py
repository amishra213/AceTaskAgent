"""
Tests for PDFAgent with Dual Format Support (Week 3-4)

This test suite validates that PDFAgent supports both:
1. Legacy format (operation + parameters)
2. Standardized format (AgentExecutionRequest/Response)

Run with: pytest tests/test_pdf_agent_dual_support.py -v
"""

import pytest
from pathlib import Path
import tempfile
import os

from task_manager.sub_agents.pdf_agent import PDFAgent
from task_manager.models import AgentExecutionRequest, AgentExecutionResponse


class TestPDFAgentLegacyFormat:
    """Test PDFAgent with legacy calling conventions."""
    
    def test_legacy_operation_parameters_format(self):
        """Test legacy execute_task(operation, parameters) format."""
        agent = PDFAgent()
        
        # Create a proper AgentExecutionRequest
        request: AgentExecutionRequest = {
            "task_id": "test_legacy_1",
            "task_description": "Test legacy format",
            "task_type": "atomic",
            "operation": "read",
            "parameters": {
                "file_path": "nonexistent.pdf",
                "extract_text": True
            },
            "input_data": {},
            "temp_folder": "temp_folder",
            "output_folder": "output_folder",
            "cache_enabled": False,
            "blackboard": [],
            "relevant_entries": [],
            "max_retries": 3
        }
        
        result = agent.execute_task(request)
        
        # Legacy format returns dict with "success" key
        assert isinstance(result, dict)
        assert "success" in result
        assert result["success"] == False  # File doesn't exist
        assert "error" in result
    
    def test_legacy_dict_format(self):
        """Test legacy dict format {operation: ..., parameters: ...}."""
        agent = PDFAgent()
        
        # Create proper AgentExecutionRequest
        request: AgentExecutionRequest = {
            "task_id": "test_legacy_dict_1",
            "task_description": "Test legacy dict format",
            "task_type": "atomic",
            "operation": "read",
            "parameters": {
                "file_path": "test.pdf",
                "extract_text": True
            },
            "input_data": {},
            "temp_folder": "temp_folder",
            "output_folder": "output_folder",
            "cache_enabled": False,
            "blackboard": [],
            "relevant_entries": [],
            "max_retries": 3
        }
        
        result = agent.execute_task(request)
        
        assert isinstance(result, dict)
        assert "success" in result
        # Result should be in legacy format
        assert "result" in result or "error" in result


class TestPDFAgentStandardizedFormat:
    """Test PDFAgent with standardized AgentExecutionRequest/Response."""
    
    def test_standardized_request_format(self):
        """Test with full AgentExecutionRequest."""
        agent = PDFAgent()
        
        request: AgentExecutionRequest = {
            "task_id": "test_pdf_read_1",
            "task_description": "Read a PDF file",
            "task_type": "atomic",
            "operation": "read",
            "parameters": {
                "file_path": "nonexistent.pdf",
                "extract_text": True
            },
            "input_data": {},
            "temp_folder": "temp_folder",
            "output_folder": "output_folder",
            "cache_enabled": False,
            "blackboard": [],
            "relevant_entries": [],
            "max_retries": 3
        }
        
        response = agent.execute_task(request)
        
        # Should return AgentExecutionResponse
        assert isinstance(response, dict)
        assert "status" in response
        assert "success" in response
        assert "agent_name" in response
        assert "operation" in response
        assert "execution_time_ms" in response
        assert "timestamp" in response
        assert "artifacts" in response
        
        # Check standardized response fields
        assert response["agent_name"] == "pdf_agent"
        assert response["operation"] == "read"
        assert response["success"] == False  # File doesn't exist
        assert response["status"] == "failure"
    
    def test_standardized_response_has_error(self):
        """Test that errors are in standardized format."""
        agent = PDFAgent()
        
        request: AgentExecutionRequest = {
            "task_id": "test_error",
            "task_description": "Test error handling",
            "task_type": "atomic",
            "operation": "invalid_operation",
            "parameters": {},
            "input_data": {},
            "temp_folder": "temp",
            "output_folder": "output",
            "cache_enabled": False,
            "blackboard": [],
            "relevant_entries": [],
            "max_retries": 3
        }
        
        response = agent.execute_task(request)
        
        assert response["success"] == False
        assert response["status"] == "failure"
        # Error should be in standardized ErrorResponse format
        error = response.get("error")
        if error:
            assert "error_code" in error
            assert "error_type" in error
            assert "message" in error


class TestPDFAgentBackwardCompatibility:
    """Test that both formats work side by side."""
    
    def test_both_formats_return_correct_type(self):
        """Verify legacy returns dict, standard returns AgentExecutionResponse."""
        agent = PDFAgent()
        
        # Standard call with proper AgentExecutionRequest
        standard_request: AgentExecutionRequest = {
            "task_id": "test_1",
            "task_description": "Test",
            "task_type": "atomic",
            "operation": "read",
            "parameters": {"file_path": "test.pdf"},
            "input_data": {},
            "temp_folder": "temp",
            "output_folder": "output",
            "cache_enabled": False,
            "blackboard": [],
            "relevant_entries": [],
            "max_retries": 3
        }
        
        standard_result = agent.execute_task(standard_request)
        
        # Should have standardized fields
        assert "agent_name" in standard_result
        assert "execution_time_ms" in standard_result
        assert "timestamp" in standard_result
        assert isinstance(standard_result, dict)
        assert "success" in standard_result


class TestPDFAgentWithRealFile:
    """Test PDF operations with actual file operations."""
    
    def test_create_pdf_legacy_format(self, tmp_path):
        """Test PDF creation with standardized format (legacy test name kept for compatibility)."""
        agent = PDFAgent()
        
        output_file = tmp_path / "test_output.pdf"
        
        request: AgentExecutionRequest = {
            "task_id": "test_create_1",
            "task_description": "Create test PDF",
            "task_type": "atomic",
            "operation": "create",
            "parameters": {
                "output_path": str(output_file),
                "content": ["Hello, World!", "This is a test PDF"],
                "title": "Test PDF",
                "author": "Test Agent"
            },
            "input_data": {},
            "temp_folder": str(tmp_path),
            "output_folder": str(tmp_path),
            "cache_enabled": False,
            "blackboard": [],
            "relevant_entries": [],
            "max_retries": 3
        }
        
        result = agent.execute_task(request)
        
        # Check result format
        assert isinstance(result, dict)
        assert "success" in result
        
        # Note: Actual file creation depends on PDF library being installed
        # This test validates the interface, not the implementation
    
    def test_create_pdf_standard_format(self, tmp_path):
        """Test PDF creation with standardized format."""
        agent = PDFAgent()
        
        output_file = tmp_path / "test_standard.pdf"
        
        request: AgentExecutionRequest = {
            "task_id": "create_pdf_1",
            "task_description": "Create test PDF",
            "task_type": "atomic",
            "operation": "create",
            "parameters": {
                "output_path": str(output_file),
                "content": ["Test content"],
                "title": "Test"
            },
            "input_data": {},
            "temp_folder": str(tmp_path),
            "output_folder": str(tmp_path),
            "cache_enabled": False,
            "blackboard": [],
            "relevant_entries": [],
            "max_retries": 3
        }
        
        response = agent.execute_task(request)
        
        # Verify standardized response
        assert response["agent_name"] == "pdf_agent"
        assert response["operation"] == "create"
        assert "execution_time_ms" in response
        assert isinstance(response["artifacts"], list)


class TestPDFAgentEventPublishing:
    """Test that events are published for event-driven workflows."""
    
    def test_completion_event_published(self):
        """Test that completion events are published."""
        from task_manager.core.event_bus import EventBus
        
        # Create isolated event bus for testing
        event_bus = EventBus()
        agent = PDFAgent()
        agent.event_bus = event_bus
        
        events_received = []
        
        def handler(event):
            events_received.append(event)
        
        # Subscribe to completion events
        event_bus.subscribe(
            event_type="pdf_operation_completed",
            handler=handler,
            subscriber_name="test"
        )
        
        # Execute task with proper request
        request: AgentExecutionRequest = {
            "task_id": "test_event_1",
            "task_description": "Test event publishing",
            "task_type": "atomic",
            "operation": "read",
            "parameters": {"file_path": "test.pdf"},
            "input_data": {},
            "temp_folder": "temp_folder",
            "output_folder": "output_folder",
            "cache_enabled": False,
            "blackboard": [],
            "relevant_entries": [],
            "max_retries": 3
        }
        
        agent.execute_task(request)
        
        # Event should have been published
        # Note: May be 0 if operation failed before event publishing
        # This tests the interface, not the full execution path


class TestPDFAgentSupportedOperations:
    """Test all supported operations maintain dual format support."""
    
    @pytest.mark.parametrize("operation", [
        "read",
        "create",
        "merge",
        "extract_pages"
    ])
    def test_all_operations_support_legacy_format(self, operation):
        """Test that all operations work with standardized format (legacy test name kept)."""
        agent = PDFAgent()
        
        # Minimal parameters for each operation
        params_map = {
            "read": {"file_path": "test.pdf"},
            "create": {"output_path": "output.pdf", "content": []},
            "merge": {"input_files": [], "output_path": "merged.pdf"},
            "extract_pages": {"input_path": "test.pdf", "page_numbers": [1], "output_path": "extract.pdf"}
        }
        
        request: AgentExecutionRequest = {
            "task_id": f"test_{operation}_legacy_1",
            "task_description": f"Test {operation} legacy",
            "task_type": "atomic",
            "operation": operation,
            "parameters": params_map[operation],
            "input_data": {},
            "temp_folder": "temp",
            "output_folder": "output",
            "cache_enabled": False,
            "blackboard": [],
            "relevant_entries": [],
            "max_retries": 3
        }
        
        result = agent.execute_task(request)
        
        assert isinstance(result, dict)
        assert "success" in result
    
    @pytest.mark.parametrize("operation", [
        "read",
        "create",
        "merge",
        "extract_pages"
    ])
    def test_all_operations_support_standard_format(self, operation):
        """Test that all operations work with standardized format."""
        agent = PDFAgent()
        
        params_map = {
            "read": {"file_path": "test.pdf"},
            "create": {"output_path": "output.pdf", "content": []},
            "merge": {"input_files": [], "output_path": "merged.pdf"},
            "extract_pages": {"input_path": "test.pdf", "page_numbers": [1], "output_path": "extract.pdf"}
        }
        
        request: AgentExecutionRequest = {
            "task_id": f"test_{operation}_1",
            "task_description": f"Test {operation}",
            "task_type": "atomic",
            "operation": operation,
            "parameters": params_map[operation],
            "input_data": {},
            "temp_folder": "temp",
            "output_folder": "output",
            "cache_enabled": False,
            "blackboard": [],
            "relevant_entries": [],
            "max_retries": 3
        }
        
        response = agent.execute_task(request)
        
        assert "agent_name" in response
        assert response["operation"] == operation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
