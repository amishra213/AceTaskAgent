"""
Test suite for DataExtractionAgent dual-format support (Week 7 Day 3).

Tests:
1. Legacy positional call format
2. Legacy dict call format  
3. Standardized AgentExecutionRequest format
4. Response validation
5. Event publication
6. Blackboard entries
7. All 3 operations (extract_relevant_data, get_file_preview, search_in_files)
8. Error handling
9. Backward compatibility
"""

import pytest
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from task_manager.sub_agents.data_extraction_agent import DataExtractionAgent
from task_manager.models import AgentExecutionRequest, AgentExecutionResponse


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_files(temp_dir):
    """Create test files in temp directory."""
    # Create a text file
    (temp_dir / "test.txt").write_text("This is a test file with some content about Python programming.")
    
    # Create a JSON file
    (temp_dir / "data.json").write_text('{"key": "value", "test": "data"}')
    
    # Create a CSV file
    (temp_dir / "data.csv").write_text("name,value\ntest,123\npython,456")
    
    return temp_dir


@pytest.fixture
def agent():
    """Create DataExtractionAgent instance for testing."""
    return DataExtractionAgent(cache_extractions=False)  # Disable caching for tests


def create_test_request(
    operation: str = "extract_relevant_data",
    parameters: Dict[str, Any] | None = None,
    task_id: str = "test_task_123"
) -> AgentExecutionRequest:
    """Helper to create properly structured AgentExecutionRequest."""
    if parameters is None:
        parameters = {"input_folder": "/tmp/test", "objective": "test"}
    
    return {
        "task_id": task_id,
        "task_description": f"Test {operation} operation",
        "task_type": "atomic",
        "operation": operation,
        "parameters": parameters,
        "input_data": {},
        "temp_folder": "/tmp/test",
        "output_folder": "/tmp/output",
        "cache_enabled": False,
        "blackboard": [],
        "relevant_entries": [],
        "max_retries": 3
    }


# ==================== Test 1-3: Calling Convention Tests ====================

def test_execute_task_legacy_positional(agent, test_files):
    """Test execute_task with legacy positional arguments (operation, parameters)."""
    result = agent.execute_task("extract_relevant_data", {
        "input_folder": str(test_files),
        "objective": "Python programming",
        "max_files": 5
    })
    
    # Should return legacy dict format
    assert isinstance(result, dict)
    assert "success" in result
    assert result["success"] is True
    assert "files_found" in result
    assert "extractions" in result


def test_execute_task_legacy_dict(agent, test_files):
    """Test execute_task with legacy dict format."""
    result = agent.execute_task({
        "operation": "get_file_preview",
        "parameters": {
            "file_path": str(test_files / "test.txt"),
            "max_chars": 100
        }
    })
    
    # Should return legacy dict format
    assert isinstance(result, dict)
    assert "success" in result
    assert "content" in result


def test_execute_task_standardized_request(agent, test_files):
    """Test execute_task with standardized AgentExecutionRequest."""
    request = create_test_request(
        operation="extract_relevant_data",
        parameters={
            "input_folder": str(test_files),
            "objective": "test data",
            "max_files": 3
        }
    )
    
    result = agent.execute_task(request)
    
    # Should return AgentExecutionResponse format
    assert isinstance(result, dict)
    assert "status" in result
    assert result["status"] == "success"
    assert result["success"] is True
    assert "agent_name" in result
    assert result["agent_name"] == "data_extraction_agent"
    assert "operation" in result
    assert result["operation"] == "extract_relevant_data"


# ==================== Test 4: Response Validation ====================

def test_standardized_response_validation(agent, test_files):
    """Test that standardized response has all required fields."""
    request = create_test_request(
        operation="get_file_preview",
        parameters={"file_path": str(test_files / "test.txt")}
    )
    
    result = agent.execute_task(request)
    
    # Check all required AgentExecutionResponse fields
    required_fields = [
        "status", "success", "result", "artifacts", "execution_time_ms",
        "timestamp", "agent_name", "operation", "blackboard_entries", "warnings"
    ]
    for field in required_fields:
        assert field in result, f"Missing required field: {field}"
    
    # Validate field types
    assert isinstance(result["status"], str)
    assert isinstance(result["success"], bool)
    assert isinstance(result["result"], dict)
    assert isinstance(result["artifacts"], list)
    assert isinstance(result["execution_time_ms"], int)
    assert isinstance(result["timestamp"], str)
    assert isinstance(result["agent_name"], str)
    assert isinstance(result["operation"], str)
    assert isinstance(result["blackboard_entries"], list)
    assert isinstance(result["warnings"], list)


# ==================== Test 5: Event Publication ====================

def test_event_publication(agent, test_files):
    """Test that completion events are published correctly."""
    request = create_test_request(
        operation="extract_relevant_data",
        parameters={"input_folder": str(test_files), "objective": "test"}
    )
    
    # Mock the event bus
    mock_event_bus = Mock()
    agent.event_bus = mock_event_bus
    
    result = agent.execute_task(request)
    
    # Verify event was published
    assert mock_event_bus.publish.called
    published_event = mock_event_bus.publish.call_args[0][0]
    
    # Validate event structure
    assert published_event["event_type"] == "data_extracted"
    assert published_event["event_category"] == "task_lifecycle"
    assert published_event["source_agent"] == "data_extraction_agent"
    assert "payload" in published_event
    assert published_event["payload"]["task_id"] == "test_task_123"


# ==================== Test 6: Blackboard Entries ====================

def test_blackboard_entries_for_extraction(agent, test_files):
    """Test blackboard entries are created for extract_relevant_data operation."""
    request = create_test_request(
        operation="extract_relevant_data",
        parameters={
            "input_folder": str(test_files),
            "objective": "Python programming",
            "max_files": 5
        },
        task_id="extract_123"
    )
    
    result = agent.execute_task(request)
    
    # Check blackboard entries
    assert len(result["blackboard_entries"]) >= 1
    
    # Find the extracted_data entry
    data_entry = next(
        (entry for entry in result["blackboard_entries"] if "extracted_data" in entry["key"]),
        None
    )
    assert data_entry is not None
    assert data_entry["key"] == "extracted_data_extract_123"
    assert data_entry["scope"] == "workflow"
    assert data_entry["ttl_seconds"] == 3600


def test_blackboard_entries_for_preview(agent, test_files):
    """Test blackboard entries for get_file_preview operation."""
    request = create_test_request(
        operation="get_file_preview",
        parameters={"file_path": str(test_files / "test.txt")},
        task_id="preview_123"
    )
    
    result = agent.execute_task(request)
    
    # Should have blackboard entry with file content
    if result["success"]:
        assert len(result["blackboard_entries"]) >= 1
        entry = result["blackboard_entries"][0]
        assert "file_preview" in entry["key"]
        assert entry["scope"] == "workflow"


def test_blackboard_entries_for_search(agent, test_files):
    """Test blackboard entries for search_in_files operation."""
    request = create_test_request(
        operation="search_in_files",
        parameters={
            "input_folder": str(test_files),
            "search_query": "Python"
        },
        task_id="search_123"
    )
    
    result = agent.execute_task(request)
    
    # Should have blackboard entry with search results
    if result["success"] and result["result"].get("results"):
        assert len(result["blackboard_entries"]) >= 1
        entry = result["blackboard_entries"][0]
        assert "search_results" in entry["key"]


# ==================== Test 7: All Operations ====================

def test_operation_extract_relevant_data(agent, test_files):
    """Test extract_relevant_data operation."""
    result = agent.execute_task("extract_relevant_data", {
        "input_folder": str(test_files),
        "objective": "Python programming",
        "max_files": 10
    })
    
    assert result["success"] is True
    assert "files_found" in result
    assert result["files_found"] >= 1  # At least test.txt should be found
    assert "extractions" in result
    assert "summary" in result


def test_operation_get_file_preview(agent, test_files):
    """Test get_file_preview operation."""
    result = agent.execute_task("get_file_preview", {
        "file_path": str(test_files / "test.txt"),
        "max_chars": 100
    })
    
    assert result["success"] is True
    assert "content" in result
    assert "file_name" in result
    assert result["file_name"] == "test.txt"


def test_operation_search_in_files(agent, test_files):
    """Test search_in_files operation."""
    result = agent.execute_task("search_in_files", {
        "input_folder": str(test_files),
        "search_query": "Python",
        "max_results": 5
    })
    
    assert result["success"] is True
    assert "results" in result
    # Should find "Python" in test.txt
    assert len(result["results"]) >= 1


def test_unknown_operation(agent):
    """Test handling of unknown operation."""
    result = agent.execute_task("unknown_operation", {})
    
    # Should return error
    assert result["success"] is False
    assert "error" in result
    assert "Unknown operation" in result["error"]


# ==================== Test 8: Error Handling ====================

def test_error_handling_file_not_found(agent):
    """Test error handling for missing file."""
    request = create_test_request(
        operation="get_file_preview",
        parameters={"file_path": "/nonexistent/file.txt"}
    )
    
    result = agent.execute_task(request)
    
    # Should return error response
    assert result["success"] is False
    assert "error" in result


def test_error_handling_missing_folder(agent):
    """Test error handling for missing input folder."""
    result = agent.execute_task("extract_relevant_data", {
        "input_folder": "/nonexistent/folder",
        "objective": "test"
    })
    
    # Should handle gracefully (may return success with 0 files or error)
    assert isinstance(result, dict)
    assert "success" in result


def test_error_handling_standardized(agent):
    """Test error handling returns proper error response in standardized format."""
    request = create_test_request(
        operation="get_file_preview",
        parameters={}  # Missing required file_path
    )
    
    result = agent.execute_task(request)
    
    # Should still return a response
    assert "status" in result
    assert "success" in result


# ==================== Test 9: Invalid Call Detection ====================

def test_invalid_call_raises_error(agent):
    """Test that invalid calling convention raises ValueError."""
    with pytest.raises(ValueError, match="Invalid call to execute_task"):
        # Call with no arguments
        agent.execute_task()


# ==================== Test 10: Backward Compatibility ====================

def test_backward_compatibility_complete(agent, test_files):
    """Test that legacy code works exactly as before."""
    # Old positional call
    result1 = agent.execute_task("get_file_preview", {
        "file_path": str(test_files / "test.txt")
    })
    
    # Old dict call
    result2 = agent.execute_task({
        "operation": "get_file_preview",
        "parameters": {"file_path": str(test_files / "test.txt")}
    })
    
    # Both should return legacy format
    assert isinstance(result1, dict)
    assert isinstance(result2, dict)
    assert "success" in result1
    assert "success" in result2
    
    # Neither should have standardized fields like "status", "agent_name"
    assert "status" not in result1
    assert "agent_name" not in result1
    assert "status" not in result2
    assert "agent_name" not in result2


# ==================== Test 11: Event Types for Different Operations ====================

def test_event_types_for_operations(agent, test_files):
    """Test that different operations publish appropriate event types."""
    mock_event_bus = Mock()
    agent.event_bus = mock_event_bus
    
    operations_and_events = [
        ("extract_relevant_data", "data_extracted", {
            "input_folder": str(test_files), "objective": "test"
        }),
        ("get_file_preview", "file_preview_ready", {
            "file_path": str(test_files / "test.txt")
        }),
        ("search_in_files", "search_completed", {
            "input_folder": str(test_files), "search_query": "test"
        }),
    ]
    
    for operation, expected_event, params in operations_and_events:
        mock_event_bus.reset_mock()
        
        request = create_test_request(operation=operation, parameters=params)
        agent.execute_task(request)
        
        event = mock_event_bus.publish.call_args[0][0]
        assert event["event_type"] == expected_event, f"Wrong event for {operation}"


# ==================== Test 12: File Type Detection ====================

def test_extract_handles_different_file_types(agent, test_files):
    """Test that extraction handles different file types."""
    result = agent.execute_task("extract_relevant_data", {
        "input_folder": str(test_files),
        "objective": "data",
        "max_files": 10
    })
    
    assert result["success"] is True
    # Should find multiple file types (.txt, .json, .csv)
    assert result["files_found"] >= 3


# ==================== Test 13: Search Functionality ====================

def test_search_finds_exact_match(agent, test_files):
    """Test search finds exact string matches."""
    result = agent.execute_task("search_in_files", {
        "input_folder": str(test_files),
        "search_query": "Python programming",
        "max_results": 10
    })
    
    assert result["success"] is True
    if result["results"]:
        # Should find exact match
        assert any(r["match_type"] == "exact" for r in result["results"])


def test_search_partial_match(agent, test_files):
    """Test search finds partial word matches."""
    result = agent.execute_task("search_in_files", {
        "input_folder": str(test_files),
        "search_query": "test data",
        "max_results": 10
    })
    
    assert result["success"] is True
    assert "results" in result


# ==================== Test 14: Conversion Helpers ====================

def test_convert_to_legacy_response(agent, test_files):
    """Test _convert_to_legacy_response preserves legacy structure."""
    request = create_test_request(
        operation="get_file_preview",
        parameters={"file_path": str(test_files / "test.txt")}
    )
    
    # Get standardized response
    standardized = agent.execute_task(request)
    
    # Convert to legacy
    legacy = agent._convert_to_legacy_response(standardized)
    
    # Check legacy format
    assert isinstance(legacy, dict)
    assert "success" in legacy
    # Should have file preview fields
    assert "content" in legacy or "file_name" in legacy


# ==================== Test 15: Empty Folder Handling ====================

def test_extract_empty_folder(agent, temp_dir):
    """Test extraction from empty folder returns gracefully."""
    empty_dir = temp_dir / "empty"
    empty_dir.mkdir()
    
    result = agent.execute_task("extract_relevant_data", {
        "input_folder": str(empty_dir),
        "objective": "test"
    })
    
    assert result["success"] is True
    assert result["files_found"] == 0
    assert result["files_processed"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
