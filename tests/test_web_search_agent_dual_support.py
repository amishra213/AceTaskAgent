"""
Test suite for WebSearchAgent dual-format support (Week 7 Day 2).

Tests:
1. Legacy positional call format
2. Legacy dict call format
3. Standardized AgentExecutionRequest format
4. Response validation
5. Event publication
6. Artifact generation
7. Blackboard entries
8. Operation alias normalization
9. Parameter alias mapping
10. Error handling
11. All 9 operations (search, scrape, fetch, etc.)
"""

import pytest
from pathlib import Path
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from task_manager.sub_agents.web_search_agent import WebSearchAgent
from task_manager.models import AgentExecutionRequest, AgentExecutionResponse
# Logger not needed for tests


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_search_results():
    """Mock search results for testing."""
    return {
        "success": True,
        "query": "test query",
        "results": [
            {"title": "Result 1", "url": "https://example.com/1", "snippet": "Test snippet 1"},
            {"title": "Result 2", "url": "https://example.com/2", "snippet": "Test snippet 2"}
        ],
        "count": 2
    }


@pytest.fixture
def mock_scrape_results():
    """Mock scrape results for testing."""
    return {
        "success": True,
        "url": "https://example.com",
        "text": "Scraped content here",
        "links": ["https://example.com/link1", "https://example.com/link2"],
        "title": "Example Page"
    }


@pytest.fixture
def agent():
    """Create WebSearchAgent instance for testing."""
    return WebSearchAgent()


def create_test_request(
    operation: str = "search",
    parameters: Dict[str, Any] | None = None,
    task_id: str = "test_task_123"
) -> AgentExecutionRequest:
    """Helper to create properly structured AgentExecutionRequest."""
    if parameters is None:
        parameters = {"query": "test query"}
    
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

def test_execute_task_legacy_positional(agent, mock_search_results):
    """Test execute_task with legacy positional arguments (operation, parameters)."""
    with patch.object(agent, 'search', return_value=mock_search_results):
        result = agent.execute_task("search", {"query": "test query", "max_results": 5})
    
    # Should return legacy dict format
    assert isinstance(result, dict)
    assert result["success"] is True
    assert "query" in result
    assert result["query"] == "test query"
    assert "results" in result
    assert len(result["results"]) == 2


def test_execute_task_legacy_dict(agent, mock_search_results):
    """Test execute_task with legacy dict format."""
    with patch.object(agent, 'search', return_value=mock_search_results):
        result = agent.execute_task({
            "operation": "search",
            "parameters": {"query": "test query", "max_results": 5}
        })
    
    # Should return legacy dict format
    assert isinstance(result, dict)
    assert result["success"] is True
    assert "results" in result


def test_execute_task_standardized_request(agent, mock_search_results):
    """Test execute_task with standardized AgentExecutionRequest."""
    request = create_test_request(
        operation="search",
        parameters={"query": "test query", "max_results": 5}
    )
    
    with patch.object(agent, 'search', return_value=mock_search_results):
        result = agent.execute_task(request)
    
    # Should return AgentExecutionResponse format
    assert isinstance(result, dict)
    assert "status" in result
    assert result["status"] == "success"
    assert result["success"] is True
    assert "agent_name" in result
    assert result["agent_name"] == "web_search_agent"
    assert "operation" in result
    assert result["operation"] == "search"
    assert "execution_time_ms" in result
    assert "timestamp" in result
    assert "artifacts" in result
    assert "blackboard_entries" in result


# ==================== Test 4: Response Validation ====================

def test_standardized_response_validation(agent, mock_search_results):
    """Test that standardized response has all required fields."""
    request = create_test_request(operation="search")
    
    with patch.object(agent, 'search', return_value=mock_search_results):
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

def test_event_publication(agent, mock_search_results):
    """Test that completion events are published correctly."""
    request = create_test_request(operation="search")
    
    # Mock the event bus
    mock_event_bus = Mock()
    agent.event_bus = mock_event_bus
    
    with patch.object(agent, 'search', return_value=mock_search_results):
        result = agent.execute_task(request)
    
    # Verify event was published
    assert mock_event_bus.publish.called
    published_event = mock_event_bus.publish.call_args[0][0]
    
    # Validate event structure
    assert published_event["event_type"] == "web_search_completed"
    assert published_event["event_category"] == "task_lifecycle"
    assert published_event["source_agent"] == "web_search_agent"
    assert "payload" in published_event
    assert published_event["payload"]["task_id"] == "test_task_123"
    assert published_event["payload"]["operation"] == "search"
    assert published_event["payload"]["success"] is True


# ==================== Test 6-7: Artifacts and Blackboard ====================

def test_artifact_generation_csv(agent, temp_dir):
    """Test artifact generation for CSV output files."""
    # Create a mock CSV file
    csv_file = temp_dir / "search_results.csv"
    csv_file.write_text("title,url\nResult 1,https://example.com/1\n")
    
    mock_result = {
        "success": True,
        "query": "test query",
        "results": [{"title": "Result 1", "url": "https://example.com/1"}],
        "output_file": str(csv_file)
    }
    
    request = create_test_request(operation="search")
    
    with patch.object(agent, 'search', return_value=mock_result):
        result = agent.execute_task(request)
    
    # Check artifacts
    assert len(result["artifacts"]) == 1
    artifact = result["artifacts"][0]
    assert artifact["type"] == "csv"
    assert artifact["path"] == str(csv_file)
    assert artifact["size_bytes"] > 0
    assert "description" in artifact


def test_blackboard_entries_creation(agent, mock_search_results):
    """Test blackboard entries are created for data sharing."""
    request = create_test_request(operation="search", task_id="search_123")
    
    with patch.object(agent, 'search', return_value=mock_search_results):
        result = agent.execute_task(request)
    
    # Check blackboard entries
    assert len(result["blackboard_entries"]) >= 1
    
    # Find the results entry
    results_entry = next(
        (entry for entry in result["blackboard_entries"] if "results" in entry["key"]),
        None
    )
    assert results_entry is not None
    assert results_entry["key"] == "websearch_results_search_123"
    assert results_entry["scope"] == "workflow"
    assert results_entry["ttl_seconds"] == 3600
    assert isinstance(results_entry["value"], list)


# ==================== Test 8-9: Alias Normalization ====================

def test_operation_alias_normalization(agent, mock_search_results):
    """Test that operation aliases are normalized correctly."""
    # Test various search aliases
    aliases = ['web_search', 'websearch', 'search_web']
    
    for alias in aliases:
        with patch.object(agent, 'search', return_value=mock_search_results) as mock_search:
            result = agent.execute_task(alias, {"query": "test"})
            
            # Verify the search method was called (not the alias)
            assert mock_search.called
            assert result["success"] is True


def test_parameter_alias_mapping(agent, mock_search_results):
    """Test that parameter aliases are mapped correctly."""
    # Test num_results -> max_results mapping
    with patch.object(agent, 'search', return_value=mock_search_results) as mock_search:
        result = agent.execute_task("search", {"query": "test", "num_results": 5})
        
        # Verify search was called with max_results
        call_kwargs = mock_search.call_args[1]
        assert 'max_results' in call_kwargs
        assert call_kwargs['max_results'] == 5


# ==================== Test 10: Error Handling ====================

def test_error_handling_standardized(agent):
    """Test error handling returns proper error response."""
    # Mock search to raise an exception
    with patch.object(agent, 'search', side_effect=Exception("Test error")):
        request = create_test_request(operation="search")
        result = agent.execute_task(request)
    
    # Should return standardized error response
    assert result["status"] == "failure"
    assert result["success"] is False
    assert "error" in result
    assert result["error"]["message"] == "Test error"
    assert result["error"]["source"] == "web_search_agent"


def test_error_handling_legacy(agent):
    """Test error handling returns legacy error format."""
    # Mock search to raise an exception
    with patch.object(agent, 'search', side_effect=Exception("Test error")):
        result = agent.execute_task("search", {"query": "test"})
    
    # Should return legacy error dict
    assert isinstance(result, dict)
    assert result["success"] is False
    assert "error" in result
    assert "Test error" in result["error"]


# ==================== Test 11: All Operations ====================

def test_operation_search(agent, mock_search_results):
    """Test search operation."""
    with patch.object(agent, 'search', return_value=mock_search_results):
        result = agent.execute_task("search", {"query": "test", "max_results": 5})
    
    assert result["success"] is True
    assert "results" in result


def test_operation_scrape(agent, mock_scrape_results):
    """Test scrape operation."""
    with patch.object(agent, 'scrape', return_value=mock_scrape_results):
        result = agent.execute_task("scrape", {"url": "https://example.com"})
    
    assert result["success"] is True
    assert "text" in result


def test_operation_fetch(agent):
    """Test fetch operation."""
    mock_result = {"success": True, "url": "https://example.com", "content": "Fetched content"}
    
    with patch.object(agent, 'fetch', return_value=mock_result):
        result = agent.execute_task("fetch", {"url": "https://example.com"})
    
    assert result["success"] is True
    assert "content" in result


def test_operation_deep_search(agent):
    """Test deep_search operation."""
    mock_result = {
        "success": True,
        "query": "test query",
        "results": [{"title": "Deep result", "url": "https://example.com", "relevance_score": 0.9}],
        "structured_data": {}
    }
    
    with patch.object(agent, 'deep_search', return_value=mock_result):
        result = agent.execute_task("deep_search", {"query": "test query", "max_depth": 2})
    
    assert result["success"] is True
    assert "results" in result


def test_operation_research(agent):
    """Test research operation."""
    mock_result = {
        "success": True,
        "topic": "test topic",
        "summary": "Research summary here",
        "sources": [{"url": "https://example.com", "title": "Source 1"}]
    }
    
    with patch.object(agent, 'research', return_value=mock_result):
        result = agent.execute_task("research", {"topic": "test topic", "max_sources": 5})
    
    assert result["success"] is True
    assert "summary" in result


def test_unknown_operation(agent):
    """Test handling of unknown operation."""
    result = agent.execute_task("unknown_operation", {})
    
    # Should return error
    assert result["success"] is False
    assert "error" in result
    assert "Unknown operation" in result["error"]


# ==================== Test 12: Invalid Call Detection ====================

def test_invalid_call_raises_error(agent):
    """Test that invalid calling convention raises ValueError."""
    with pytest.raises(ValueError, match="Invalid call to execute_task"):
        # Call with no arguments
        agent.execute_task()


# ==================== Test 13: Backward Compatibility ====================

def test_backward_compatibility_complete(agent, mock_search_results):
    """Test that legacy code works exactly as before."""
    # Simulate old code calling pattern
    with patch.object(agent, 'search', return_value=mock_search_results):
        # Old positional call
        result1 = agent.execute_task("search", {"query": "test"})
        
        # Old dict call
        result2 = agent.execute_task({"operation": "search", "parameters": {"query": "test"}})
    
    # Both should return legacy format
    assert isinstance(result1, dict)
    assert isinstance(result2, dict)
    assert result1["success"] is True
    assert result2["success"] is True
    
    # Neither should have standardized fields like "status", "agent_name"
    assert "status" not in result1
    assert "agent_name" not in result1
    assert "status" not in result2
    assert "agent_name" not in result2


# ==================== Test 14: Screenshot Operation ====================

def test_operation_screenshot(agent, temp_dir):
    """Test capture_screenshot operation with artifact."""
    screenshot_path = temp_dir / "screenshot.png"
    screenshot_path.write_bytes(b"fake_png_data")
    
    mock_result = {
        "success": True,
        "url": "https://example.com",
        "screenshot_path": str(screenshot_path)
    }
    
    request = create_test_request(
        operation="capture_screenshot",
        parameters={"url": "https://example.com"}
    )
    
    with patch.object(agent, 'capture_screenshot', return_value=mock_result):
        result = agent.execute_task(request)
    
    # Check screenshot artifact
    assert result["success"] is True
    assert len(result["artifacts"]) == 1
    artifact = result["artifacts"][0]
    assert artifact["type"] == "png"
    assert artifact["path"] == str(screenshot_path)


# ==================== Test 15: Event Types for Different Operations ====================

def test_event_types_for_operations(agent, mock_search_results, mock_scrape_results):
    """Test that different operations publish appropriate event types."""
    mock_event_bus = Mock()
    agent.event_bus = mock_event_bus
    
    # Test search operation
    with patch.object(agent, 'search', return_value=mock_search_results):
        request = create_test_request(operation="search")
        agent.execute_task(request)
    
    event = mock_event_bus.publish.call_args[0][0]
    assert event["event_type"] == "web_search_completed"
    
    # Test scrape operation
    mock_event_bus.reset_mock()
    with patch.object(agent, 'scrape', return_value=mock_scrape_results):
        request = create_test_request(operation="scrape")
        agent.execute_task(request)
    
    event = mock_event_bus.publish.call_args[0][0]
    assert event["event_type"] == "web_scrape_completed"


# ==================== Test 16: Conversion Helpers ====================

def test_convert_to_legacy_response(agent, mock_search_results):
    """Test _convert_to_legacy_response preserves legacy structure."""
    request = create_test_request(operation="search")
    
    with patch.object(agent, 'search', return_value=mock_search_results):
        # Get standardized response
        standardized = agent.execute_task(request)
        
        # Convert to legacy
        legacy = agent._convert_to_legacy_response(standardized)
    
    # Check legacy format
    assert isinstance(legacy, dict)
    assert "success" in legacy
    assert legacy["success"] is True
    
    # Should have result fields at top level
    assert "query" in legacy or "results" in legacy


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
