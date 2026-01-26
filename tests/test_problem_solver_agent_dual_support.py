"""
Test suite for ProblemSolverAgent dual-format support (Week 7 Day 2).

Tests:
1. Legacy positional call format
2. Legacy dict call format
3. Standardized AgentExecutionRequest format
4. Response validation
5. Event publication
6. Blackboard entries
7. All 6 operations (diagnose_error, get_solution, interpret_human_input, etc.)
8. Error handling
9. Backward compatibility
"""

import pytest
from pathlib import Path
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from task_manager.sub_agents.problem_solver_agent import ProblemSolverAgent
from task_manager.models import AgentExecutionRequest, AgentExecutionResponse


@pytest.fixture
def agent():
    """Create ProblemSolverAgent instance for testing."""
    return ProblemSolverAgent()


@pytest.fixture
def agent_with_llm():
    """Create ProblemSolverAgent with mocked LLM client."""
    mock_llm = Mock()
    agent = ProblemSolverAgent(llm_client=mock_llm)
    return agent


def create_test_request(
    operation: str = "diagnose_error",
    parameters: Dict[str, Any] | None = None,
    task_id: str = "test_task_123"
) -> AgentExecutionRequest:
    """Helper to create properly structured AgentExecutionRequest."""
    if parameters is None:
        parameters = {"error_message": "Test error"}
    
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

def test_execute_task_legacy_positional(agent):
    """Test execute_task with legacy positional arguments (operation, parameters)."""
    result = agent.execute_task("diagnose_error", {
        "error_message": "File not found: test.txt",
        "agent_type": "excel_agent"
    })
    
    # Should return legacy dict format
    assert isinstance(result, dict)
    assert "error_message" in result
    assert "error_category" in result
    assert result["error_category"] == "file_not_found"


def test_execute_task_legacy_dict(agent):
    """Test execute_task with legacy dict format."""
    result = agent.execute_task({
        "operation": "diagnose_error",
        "parameters": {
            "error_message": "Permission denied",
            "agent_type": "pdf_agent"
        }
    })
    
    # Should return legacy dict format
    assert isinstance(result, dict)
    assert "error_category" in result
    assert result["error_category"] == "permission_denied"


def test_execute_task_standardized_request(agent):
    """Test execute_task with standardized AgentExecutionRequest."""
    request = create_test_request(
        operation="diagnose_error",
        parameters={
            "error_message": "API rate limit exceeded",
            "agent_type": "web_search_agent"
        }
    )
    
    result = agent.execute_task(request)
    
    # Should return AgentExecutionResponse format
    assert isinstance(result, dict)
    assert "status" in result
    assert result["status"] == "success"
    assert result["success"] is True
    assert "agent_name" in result
    assert result["agent_name"] == "problem_solver_agent"
    assert "operation" in result
    assert result["operation"] == "diagnose_error"
    assert "execution_time_ms" in result
    assert "timestamp" in result


# ==================== Test 4: Response Validation ====================

def test_standardized_response_validation(agent):
    """Test that standardized response has all required fields."""
    request = create_test_request(operation="diagnose_error")
    
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

def test_event_publication(agent):
    """Test that completion events are published correctly."""
    request = create_test_request(operation="diagnose_error")
    
    # Mock the event bus
    mock_event_bus = Mock()
    agent.event_bus = mock_event_bus
    
    result = agent.execute_task(request)
    
    # Verify event was published
    assert mock_event_bus.publish.called
    published_event = mock_event_bus.publish.call_args[0][0]
    
    # Validate event structure
    assert published_event["event_type"] == "error_diagnosed"
    assert published_event["event_category"] == "task_lifecycle"
    assert published_event["source_agent"] == "problem_solver_agent"
    assert "payload" in published_event
    assert published_event["payload"]["task_id"] == "test_task_123"
    assert published_event["payload"]["operation"] == "diagnose_error"
    assert published_event["payload"]["success"] is True


# ==================== Test 6: Blackboard Entries ====================

def test_blackboard_entries_for_solution(agent_with_llm):
    """Test blackboard entries are created for get_solution operation."""
    # Mock LLM response
    mock_response = Mock()
    mock_response.content = json.dumps({
        "solution_type": "retry",
        "explanation": "Test solution",
        "suggested_action": "Retry with modified params",
        "modified_parameters": {"timeout": 30},
        "alternative_approaches": [],
        "requires_human_input": False,
        "confidence": 0.8
    })
    agent_with_llm.llm_client.invoke.return_value = mock_response
    
    request = create_test_request(
        operation="get_solution",
        parameters={
            "error_message": "Timeout error",
            "agent_type": "web_search_agent"
        },
        task_id="solution_123"
    )
    
    result = agent_with_llm.execute_task(request)
    
    # Check blackboard entries
    assert len(result["blackboard_entries"]) >= 1
    
    # Find the solution entry
    solution_entry = next(
        (entry for entry in result["blackboard_entries"] if "solution" in entry["key"]),
        None
    )
    assert solution_entry is not None
    assert solution_entry["key"] == "solution_solution_123"
    assert solution_entry["scope"] == "workflow"
    assert solution_entry["ttl_seconds"] == 3600


def test_blackboard_entries_for_interpret_input(agent):
    """Test blackboard entries for interpret_human_input operation."""
    request = create_test_request(
        operation="interpret_human_input",
        parameters={
            "human_input": "Search for Python tutorials and save to results.csv",
            "target_format": "web_search_params"
        },
        task_id="interpret_123"
    )
    
    result = agent.execute_task(request)
    
    # Should have blackboard entry with parsed data
    if result["success"] and result["result"].get("parsed_data"):
        assert len(result["blackboard_entries"]) >= 1
        entry = result["blackboard_entries"][0]
        assert "interpreted_input" in entry["key"]


# ==================== Test 7: All Operations ====================

def test_operation_diagnose_error(agent):
    """Test diagnose_error operation."""
    result = agent.execute_task("diagnose_error", {
        "error_message": "worksheet not found: 'Data'",  # Use lowercase to match pattern
        "agent_type": "excel_agent"
    })
    
    assert "error_category" in result
    assert result["error_category"] == "sheet_not_found"
    assert "solution_prompt" in result


def test_operation_get_solution_without_llm(agent):
    """Test get_solution operation without LLM (template solution)."""
    result = agent.execute_task("get_solution", {
        "error_message": "File not found: data.xlsx",
        "agent_type": "excel_agent"
    })
    
    # Should return template solution
    assert "solution_type" in result
    assert "explanation" in result
    assert "suggested_action" in result


def test_operation_get_solution_with_llm(agent_with_llm):
    """Test get_solution operation with LLM."""
    # Mock LLM response
    mock_response = Mock()
    mock_response.content = json.dumps({
        "solution_type": "modify_params",
        "explanation": "File path is incorrect",
        "suggested_action": "Use correct file path",
        "modified_parameters": {"file_path": "correct/path.xlsx"},
        "alternative_approaches": ["Create the file", "Use different file"],
        "requires_human_input": False,
        "confidence": 0.9
    })
    agent_with_llm.llm_client.invoke.return_value = mock_response
    
    result = agent_with_llm.execute_task("get_solution", {
        "error_message": "File not found",
        "agent_type": "excel_agent"
    })
    
    assert result["solution_type"] == "modify_params"
    assert result["confidence"] == 0.9
    assert "modified_parameters" in result


def test_operation_interpret_human_input(agent):
    """Test interpret_human_input operation (rule-based without LLM)."""
    result = agent.execute_task("interpret_human_input", {
        "human_input": "Create an Excel file with data from search results",
        "target_format": "excel_params"
    })
    
    assert "success" in result
    assert "parsed_data" in result
    assert "confidence" in result


def test_operation_format_for_agent(agent):
    """Test format_for_agent operation."""
    test_data = {"query": "Python tutorials", "max_results": 10}
    
    result = agent.execute_task("format_for_agent", {
        "data": test_data,
        "agent_type": "web_search"
    })
    
    # Should return formatted parameters
    assert isinstance(result, dict)
    # Format depends on agent type, check it's been processed
    assert result is not None


def test_operation_generate_retry_parameters(agent):
    """Test generate_retry_parameters operation."""
    failed_task = {
        "id": "task_456",
        "operation": "search",
        "parameters": {"query": "test"}
    }
    error_info = {
        "error": "Timeout",
        "agent_type": "web_search_agent"
    }
    
    result = agent.execute_task("generate_retry_parameters", {
        "failed_task": failed_task,
        "error_info": error_info
    })
    
    assert "original_task" in result
    assert "retry_strategy" in result
    assert "modified_parameters" in result
    assert "confidence" in result


def test_operation_create_task_output(agent):
    """Test create_task_output operation."""
    result = agent.execute_task("create_task_output", {
        "human_input": "The total revenue is $150,000",
        "task_description": "Calculate total revenue from sales data",
        "expected_output_type": "numeric"
    })
    
    # Without LLM, returns simple output
    assert "output" in result
    assert "human_provided" in result.get("metadata", {})


def test_unknown_operation(agent):
    """Test handling of unknown operation."""
    result = agent.execute_task("unknown_operation", {})
    
    # Should return error
    assert result["success"] is False
    assert "error" in result
    assert "Unknown operation" in result["error"]


# ==================== Test 8: Error Handling ====================

def test_error_handling_standardized(agent):
    """Test error handling returns proper error response."""
    # Trigger an error by passing invalid parameters
    request = create_test_request(
        operation="interpret_human_input",
        parameters={}  # Missing required 'human_input'
    )
    
    result = agent.execute_task(request)
    
    # Should still return a response (may succeed with empty input or fail gracefully)
    assert "status" in result
    assert "success" in result


def test_error_handling_legacy(agent):
    """Test error handling returns legacy error format."""
    result = agent.execute_task("unknown_op", {})
    
    # Should return legacy error dict
    assert isinstance(result, dict)
    assert result["success"] is False
    assert "error" in result


# ==================== Test 9: Invalid Call Detection ====================

def test_invalid_call_raises_error(agent):
    """Test that invalid calling convention raises ValueError."""
    with pytest.raises(ValueError, match="Invalid call to execute_task"):
        # Call with no arguments
        agent.execute_task()


# ==================== Test 10: Backward Compatibility ====================

def test_backward_compatibility_complete(agent):
    """Test that legacy code works exactly as before."""
    # Simulate old code calling pattern
    # Old positional call
    result1 = agent.execute_task("diagnose_error", {
        "error_message": "Test error",
        "agent_type": "test_agent"
    })
    
    # Old dict call
    result2 = agent.execute_task({
        "operation": "diagnose_error",
        "parameters": {
            "error_message": "Test error",
            "agent_type": "test_agent"
        }
    })
    
    # Both should return legacy format
    assert isinstance(result1, dict)
    assert isinstance(result2, dict)
    assert "error_category" in result1
    assert "error_category" in result2
    
    # Neither should have standardized fields like "status", "agent_name"
    assert "status" not in result1
    assert "agent_name" not in result1
    assert "status" not in result2
    assert "agent_name" not in result2


# ==================== Test 11: Event Types for Different Operations ====================

def test_event_types_for_operations(agent):
    """Test that different operations publish appropriate event types."""
    mock_event_bus = Mock()
    agent.event_bus = mock_event_bus
    
    operations_and_events = [
        ("diagnose_error", "error_diagnosed"),
        ("get_solution", "solution_generated"),
        ("interpret_human_input", "human_input_interpreted"),
        ("format_for_agent", "data_formatted"),
        ("generate_retry_parameters", "retry_params_generated"),
        ("create_task_output", "task_output_created"),
    ]
    
    for operation, expected_event in operations_and_events:
        mock_event_bus.reset_mock()
        
        params = {
            "diagnose_error": {"error_message": "test"},
            "get_solution": {"error_message": "test"},
            "interpret_human_input": {"human_input": "test", "target_format": "json"},
            "format_for_agent": {"data": {}, "agent_type": "excel"},
            "generate_retry_parameters": {"failed_task": {}, "error_info": {}},
            "create_task_output": {"human_input": "test", "task_description": "test"}
        }
        
        request = create_test_request(operation=operation, parameters=params[operation])
        agent.execute_task(request)
        
        event = mock_event_bus.publish.call_args[0][0]
        assert event["event_type"] == expected_event, f"Wrong event for {operation}"


# ==================== Test 12: Conversion Helpers ====================

def test_convert_to_legacy_response(agent):
    """Test _convert_to_legacy_response preserves legacy structure."""
    request = create_test_request(operation="diagnose_error")
    
    # Get standardized response
    standardized = agent.execute_task(request)
    
    # Convert to legacy
    legacy = agent._convert_to_legacy_response(standardized)
    
    # Check legacy format
    assert isinstance(legacy, dict)
    # Should have diagnostic fields
    assert "error_message" in legacy or "error_category" in legacy


# ==================== Test 13: LLM Integration ====================

def test_llm_client_can_be_set(agent):
    """Test that LLM client can be set after initialization."""
    mock_llm = Mock()
    agent.set_llm_client(mock_llm)
    
    assert agent.llm_client == mock_llm


def test_solution_with_llm_integration(agent):
    """Test get_solution with actual LLM call structure."""
    from langchain_core.messages import HumanMessage, SystemMessage
    
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.content = json.dumps({
        "solution_type": "retry",
        "explanation": "Network timeout occurred",
        "suggested_action": "Retry with increased timeout",
        "modified_parameters": {"timeout": 60},
        "alternative_approaches": ["Use different API endpoint"],
        "requires_human_input": False,
        "confidence": 0.85
    })
    mock_llm.invoke.return_value = mock_response
    
    agent.set_llm_client(mock_llm)
    
    result = agent.execute_task("get_solution", {
        "error_message": "Request timeout after 30 seconds",
        "agent_type": "web_search_agent"
    })
    
    # Verify LLM was called
    assert mock_llm.invoke.called
    call_args = mock_llm.invoke.call_args[0][0]
    assert len(call_args) == 2
    assert isinstance(call_args[0], SystemMessage)
    assert isinstance(call_args[1], HumanMessage)
    
    # Verify result
    assert result["solution_type"] == "retry"
    assert result["confidence"] == 0.85


# ==================== Test 14: Error Categories ====================

def test_error_category_detection(agent):
    """Test that various error types are correctly categorized."""
    error_tests = [
        ("File not found: data.xlsx", "file_not_found"),
        ("Permission denied accessing /root/file", "permission_denied"),
        ("Invalid path provided: ''", "invalid_path"),
        ("worksheet not found: 'Summary'", "sheet_not_found"),  # lowercase to match pattern
        ("API rate limit exceeded (429)", "api_error"),
        ("JSON decode error at line 5", "parse_error"),
        ("ModuleNotFoundError: No module named 'pandas'", "missing_dependency"),
        ("ValueError: invalid data type", "data_validation"),
        ("Some unknown error type", "unknown")
    ]
    
    for error_msg, expected_category in error_tests:
        result = agent.execute_task("diagnose_error", {
            "error_message": error_msg
        })
        
        assert result["error_category"] == expected_category, \
            f"Failed to categorize: {error_msg}"


# ==================== Test 15: Rule-Based Parsing ====================

def test_rule_based_excel_params_parsing(agent):
    """Test rule-based parsing for Excel parameters."""
    result = agent.execute_task("interpret_human_input", {
        "human_input": "Create an Excel file named report.xlsx with sheet Data",
        "target_format": "excel_params"
    })
    
    # Should extract file name and sheet name
    assert result["success"]
    parsed = result["parsed_data"]
    # Rule-based parser should find keywords
    assert isinstance(parsed, dict)


def test_rule_based_search_params_parsing(agent):
    """Test rule-based parsing for search parameters."""
    result = agent.execute_task("interpret_human_input", {
        "human_input": "Search for 'Python tutorials' max 20 results",
        "target_format": "web_search_params"
    })
    
    assert result["success"]
    parsed = result["parsed_data"]
    # Should extract query
    assert "query" in parsed or len(parsed) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
