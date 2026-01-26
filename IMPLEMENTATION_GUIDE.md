# Interface Standardization - Implementation Guide

## Quick Start

This guide shows how to migrate existing TaskManager code to use the new standardized interfaces.

---

## 1. Import the New Schemas

```python
# In any agent file
from task_manager.models import (
    AgentExecutionRequest,
    AgentExecutionResponse,
    ArtifactMetadata,
    SystemEvent,
    create_system_event,
    create_error_response
)
from task_manager.core.event_bus import get_event_bus, publish_event
```

---

## 2. Update Agent Execute Methods

### Before (Current Code):
```python
def execute_task(self, task: Task, state: AgentState) -> Dict[str, Any]:
    """Execute web search task."""
    # ... execution logic ...
    
    return {
        "success": True,
        "result": {"data": search_results},
        "file": output_csv_path
    }
```

### After (Standardized):
```python
def execute_task(self, task: Task, state: AgentState) -> AgentExecutionResponse:
    """Execute web search task."""
    start_time = datetime.now()
    
    # ... execution logic ...
    
    # Create standardized response
    response = AgentExecutionResponse(
        status="success",
        success=True,
        result={"data": search_results},
        artifacts=[
            ArtifactMetadata(
                type="csv",
                path=str(output_csv_path),
                size_bytes=os.path.getsize(output_csv_path),
                description="Web search results",
                mime_type="text/csv"
            )
        ],
        execution_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
        timestamp=datetime.now().isoformat(),
        agent_name="web_search_agent",
        operation="search",
        blackboard_entries=[],
        next_agents=["excel_agent"],  # Chain to Excel for analysis
        chain_data={"csv_file": str(output_csv_path)},
        event_triggers=["web_findings_ready"],
        error=None,
        warnings=[],
        confidence_score=0.95,
        completeness_score=1.0
    )
    
    return response
```

---

## 3. Publish Events Instead of State Flags

### Before (Current Code):
```python
# In _execute_ocr_task
return {
    **state,
    "last_updated_key": "ocr_results"  # Triggers observer via state flag
}
```

### After (Event-Driven):
```python
# In _execute_ocr_task
def _execute_ocr_task(self, state: AgentState) -> AgentState:
    # ... OCR execution ...
    
    # Publish event instead of state flag
    event = create_system_event(
        event_type="ocr_results_ready",
        event_category="data_flow",
        source_agent="ocr_image_agent",
        payload={
            "results": ocr_results,
            "confidence": 0.87,
            "tables_detected": True
        },
        source_task_id=current_task['id'],
        severity="info"
    )
    
    # Publish to event bus
    publish_event(event)
    
    # Also add to state event queue for LangGraph compatibility
    return {
        **state,
        "event_queue": [event]
    }
```

---

## 4. Subscribe to Events

### Option A: Programmatic Subscription
```python
from task_manager.core.event_bus import get_event_bus

# In agent __init__ or workflow setup
event_bus = get_event_bus()

def handle_ocr_results(event: SystemEvent):
    """Handle OCR completion events."""
    logger.info(f"OCR results received: {event['payload']}")
    # Trigger synthesis or Excel analysis
    if event['payload'].get('tables_detected'):
        trigger_excel_analysis(event)

# Subscribe
event_bus.subscribe(
    event_type="ocr_results_ready",
    handler=handle_ocr_results,
    subscriber_name="synthesis_agent",
    priority=1  # High priority
)
```

### Option B: Decorator-Based Subscription
```python
from task_manager.core.event_bus import event_handler

@event_handler("task_completed", subscriber_name="master_planner")
def on_task_complete(event: SystemEvent):
    """React to task completion."""
    task_id = event['payload']['task_id']
    logger.info(f"Task {task_id} completed!")
    # Update plan, trigger next tasks, etc.
```

---

## 5. Handle Errors Consistently

### Before:
```python
return {
    "success": False,
    "error": "Something went wrong",
    "file": file_path
}
```

### After:
```python
from task_manager.models import create_error_response

# Create standardized error
error = create_error_response(
    error_code="WEB_SEARCH_TIMEOUT_001",
    error_type="timeout_error",
    message="Web search timed out after 30 seconds",
    source="web_search_agent",
    severity="medium",
    details={
        "query": query,
        "timeout_seconds": 30,
        "partial_results": len(partial_results)
    },
    task_id=task['id'],
    operation="search",
    recoverable=True,
    recovery_suggestions=[
        "Retry with longer timeout",
        "Use cached results",
        "Switch to alternative backend"
    ],
    retry_after_seconds=60
)

# Publish error event
error_event = create_system_event(
    event_type="agent_execution_failed",
    event_category="agent_execution",
    source_agent="web_search_agent",
    payload=error,
    severity="error"
)
publish_event(error_event)

# Return error in response
return AgentExecutionResponse(
    status="failure",
    success=False,
    result={},
    artifacts=[],
    execution_time_ms=execution_time,
    timestamp=datetime.now().isoformat(),
    agent_name="web_search_agent",
    operation="search",
    blackboard_entries=[],
    error=error['message'],
    warnings=[],
    confidence_score=0.0,
    completeness_score=0.0
)
```

---

## 6. Update Temp Storage

### Before:
```python
self.temp_manager.save_data(
    category="cache",
    key="web_search_results",
    data={"results": search_results}
)
```

### After (with schema):
```python
from task_manager.models import TempDataSchema

temp_data = TempDataSchema(
    schema_version="1.0",
    data_type="cache_entry",
    created_at=datetime.now().isoformat(),
    updated_at=datetime.now().isoformat(),
    key="web_search_results",
    task_id=task['id'],
    session_id=self.temp_manager.session_id,
    data={"results": search_results},
    ttl_hours=24,
    expires_at=(datetime.now() + timedelta(hours=24)).isoformat(),
    source_agent="web_search_agent",
    source_operation="search"
)

self.temp_manager.save_data(
    category="cache",
    key="web_search_results",
    data=temp_data
)
```

---

## 7. Request Human Input

```python
from task_manager.models import HumanInputRequest, HumanInputResponse

# Create request
request = HumanInputRequest(
    request_id=str(uuid.uuid4()),
    request_type="review_required",
    task_id=task['id'],
    task_description=task['description'],
    current_state={
        "ocr_confidence": 0.65,
        "extracted_text": "Bangalo?e Urban"
    },
    prompt="OCR confidence is low. Please review extracted text.",
    options=["Approve", "Correct", "Retry OCR"],
    background="Image quality is poor, OCR may have errors.",
    recommendations={
        "action": "Correct",
        "reason": "Manual correction is faster"
    },
    timeout_seconds=300,
    default_action="Retry OCR"
)

# Publish request event
event = create_system_event(
    event_type="human_input_requested",
    event_category="human_interaction",
    source_agent="ocr_image_agent",
    payload=request,
    source_task_id=task['id']
)
publish_event(event)

# ... wait for response via event subscription ...
```

---

## 8. Migration Checklist

### Phase 1: Backward Compatible (Week 1-2)
- [ ] Import new message schemas in all agent files
- [ ] Create wrapper functions that convert legacy dict → TypedDict
- [ ] Add event bus initialization to agent.__init__
- [ ] Keep existing return types, add new schemas internally

### Phase 2: Dual Support (Week 3-4)
- [ ] Update `execute_task()` to return `AgentExecutionResponse`
- [ ] Add validation at module boundaries
- [ ] Publish events alongside existing state updates
- [ ] Update tests to check both formats

### Phase 3: Event Migration (Week 5-6)
- [ ] Replace state flags with event publications
- [ ] Subscribe workflow nodes to events
- [ ] Update conditional routing to use event queue
- [ ] Test event-driven workflows

### Phase 4: Full Standardization (Week 7-8)
- [ ] Remove legacy dict returns
- [ ] Enforce schema validation everywhere
- [ ] Update all temp/cache storage to use schemas
- [ ] Add mypy type checking to CI/CD

### Phase 5: Cleanup (Week 9-10)
- [ ] Remove backward compatibility wrappers
- [ ] Update all documentation
- [ ] Add schema versioning support
- [ ] Performance optimization

---

## 9. Testing

### Unit Test Example:
```python
def test_agent_response_format():
    """Verify agent returns standardized response."""
    agent = WebSearchAgent()
    task = Task(
        id="task_1",
        description="Search for Karnataka districts",
        status=TaskStatus.PENDING,
        parent_id=None,
        depth=0,
        context="",
        result=None,
        error=None,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat()
    )
    
    response = agent.execute_task(task, {})
    
    # Validate structure
    assert isinstance(response, dict)
    assert "status" in response
    assert "success" in response
    assert "result" in response
    assert "execution_time_ms" in response
    assert "agent_name" in response
    
    # Validate types
    assert isinstance(response["success"], bool)
    assert isinstance(response["execution_time_ms"], int)
    assert response["agent_name"] == "web_search_agent"
```

### Event Test Example:
```python
def test_event_publishing():
    """Verify events are published correctly."""
    from task_manager.core.event_bus import get_event_bus
    
    event_bus = get_event_bus()
    events_received = []
    
    def handler(event: SystemEvent):
        events_received.append(event)
    
    # Subscribe
    event_bus.subscribe("test_event", handler, subscriber_name="test")
    
    # Publish
    event = create_system_event(
        event_type="test_event",
        event_category="system_state",
        source_agent="test_agent",
        payload={"test": "data"}
    )
    event_bus.publish(event)
    
    # Verify
    assert len(events_received) == 1
    assert events_received[0]["event_type"] == "test_event"
    assert events_received[0]["payload"]["test"] == "data"
```

---

## 10. Common Patterns

### Pattern 1: Agent Chain Execution
```python
# Agent A completes, triggers Agent B
response_a = AgentExecutionResponse(
    # ... other fields ...
    next_agents=["excel_agent"],
    chain_data={"csv_file": "output.csv"},
    event_triggers=["web_findings_ready"]
)

# Workflow automatically routes to Agent B with chain_data
```

### Pattern 2: Conditional Event Routing
```python
# Subscribe with filter
event_bus.subscribe(
    event_type="task_completed",
    handler=handle_high_priority_tasks,
    subscriber_name="priority_handler",
    filter_func=lambda evt: evt['payload'].get('priority', 10) <= 3
)
```

### Pattern 3: Error Recovery
```python
def resilient_handler(event: SystemEvent):
    try:
        process_event(event)
    except Exception as e:
        # Create error and retry
        error = create_error_response(
            error_code="HANDLER_ERROR_001",
            error_type="execution_error",
            message=str(e),
            source="my_handler",
            recoverable=True,
            retry_after_seconds=30
        )
        publish_event(create_system_event(
            event_type="handler_failed",
            event_category="system_state",
            source_agent="my_handler",
            payload=error
        ))
```

---

## 11. Benefits Summary

✅ **Type Safety**: Catch errors at development time  
✅ **Consistency**: Same format across all agents  
✅ **Debugging**: Easy to trace message flows  
✅ **Testing**: Standardized assertions  
✅ **Scalability**: Easy to add new agents/events  
✅ **Documentation**: Self-documenting schemas  
✅ **Monitoring**: Complete audit trail  
✅ **Event-Driven**: Reactive, decoupled architecture  

---

**Questions? See INTERFACE_STANDARDS.md for full specification.**
