"""
Test script for Mermaid Agent - Diagram generation.

This script demonstrates:
1. Creating flowcharts for process flows
2. Creating sequence diagrams
3. Creating state diagrams
4. Creating Gantt charts
5. Creating custom diagrams
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from task_manager.sub_agents.mermaid_agent import MermaidAgent
from task_manager.utils.logger import get_logger

logger = get_logger(__name__)


def test_create_flowchart():
    """Test creating a process flowchart."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Creating Process Flowchart")
    logger.info("=" * 80)
    
    agent = MermaidAgent()
    
    result = agent.execute_task(
        operation="create_flowchart",
        parameters={
            "output_path": "output_folder/process_flow.md",
            "title": "Task Processing Workflow",
            "direction": "TD",
            "nodes": [
                {"id": "A", "text": "Start", "shape": "rounded"},
                {"id": "B", "text": "Receive Task", "shape": "rectangle"},
                {"id": "C", "text": "Analyze Task", "shape": "rectangle"},
                {"id": "D", "text": "Is Complex?", "shape": "rhombus"},
                {"id": "E", "text": "Break Down Task", "shape": "rectangle"},
                {"id": "F", "text": "Assign to Agent", "shape": "rectangle"},
                {"id": "G", "text": "Execute Task", "shape": "rectangle"},
                {"id": "H", "text": "Success?", "shape": "rhombus"},
                {"id": "I", "text": "Return Result", "shape": "rectangle"},
                {"id": "J", "text": "Handle Error", "shape": "rectangle"},
                {"id": "K", "text": "End", "shape": "rounded"}
            ],
            "connections": [
                {"from": "A", "to": "B"},
                {"from": "B", "to": "C"},
                {"from": "C", "to": "D"},
                {"from": "D", "to": "E", "text": "Yes"},
                {"from": "D", "to": "F", "text": "No"},
                {"from": "E", "to": "F"},
                {"from": "F", "to": "G"},
                {"from": "G", "to": "H"},
                {"from": "H", "to": "I", "text": "Yes"},
                {"from": "H", "to": "J", "text": "No"},
                {"from": "J", "to": "G", "type": "dotted"},
                {"from": "I", "to": "K"}
            ]
        }
    )
    
    logger.info(f"Result: {result}")
    return result


def test_create_sequence_diagram():
    """Test creating a sequence diagram."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Creating Sequence Diagram")
    logger.info("=" * 80)
    
    agent = MermaidAgent()
    
    result = agent.execute_task(
        operation="create_sequence",
        parameters={
            "output_path": "output_folder/agent_communication.md",
            "title": "Sub-Agent Communication Flow",
            "participants": ["User", "MasterPlanner", "WebSearchAgent", "DataExtractionAgent", "ExcelAgent"],
            "messages": [
                {"from": "User", "to": "MasterPlanner", "text": "Analyze market data", "type": "sync"},
                {"from": "MasterPlanner", "to": "WebSearchAgent", "text": "Search for market reports", "type": "async"},
                {"from": "WebSearchAgent", "to": "MasterPlanner", "text": "Return search results", "type": "return"},
                {"from": "MasterPlanner", "to": "DataExtractionAgent", "text": "Extract key metrics", "type": "async"},
                {"type": "note", "to": "DataExtractionAgent", "text": "Processing data...", "position": "right of"},
                {"from": "DataExtractionAgent", "to": "MasterPlanner", "text": "Return extracted data", "type": "return"},
                {"from": "MasterPlanner", "to": "ExcelAgent", "text": "Create report", "type": "async"},
                {"from": "ExcelAgent", "to": "MasterPlanner", "text": "Report created", "type": "return"},
                {"from": "MasterPlanner", "to": "User", "text": "Analysis complete", "type": "sync"}
            ]
        }
    )
    
    logger.info(f"Result: {result}")
    return result


def test_create_state_diagram():
    """Test creating a state diagram."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Creating State Diagram")
    logger.info("=" * 80)
    
    agent = MermaidAgent()
    
    result = agent.execute_task(
        operation="create_state_diagram",
        parameters={
            "output_path": "output_folder/task_states.md",
            "title": "Task Lifecycle States",
            "states": [
                {"id": "Pending", "description": "Waiting for processing"},
                {"id": "Analyzing", "description": "Analyzing requirements"},
                {"id": "BrokenDown", "description": "Decomposed into subtasks"},
                {"id": "Executing", "description": "Being executed"},
                {"id": "Completed", "description": "Successfully completed"},
                {"id": "Failed", "description": "Execution failed"},
                {"id": "NeedsReview", "description": "Requires human review"}
            ],
            "transitions": [
                {"from": "[*]", "to": "Pending"},
                {"from": "Pending", "to": "Analyzing", "event": "start_analysis"},
                {"from": "Analyzing", "to": "BrokenDown", "event": "complex_task"},
                {"from": "Analyzing", "to": "Executing", "event": "simple_task"},
                {"from": "BrokenDown", "to": "Executing", "event": "subtasks_ready"},
                {"from": "Executing", "to": "Completed", "event": "success"},
                {"from": "Executing", "to": "Failed", "event": "error"},
                {"from": "Executing", "to": "NeedsReview", "event": "ambiguous_result"},
                {"from": "Failed", "to": "Analyzing", "event": "retry"},
                {"from": "NeedsReview", "to": "Executing", "event": "reviewed"},
                {"from": "Completed", "to": "[*]"}
            ]
        }
    )
    
    logger.info(f"Result: {result}")
    return result


def test_create_gantt_chart():
    """Test creating a Gantt chart."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Creating Gantt Chart")
    logger.info("=" * 80)
    
    agent = MermaidAgent()
    
    result = agent.execute_task(
        operation="create_gantt",
        parameters={
            "output_path": "output_folder/project_timeline.md",
            "title": "TaskManager Development Timeline - Q1 2026",
            "sections": [
                {
                    "name": "Planning Phase",
                    "tasks": [
                        {"name": "Requirements Analysis", "status": "done", "start": "2026-01-01", "duration": "7d"},
                        {"name": "Architecture Design", "status": "done", "start": "2026-01-08", "duration": "5d"}
                    ]
                },
                {
                    "name": "Development Phase",
                    "tasks": [
                        {"name": "Core Framework", "status": "done", "start": "2026-01-13", "duration": "10d"},
                        {"name": "Sub-Agent Implementation", "status": "active", "start": "2026-01-23", "duration": "14d"},
                        {"name": "Document Agent", "status": "done", "start": "2026-01-26", "duration": "2d"},
                        {"name": "Mermaid Agent", "status": "active", "start": "2026-01-26", "duration": "2d"}
                    ]
                },
                {
                    "name": "Testing Phase",
                    "tasks": [
                        {"name": "Unit Tests", "status": "active", "start": "2026-01-28", "duration": "5d"},
                        {"name": "Integration Tests", "status": "", "start": "2026-02-03", "duration": "7d"},
                        {"name": "Performance Testing", "status": "", "start": "2026-02-10", "duration": "5d"}
                    ]
                },
                {
                    "name": "Deployment",
                    "tasks": [
                        {"name": "Documentation", "status": "", "start": "2026-02-15", "duration": "5d"},
                        {"name": "Release Preparation", "status": "", "start": "2026-02-20", "duration": "3d"}
                    ]
                }
            ]
        }
    )
    
    logger.info(f"Result: {result}")
    return result


def test_create_data_processing_flow():
    """Create a detailed data processing flowchart."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: Creating Data Processing Flow")
    logger.info("=" * 80)
    
    agent = MermaidAgent()
    
    result = agent.execute_task(
        operation="create_flowchart",
        parameters={
            "output_path": "output_folder/data_processing_flow.md",
            "title": "Data Extraction and Processing Workflow",
            "direction": "LR",
            "nodes": [
                {"id": "START", "text": "Input Data", "shape": "cylinder"},
                {"id": "DETECT", "text": "Detect File Type", "shape": "hexagon"},
                {"id": "PDF", "text": "PDF Agent", "shape": "subroutine"},
                {"id": "EXCEL", "text": "Excel Agent", "shape": "subroutine"},
                {"id": "IMG", "text": "OCR Agent", "shape": "subroutine"},
                {"id": "EXTRACT", "text": "Data Extraction Agent", "shape": "rectangle"},
                {"id": "TRANSFORM", "text": "Transform Data", "shape": "rectangle"},
                {"id": "VALIDATE", "text": "Validate?", "shape": "rhombus"},
                {"id": "SAVE", "text": "Save Results", "shape": "rectangle"},
                {"id": "END", "text": "Output", "shape": "cylinder"}
            ],
            "connections": [
                {"from": "START", "to": "DETECT"},
                {"from": "DETECT", "to": "PDF", "text": "PDF"},
                {"from": "DETECT", "to": "EXCEL", "text": "Excel"},
                {"from": "DETECT", "to": "IMG", "text": "Image"},
                {"from": "PDF", "to": "EXTRACT"},
                {"from": "EXCEL", "to": "EXTRACT"},
                {"from": "IMG", "to": "EXTRACT"},
                {"from": "EXTRACT", "to": "TRANSFORM"},
                {"from": "TRANSFORM", "to": "VALIDATE"},
                {"from": "VALIDATE", "to": "SAVE", "text": "Valid"},
                {"from": "VALIDATE", "to": "TRANSFORM", "text": "Invalid", "type": "dotted"},
                {"from": "SAVE", "to": "END"}
            ]
        }
    )
    
    logger.info(f"Result: {result}")
    return result


def test_create_custom_diagram():
    """Test creating a custom Mermaid diagram."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 6: Creating Custom Diagram")
    logger.info("=" * 80)
    
    agent = MermaidAgent()
    
    custom_mermaid = """graph TB
    subgraph "Input Layer"
        A[User Request]
        B[File Upload]
    end
    
    subgraph "Processing Layer"
        C[Master Planner]
        D[Task Queue]
        E[Sub-Agents]
    end
    
    subgraph "Output Layer"
        F[Results]
        G[Artifacts]
        H[Events]
    end
    
    A --> C
    B --> C
    C --> D
    D --> E
    E --> F
    E --> G
    E --> H
    
    style C fill:#f9f,stroke:#333,stroke-width:4px
    style E fill:#bbf,stroke:#333,stroke-width:2px
"""
    
    result = agent.execute_task(
        operation="create_custom",
        parameters={
            "output_path": "output_folder/system_architecture.md",
            "title": "System Architecture Overview",
            "mermaid_code": custom_mermaid
        }
    )
    
    logger.info(f"Result: {result}")
    return result


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MERMAID AGENT TEST SUITE")
    print("=" * 80)
    
    try:
        # Run all tests
        test_create_flowchart()
        test_create_sequence_diagram()
        test_create_state_diagram()
        test_create_gantt_chart()
        test_create_data_processing_flow()
        test_create_custom_diagram()
        
        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED")
        print("=" * 80)
        print("\nCheck the output_folder for generated diagrams:")
        print("- process_flow.md")
        print("- agent_communication.md")
        print("- task_states.md")
        print("- project_timeline.md")
        print("- data_processing_flow.md")
        print("- system_architecture.md")
        print("\nYou can view these diagrams in:")
        print("- VS Code with Markdown Preview Mermaid Support extension")
        print("- GitHub (renders Mermaid automatically)")
        print("- https://mermaid.live (paste the code)")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
