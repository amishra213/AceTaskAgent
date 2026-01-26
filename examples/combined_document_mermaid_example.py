"""
Combined Example: Document Agent + Mermaid Agent

This example demonstrates using both agents together to create:
1. A comprehensive project documentation in DOCX format
2. Process flow diagrams in Mermaid format
3. A complete technical specification document
"""

import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from task_manager.sub_agents.document_agent import DocumentAgent
from task_manager.sub_agents.mermaid_agent import MermaidAgent
from task_manager.utils.logger import get_logger

logger = get_logger(__name__)


def create_project_documentation():
    """Create a complete project documentation package."""
    logger.info("\n" + "=" * 80)
    logger.info("COMBINED EXAMPLE: Creating Project Documentation Package")
    logger.info("=" * 80)
    
    doc_agent = DocumentAgent()
    mermaid_agent = MermaidAgent()
    
    # 1. Create project overview document (TXT)
    logger.info("\n--- Step 1: Creating Project Overview (TXT) ---")
    overview_result = doc_agent.execute_task(
        operation="create_txt",
        parameters={
            "output_path": "output_folder/PROJECT_OVERVIEW.txt",
            "content": [
                "TaskManager - AI-Powered Task Automation System",
                "=" * 60,
                "",
                "Version: 2.0",
                f"Date: {datetime.now().strftime('%Y-%m-%d')}",
                "",
                "OVERVIEW",
                "-" * 60,
                "TaskManager is an advanced task automation system that uses",
                "specialized AI sub-agents to handle various file operations,",
                "data processing, and content generation tasks.",
                "",
                "KEY COMPONENTS",
                "-" * 60,
                "1. Master Planner - Orchestrates task execution",
                "2. Sub-Agents - Specialized workers for different tasks",
                "3. Event Bus - Event-driven communication",
                "4. Blackboard - Shared state management",
                "",
                "SUB-AGENTS",
                "-" * 60,
                "- PDF Agent: PDF processing and generation",
                "- Excel Agent: Spreadsheet operations",
                "- Web Search Agent: Information retrieval",
                "- OCR Agent: Text extraction from images",
                "- Code Interpreter: Python code execution",
                "- Data Extraction: Intelligent data parsing",
                "- Document Agent: TXT and DOCX file creation",
                "- Mermaid Agent: Diagram and visualization generation",
                "",
                "For more details, see the technical specification document.",
                ""
            ]
        }
    )
    logger.info(f"Overview created: {overview_result.get('success')}")
    
    # 2. Create workflow diagram
    logger.info("\n--- Step 2: Creating Task Processing Workflow Diagram ---")
    workflow_result = mermaid_agent.execute_task(
        operation="create_flowchart",
        parameters={
            "output_path": "output_folder/WORKFLOW_DIAGRAM.md",
            "title": "TaskManager Workflow",
            "direction": "TD",
            "nodes": [
                {"id": "START", "text": "User Request", "shape": "rounded"},
                {"id": "PLANNER", "text": "Master Planner", "shape": "rectangle"},
                {"id": "ANALYZE", "text": "Analyze Request", "shape": "rectangle"},
                {"id": "COMPLEX", "text": "Complex Task?", "shape": "rhombus"},
                {"id": "BREAKDOWN", "text": "Break Down Task", "shape": "rectangle"},
                {"id": "ROUTE", "text": "Route to Sub-Agent", "shape": "hexagon"},
                {"id": "EXECUTE", "text": "Execute Task", "shape": "rectangle"},
                {"id": "AGGREGATE", "text": "Aggregate Results", "shape": "rectangle"},
                {"id": "COMPLETE", "text": "Return Results", "shape": "rounded"}
            ],
            "connections": [
                {"from": "START", "to": "PLANNER"},
                {"from": "PLANNER", "to": "ANALYZE"},
                {"from": "ANALYZE", "to": "COMPLEX"},
                {"from": "COMPLEX", "to": "BREAKDOWN", "text": "Yes"},
                {"from": "COMPLEX", "to": "ROUTE", "text": "No"},
                {"from": "BREAKDOWN", "to": "ROUTE"},
                {"from": "ROUTE", "to": "EXECUTE"},
                {"from": "EXECUTE", "to": "AGGREGATE"},
                {"from": "AGGREGATE", "to": "COMPLETE"}
            ]
        }
    )
    logger.info(f"Workflow diagram created: {workflow_result.get('success')}")
    
    # 3. Create agent architecture diagram
    logger.info("\n--- Step 3: Creating Agent Architecture Diagram ---")
    architecture_result = mermaid_agent.execute_task(
        operation="create_flowchart",
        parameters={
            "output_path": "output_folder/ARCHITECTURE_DIAGRAM.md",
            "title": "Sub-Agent Architecture",
            "direction": "LR",
            "nodes": [
                {"id": "MP", "text": "Master Planner", "shape": "hexagon"},
                {"id": "PDF", "text": "PDF Agent", "shape": "rectangle"},
                {"id": "EXCEL", "text": "Excel Agent", "shape": "rectangle"},
                {"id": "WEB", "text": "Web Search Agent", "shape": "rectangle"},
                {"id": "OCR", "text": "OCR Agent", "shape": "rectangle"},
                {"id": "CODE", "text": "Code Interpreter", "shape": "rectangle"},
                {"id": "DATA", "text": "Data Extraction", "shape": "rectangle"},
                {"id": "DOC", "text": "Document Agent", "shape": "rectangle"},
                {"id": "MERMAID", "text": "Mermaid Agent", "shape": "rectangle"},
                {"id": "BB", "text": "Blackboard", "shape": "cylinder"},
                {"id": "EB", "text": "Event Bus", "shape": "cylinder"}
            ],
            "connections": [
                {"from": "MP", "to": "PDF", "type": "thick"},
                {"from": "MP", "to": "EXCEL", "type": "thick"},
                {"from": "MP", "to": "WEB", "type": "thick"},
                {"from": "MP", "to": "OCR", "type": "thick"},
                {"from": "MP", "to": "CODE", "type": "thick"},
                {"from": "MP", "to": "DATA", "type": "thick"},
                {"from": "MP", "to": "DOC", "type": "thick"},
                {"from": "MP", "to": "MERMAID", "type": "thick"},
                {"from": "PDF", "to": "BB", "type": "dotted"},
                {"from": "EXCEL", "to": "BB", "type": "dotted"},
                {"from": "DOC", "to": "BB", "type": "dotted"},
                {"from": "PDF", "to": "EB", "type": "dotted"},
                {"from": "EXCEL", "to": "EB", "type": "dotted"},
                {"from": "DOC", "to": "EB", "type": "dotted"}
            ]
        }
    )
    logger.info(f"Architecture diagram created: {architecture_result.get('success')}")
    
    # 4. Create agent communication sequence diagram
    logger.info("\n--- Step 4: Creating Agent Communication Sequence ---")
    sequence_result = mermaid_agent.execute_task(
        operation="create_sequence",
        parameters={
            "output_path": "output_folder/AGENT_SEQUENCE.md",
            "title": "Multi-Agent Task Execution Sequence",
            "participants": ["User", "MasterPlanner", "DocumentAgent", "MermaidAgent", "Blackboard"],
            "messages": [
                {"from": "User", "to": "MasterPlanner", "text": "Create project documentation", "type": "sync"},
                {"from": "MasterPlanner", "to": "DocumentAgent", "text": "Create overview document", "type": "async"},
                {"from": "DocumentAgent", "to": "Blackboard", "text": "Store document path", "type": "async"},
                {"from": "DocumentAgent", "to": "MasterPlanner", "text": "Document created", "type": "return"},
                {"from": "MasterPlanner", "to": "MermaidAgent", "text": "Create workflow diagram", "type": "async"},
                {"from": "MermaidAgent", "to": "Blackboard", "text": "Store diagram path", "type": "async"},
                {"from": "MermaidAgent", "to": "MasterPlanner", "text": "Diagram created", "type": "return"},
                {"from": "MasterPlanner", "to": "DocumentAgent", "text": "Create final report", "type": "async"},
                {"type": "note", "to": "DocumentAgent", "text": "Combining all artifacts", "position": "right of"},
                {"from": "DocumentAgent", "to": "MasterPlanner", "text": "Report complete", "type": "return"},
                {"from": "MasterPlanner", "to": "User", "text": "All documentation ready", "type": "sync"}
            ]
        }
    )
    logger.info(f"Sequence diagram created: {sequence_result.get('success')}")
    
    # 5. Create comprehensive technical specification (DOCX)
    logger.info("\n--- Step 5: Creating Technical Specification Document (DOCX) ---")
    spec_result = doc_agent.execute_task(
        operation="create_docx",
        parameters={
            "output_path": "output_folder/TECHNICAL_SPECIFICATION.docx",
            "title": "TaskManager Technical Specification",
            "content": [
                {"type": "heading", "text": "Document Information", "level": 1},
                {"type": "paragraph", "text": f"Version: 2.0"},
                {"type": "paragraph", "text": f"Date: {datetime.now().strftime('%Y-%m-%d')}"},
                {"type": "paragraph", "text": "Author: TaskManager Development Team"},
                
                {"type": "heading", "text": "1. System Overview", "level": 1},
                {"type": "paragraph", "text": "TaskManager is an AI-powered task automation system featuring a Master Planner and specialized sub-agents for handling various file operations, data processing, and content generation tasks."},
                
                {"type": "heading", "text": "2. Architecture", "level": 1},
                {"type": "paragraph", "text": "The system follows a hierarchical architecture with the following components:"},
                
                {"type": "heading", "text": "2.1 Master Planner", "level": 2},
                {"type": "paragraph", "text": "The Master Planner is responsible for:"},
                {"type": "list", "items": [
                    "Receiving and analyzing user requests",
                    "Breaking down complex tasks into subtasks",
                    "Routing tasks to appropriate sub-agents",
                    "Aggregating and returning results"
                ]},
                
                {"type": "heading", "text": "2.2 Sub-Agents", "level": 2},
                {"type": "paragraph", "text": "The following specialized sub-agents are available:"},
                
                {"type": "heading", "text": "PDF Agent", "level": 3},
                {"type": "list", "items": [
                    "Read PDF files and extract text",
                    "Create new PDF documents",
                    "Merge multiple PDFs",
                    "Extract specific pages"
                ]},
                
                {"type": "heading", "text": "Excel Agent", "level": 3},
                {"type": "list", "items": [
                    "Read Excel files and extract data",
                    "Create formatted spreadsheets",
                    "Perform data aggregation",
                    "Generate charts and visualizations"
                ]},
                
                {"type": "heading", "text": "Document Agent", "level": 3},
                {"type": "list", "items": [
                    "Create TXT files with formatted content",
                    "Create DOCX files with headings and lists",
                    "Read and parse text documents",
                    "Append content to existing files"
                ]},
                
                {"type": "heading", "text": "Mermaid Agent", "level": 3},
                {"type": "list", "items": [
                    "Generate flowcharts for process flows",
                    "Create sequence diagrams",
                    "Build state diagrams",
                    "Produce Gantt charts for project timelines"
                ]},
                
                {"type": "heading", "text": "3. Event-Driven Architecture", "level": 1},
                {"type": "paragraph", "text": "The system uses an event bus for asynchronous communication between components. Key event types include:"},
                {"type": "list", "items": [
                    "Task lifecycle events (started, completed, failed)",
                    "Document processing events",
                    "Diagram generation events",
                    "System state changes"
                ]},
                
                {"type": "heading", "text": "4. Data Flow", "level": 1},
                {"type": "paragraph", "text": "Data flows through the system as follows:"},
                {"type": "list", "items": [
                    "User submits request to Master Planner",
                    "Master Planner analyzes and routes to sub-agents",
                    "Sub-agents execute tasks and store results in Blackboard",
                    "Results are aggregated and returned to user",
                    "Events are published for monitoring and coordination"
                ]},
                
                {"type": "heading", "text": "5. Supported Operations", "level": 1},
                
                {"type": "heading", "text": "5.1 Document Operations", "level": 2},
                {"type": "list", "items": [
                    "create_txt - Create text files",
                    "create_docx - Create Word documents",
                    "read_txt - Read text files",
                    "read_docx - Read Word documents",
                    "append_txt - Append to text files",
                    "append_docx - Append to Word documents"
                ]},
                
                {"type": "heading", "text": "5.2 Diagram Operations", "level": 2},
                {"type": "list", "items": [
                    "create_flowchart - Generate process flowcharts",
                    "create_sequence - Generate sequence diagrams",
                    "create_state_diagram - Generate state diagrams",
                    "create_gantt - Generate Gantt charts",
                    "create_custom - Custom Mermaid diagrams"
                ]},
                
                {"type": "heading", "text": "6. Output Formats", "level": 1},
                {"type": "paragraph", "text": "The system supports multiple output formats:"},
                {"type": "list", "items": [
                    "TXT - Plain text files",
                    "DOCX - Microsoft Word documents",
                    "PDF - Portable Document Format",
                    "XLSX - Excel spreadsheets",
                    "MD - Markdown with Mermaid diagrams",
                    "MMD - Pure Mermaid diagram files"
                ]},
                
                {"type": "heading", "text": "7. Future Enhancements", "level": 1},
                {"type": "list", "items": [
                    "HTML report generation",
                    "PowerPoint presentation creation",
                    "Database integration",
                    "Real-time collaboration features",
                    "Advanced AI-powered content generation"
                ]},
                
                {"type": "heading", "text": "8. Conclusion", "level": 1},
                {"type": "paragraph", "text": "TaskManager provides a comprehensive solution for automating document creation, data processing, and visualization tasks through its modular sub-agent architecture."}
            ]
        }
    )
    logger.info(f"Technical specification created: {spec_result.get('success')}")
    
    # 6. Create project timeline Gantt chart
    logger.info("\n--- Step 6: Creating Project Timeline ---")
    timeline_result = mermaid_agent.execute_task(
        operation="create_gantt",
        parameters={
            "output_path": "output_folder/PROJECT_TIMELINE.md",
            "title": "TaskManager Development Timeline",
            "sections": [
                {
                    "name": "Foundation",
                    "tasks": [
                        {"name": "Core Framework", "status": "done", "start": "2026-01-01", "duration": "14d"},
                        {"name": "Event System", "status": "done", "start": "2026-01-08", "duration": "7d"}
                    ]
                },
                {
                    "name": "File Processing Agents",
                    "tasks": [
                        {"name": "PDF Agent", "status": "done", "start": "2026-01-15", "duration": "5d"},
                        {"name": "Excel Agent", "status": "done", "start": "2026-01-20", "duration": "5d"}
                    ]
                },
                {
                    "name": "Content Creation Agents",
                    "tasks": [
                        {"name": "Document Agent", "status": "done", "start": "2026-01-26", "duration": "2d"},
                        {"name": "Mermaid Agent", "status": "active", "start": "2026-01-26", "duration": "2d"}
                    ]
                },
                {
                    "name": "Testing & Documentation",
                    "tasks": [
                        {"name": "Unit Tests", "status": "active", "start": "2026-01-28", "duration": "5d"},
                        {"name": "Integration Tests", "status": "", "start": "2026-02-02", "duration": "7d"},
                        {"name": "Documentation", "status": "", "start": "2026-02-09", "duration": "5d"}
                    ]
                }
            ]
        }
    )
    logger.info(f"Timeline created: {timeline_result.get('success')}")
    
    # 7. Create a summary index file
    logger.info("\n--- Step 7: Creating Documentation Index (TXT) ---")
    index_result = doc_agent.execute_task(
        operation="create_txt",
        parameters={
            "output_path": "output_folder/DOCUMENTATION_INDEX.txt",
            "content": [
                "TaskManager Documentation Package",
                "=" * 60,
                "",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "CONTENTS",
                "-" * 60,
                "",
                "1. PROJECT_OVERVIEW.txt",
                "   Quick overview of the TaskManager system",
                "",
                "2. TECHNICAL_SPECIFICATION.docx",
                "   Comprehensive technical specification document",
                "   (Microsoft Word format)",
                "",
                "3. WORKFLOW_DIAGRAM.md",
                "   Task processing workflow flowchart",
                "   (Mermaid diagram in Markdown)",
                "",
                "4. ARCHITECTURE_DIAGRAM.md",
                "   Sub-agent architecture diagram",
                "   (Mermaid diagram in Markdown)",
                "",
                "5. AGENT_SEQUENCE.md",
                "   Agent communication sequence diagram",
                "   (Mermaid diagram in Markdown)",
                "",
                "6. PROJECT_TIMELINE.md",
                "   Development timeline Gantt chart",
                "   (Mermaid diagram in Markdown)",
                "",
                "VIEWING DIAGRAMS",
                "-" * 60,
                "Mermaid diagrams (.md files) can be viewed in:",
                "- Visual Studio Code (with Mermaid extension)",
                "- GitHub (automatic rendering)",
                "- https://mermaid.live (paste the code)",
                "",
                "NOTES",
                "-" * 60,
                "- All files are saved in the output_folder directory",
                "- DOCX files require Microsoft Word or compatible software",
                "- Diagrams are in standard Mermaid syntax",
                "",
                "For questions or issues, contact the development team.",
                ""
            ]
        }
    )
    logger.info(f"Index created: {index_result.get('success')}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("DOCUMENTATION PACKAGE CREATION COMPLETE")
    logger.info("=" * 80)
    
    results = {
        "overview": overview_result.get('success'),
        "workflow_diagram": workflow_result.get('success'),
        "architecture_diagram": architecture_result.get('success'),
        "sequence_diagram": sequence_result.get('success'),
        "technical_spec": spec_result.get('success'),
        "timeline": timeline_result.get('success'),
        "index": index_result.get('success')
    }
    
    logger.info("\nResults Summary:")
    for doc_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"  {doc_name}: {status}")
    
    all_success = all(results.values())
    logger.info(f"\nOverall Status: {'✓ ALL DOCUMENTS CREATED' if all_success else '✗ SOME FAILURES'}")
    
    return results


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("COMBINED AGENTS EXAMPLE")
    print("Document Agent + Mermaid Agent")
    print("=" * 80)
    
    try:
        results = create_project_documentation()
        
        print("\n" + "=" * 80)
        print("Documentation package created in output_folder:")
        print("=" * 80)
        print("\nText Documents:")
        print("  - PROJECT_OVERVIEW.txt")
        print("  - DOCUMENTATION_INDEX.txt")
        print("\nWord Documents:")
        print("  - TECHNICAL_SPECIFICATION.docx")
        print("\nMermaid Diagrams:")
        print("  - WORKFLOW_DIAGRAM.md")
        print("  - ARCHITECTURE_DIAGRAM.md")
        print("  - AGENT_SEQUENCE.md")
        print("  - PROJECT_TIMELINE.md")
        print("\nThis example demonstrates:")
        print("  ✓ Creating TXT files with formatted content")
        print("  ✓ Creating DOCX files with headings and lists")
        print("  ✓ Generating flowcharts and sequence diagrams")
        print("  ✓ Building Gantt charts for timelines")
        print("  ✓ Coordinating multiple agents for complex tasks")
        
    except Exception as e:
        logger.error(f"Example failed with error: {e}", exc_info=True)
