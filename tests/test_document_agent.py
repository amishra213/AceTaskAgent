"""
Test script for Document Agent - TXT and DOCX file creation.

This script demonstrates:
1. Creating TXT files with simple text content
2. Creating DOCX files with formatted content (headings, paragraphs, lists)
3. Reading TXT and DOCX files
4. Appending content to existing files
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from task_manager.sub_agents.document_agent import DocumentAgent
from task_manager.utils.logger import get_logger

logger = get_logger(__name__)


def test_create_txt_file():
    """Test creating a simple TXT file."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Creating TXT File")
    logger.info("=" * 80)
    
    agent = DocumentAgent()
    
    result = agent.execute_task(
        operation="create_txt",
        parameters={
            "output_path": "output_folder/sample_document.txt",
            "content": [
                "Sample Text Document",
                "=" * 50,
                "",
                "This is a test document created by the Document Agent.",
                "It demonstrates the ability to create TXT files with multiple lines.",
                "",
                "Key Features:",
                "- Simple text content",
                "- Multiple lines",
                "- UTF-8 encoding",
                "",
                "End of document."
            ],
            "encoding": "utf-8"
        }
    )
    
    logger.info(f"Result: {result}")
    return result


def test_create_docx_file():
    """Test creating a DOCX file with formatted content."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Creating DOCX File with Formatting")
    logger.info("=" * 80)
    
    agent = DocumentAgent()
    
    result = agent.execute_task(
        operation="create_docx",
        parameters={
            "output_path": "output_folder/sample_document.docx",
            "title": "Sample Document",
            "content": [
                {"type": "heading", "text": "Introduction", "level": 1},
                {"type": "paragraph", "text": "This is a sample document created by the Document Agent."},
                {"type": "paragraph", "text": "It demonstrates various formatting capabilities."},
                
                {"type": "heading", "text": "Features", "level": 2},
                {"type": "list", "items": [
                    "Create headings at different levels",
                    "Add formatted paragraphs",
                    "Create bulleted lists",
                    "Support for bold and italic text"
                ]},
                
                {"type": "heading", "text": "Process Flow", "level": 2},
                {"type": "paragraph", "text": "The document creation process follows these steps:", "bold": True},
                {"type": "list", "items": [
                    "Define document structure",
                    "Add content blocks",
                    "Apply formatting",
                    "Save to file"
                ]},
                
                {"type": "heading", "text": "Conclusion", "level": 2},
                {"type": "paragraph", "text": "This document demonstrates the Document Agent's capabilities."}
            ]
        }
    )
    
    logger.info(f"Result: {result}")
    return result


def test_read_txt_file():
    """Test reading a TXT file."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Reading TXT File")
    logger.info("=" * 80)
    
    agent = DocumentAgent()
    
    result = agent.execute_task(
        operation="read_txt",
        parameters={
            "file_path": "output_folder/sample_document.txt",
            "encoding": "utf-8"
        }
    )
    
    if result.get("success"):
        logger.info(f"File read successfully!")
        logger.info(f"Line count: {result.get('line_count')}")
        logger.info(f"Character count: {result.get('char_count')}")
        logger.info(f"\nFirst 5 lines:")
        for i, line in enumerate(result.get('lines', [])[:5]):
            logger.info(f"  {i+1}: {line}")
    else:
        logger.error(f"Error reading file: {result.get('error')}")
    
    return result


def test_append_txt_file():
    """Test appending content to a TXT file."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Appending to TXT File")
    logger.info("=" * 80)
    
    agent = DocumentAgent()
    
    result = agent.execute_task(
        operation="append_txt",
        parameters={
            "file_path": "output_folder/sample_document.txt",
            "content": [
                "",
                "--- APPENDED CONTENT ---",
                "This content was added later.",
                f"Timestamp: {agent.agent_name}"
            ],
            "encoding": "utf-8"
        }
    )
    
    logger.info(f"Result: {result}")
    return result


def test_project_report():
    """Create a realistic project report."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: Creating Project Report")
    logger.info("=" * 80)
    
    agent = DocumentAgent()
    
    result = agent.execute_task(
        operation="create_docx",
        parameters={
            "output_path": "output_folder/project_report.docx",
            "title": "Q1 2026 Project Status Report",
            "content": [
                {"type": "heading", "text": "Executive Summary", "level": 1},
                {"type": "paragraph", "text": "This report provides an overview of the TaskManager project status for Q1 2026."},
                
                {"type": "heading", "text": "Project Overview", "level": 1},
                {"type": "paragraph", "text": "TaskManager is an AI-powered task automation system with multiple specialized sub-agents."},
                
                {"type": "heading", "text": "Recent Developments", "level": 2},
                {"type": "list", "items": [
                    "Added Document Agent for TXT and DOCX file creation",
                    "Implemented Mermaid Agent for diagram generation",
                    "Enhanced error handling and logging",
                    "Improved event-driven architecture"
                ]},
                
                {"type": "heading", "text": "Active Sub-Agents", "level": 2},
                {"type": "list", "items": [
                    "PDF Agent - Document processing",
                    "Excel Agent - Spreadsheet operations",
                    "Web Search Agent - Information retrieval",
                    "Code Interpreter Agent - Code execution",
                    "Document Agent - Text file creation",
                    "Mermaid Agent - Diagram generation"
                ]},
                
                {"type": "heading", "text": "Next Steps", "level": 1},
                {"type": "paragraph", "text": "Upcoming priorities for Q2 2026:"},
                {"type": "list", "items": [
                    "Integration testing of new agents",
                    "Performance optimization",
                    "Documentation updates",
                    "User feedback collection"
                ]},
                
                {"type": "heading", "text": "Conclusion", "level": 1},
                {"type": "paragraph", "text": "The project is progressing well with successful implementation of new capabilities."}
            ]
        }
    )
    
    logger.info(f"Result: {result}")
    return result


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DOCUMENT AGENT TEST SUITE")
    print("=" * 80)
    
    try:
        # Run all tests
        test_create_txt_file()
        test_create_docx_file()
        test_read_txt_file()
        test_append_txt_file()
        test_project_report()
        
        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED")
        print("=" * 80)
        print("\nCheck the output_folder for generated files:")
        print("- sample_document.txt")
        print("- sample_document.docx")
        print("- project_report.docx")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
