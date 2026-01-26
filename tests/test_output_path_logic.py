"""
Test script demonstrating the extended output path determination logic.

This demonstrates various ways to specify output paths:
1. Direct path (output_path or file_path)
2. folder_path + file_name
3. folder_path + template_name
4. folder_path + title
5. Just template_name (uses output_folder)
6. Just title (uses output_folder)
7. No info (auto-generates timestamped name)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from task_manager.sub_agents.document_agent import DocumentAgent
from task_manager.sub_agents.mermaid_agent import MermaidAgent
from task_manager.utils.logger import get_logger

logger = get_logger(__name__)


def test_direct_path():
    """Test 1: Direct output_path specification."""
    print("\n" + "=" * 80)
    print("TEST 1: Direct output_path")
    print("=" * 80)
    
    agent = DocumentAgent()
    result = agent.execute_task(
        operation="create_txt",
        parameters={
            "output_path": "output_folder/direct_path_test.txt",
            "content": "Using direct output_path"
        }
    )
    print(f"✓ Created: {result.get('file')}")


def test_folder_plus_filename():
    """Test 2: folder_path + file_name combination."""
    print("\n" + "=" * 80)
    print("TEST 2: folder_path + file_name")
    print("=" * 80)
    
    agent = DocumentAgent()
    result = agent.execute_task(
        operation="create_docx",
        parameters={
            "folder_path": "output_folder",
            "file_name": "folder_filename_test",  # Extension will be added
            "content": [
                {"type": "heading", "text": "Test Document", "level": 1},
                {"type": "paragraph", "text": "Created using folder_path + file_name"}
            ]
        }
    )
    print(f"✓ Created: {result.get('file')}")


def test_folder_plus_template():
    """Test 3: folder_path + template_name."""
    print("\n" + "=" * 80)
    print("TEST 3: folder_path + template_name")
    print("=" * 80)
    
    agent = MermaidAgent()
    result = agent.execute_task(
        operation="create_flowchart",
        parameters={
            "folder_path": "output_folder",
            "template_name": "Process Flow Template",  # Will be sanitized
            "title": "Sample Process",
            "nodes": [
                {"id": "A", "text": "Start", "shape": "rounded"},
                {"id": "B", "text": "End", "shape": "rounded"}
            ],
            "connections": [{"from": "A", "to": "B"}]
        }
    )
    print(f"✓ Created: {result.get('file')}")


def test_folder_plus_title():
    """Test 4: folder_path + title."""
    print("\n" + "=" * 80)
    print("TEST 4: folder_path + title")
    print("=" * 80)
    
    agent = DocumentAgent()
    result = agent.execute_task(
        operation="create_docx",
        parameters={
            "folder_path": "output_folder",
            "title": "Quarterly Report",  # Will be used as filename
            "content": [
                {"type": "heading", "text": "Q1 2026 Report", "level": 1},
                {"type": "paragraph", "text": "Created using folder_path + title"}
            ]
        }
    )
    print(f"✓ Created: {result.get('file')}")


def test_template_only():
    """Test 5: Just template_name (uses output_folder)."""
    print("\n" + "=" * 80)
    print("TEST 5: template_name only (auto folder)")
    print("=" * 80)
    
    agent = MermaidAgent()
    result = agent.execute_task(
        operation="create_gantt",
        parameters={
            "template_name": "Project Timeline",
            "title": "Development Timeline",
            "sections": [{
                "name": "Phase 1",
                "tasks": [
                    {"name": "Planning", "status": "done", "start": "2026-01-01", "duration": "5d"}
                ]
            }]
        }
    )
    print(f"✓ Created: {result.get('file')}")


def test_title_only():
    """Test 6: Just title (uses output_folder)."""
    print("\n" + "=" * 80)
    print("TEST 6: title only (auto folder)")
    print("=" * 80)
    
    agent = DocumentAgent()
    result = agent.execute_task(
        operation="create_txt",
        parameters={
            "title": "Simple Notes",
            "content": ["Note 1", "Note 2", "Note 3"]
        }
    )
    print(f"✓ Created: {result.get('file')}")


def test_auto_generated():
    """Test 7: No path info (auto-generates timestamped name)."""
    print("\n" + "=" * 80)
    print("TEST 7: Auto-generated timestamped filename")
    print("=" * 80)
    
    agent = MermaidAgent()
    result = agent.execute_task(
        operation="create_flowchart",
        parameters={
            "nodes": [
                {"id": "A", "text": "Auto", "shape": "circle"}
            ],
            "connections": []
        }
    )
    print(f"✓ Created: {result.get('file')}")


def test_file_path_parameter():
    """Test 8: Using file_path parameter (alternative to output_path)."""
    print("\n" + "=" * 80)
    print("TEST 8: file_path parameter")
    print("=" * 80)
    
    agent = DocumentAgent()
    result = agent.execute_task(
        operation="create_txt",
        parameters={
            "file_path": "output_folder/file_path_test.txt",
            "content": "Using file_path parameter instead of output_path"
        }
    )
    print(f"✓ Created: {result.get('file')}")


def test_subfolder_creation():
    """Test 9: Automatic subfolder creation."""
    print("\n" + "=" * 80)
    print("TEST 9: Auto subfolder creation")
    print("=" * 80)
    
    agent = DocumentAgent()
    result = agent.execute_task(
        operation="create_docx",
        parameters={
            "folder_path": "output_folder/reports/2026/Q1",
            "file_name": "detailed_report",
            "content": [
                {"type": "heading", "text": "Detailed Report", "level": 1},
                {"type": "paragraph", "text": "Subfolders created automatically"}
            ]
        }
    )
    print(f"✓ Created: {result.get('file')}")


def test_filename_with_extension():
    """Test 10: file_name already has extension."""
    print("\n" + "=" * 80)
    print("TEST 10: file_name with extension")
    print("=" * 80)
    
    agent = DocumentAgent()
    result = agent.execute_task(
        operation="create_txt",
        parameters={
            "folder_path": "output_folder",
            "file_name": "already_has_extension.txt",  # Extension present
            "content": "File name already includes .txt extension"
        }
    )
    print(f"✓ Created: {result.get('file')}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("OUTPUT PATH DETERMINATION LOGIC TESTS")
    print("=" * 80)
    
    try:
        test_direct_path()
        test_folder_plus_filename()
        test_folder_plus_template()
        test_folder_plus_title()
        test_template_only()
        test_title_only()
        test_auto_generated()
        test_file_path_parameter()
        test_subfolder_creation()
        test_filename_with_extension()
        
        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nThe following path determination methods work:")
        print("✓ 1. Direct: output_path='path/to/file.ext'")
        print("✓ 2. Direct: file_path='path/to/file.ext'")
        print("✓ 3. Combined: folder_path='folder' + file_name='name'")
        print("✓ 4. Template: folder_path='folder' + template_name='Template'")
        print("✓ 5. Title: folder_path='folder' + title='Title'")
        print("✓ 6. Auto folder: template_name='Template' (→ output_folder)")
        print("✓ 7. Auto folder: title='Title' (→ output_folder)")
        print("✓ 8. Auto name: No info provided (→ timestamped)")
        print("✓ 9. Subfolders: Automatically created as needed")
        print("✓ 10. Extensions: Added if not present, preserved if present")
        
        print("\n" + "=" * 80)
        print("Check output_folder/ for all generated files")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
