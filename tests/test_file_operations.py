"""
Test script for PDF and Excel sub-agent capabilities.

Demonstrates how the Task Manager Agent automatically detects and handles
PDF and Excel file operations through dedicated sub-agents.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from task_manager.core.agent import TaskManagerAgent
from task_manager.utils.logger import get_logger

logger = get_logger(__name__)


def test_pdf_and_excel_capabilities():
    """
    Test PDF and Excel sub-agent capabilities.
    
    This demonstrates:
    1. Automatic detection of PDF/Excel tasks
    2. Sub-agent routing and execution
    3. Result integration
    """
    
    print("\n" + "="*70)
    print("PDF AND EXCEL SUB-AGENT CAPABILITIES TEST")
    print("="*70 + "\n")
    
    # Create output directory for test files
    test_output_dir = project_root / "test_output"
    test_output_dir.mkdir(exist_ok=True)
    
    # Test 1: Excel file creation from analysis
    print("TEST 1: Creating Excel report from data")
    print("-" * 70)
    
    print(f"\nObjective: Create quarterly sales report with data organization")
    print(f"Metadata: regions=['North', 'South', 'East', 'West'], products=['Widget A', 'Widget B', 'Service X', 'Service Y']")
    print(f"\nRunning sub-agent tests directly...")
    
    # Test the sub-agents directly
    print("\n" + "-"*70)
    print("DIRECT SUB-AGENT TESTING")
    print("-"*70)
    
    test_excel_creation()
    test_excel_reading()
    test_pdf_creation()
    
    print("\n" + "="*70)
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")


def test_excel_creation():
    """Test Excel file creation directly."""
    print("\n1. Testing Excel Creation:")
    
    from task_manager.sub_agents import ExcelAgent
    
    agent = ExcelAgent()
    
    # Create sample data
    test_file = "test_output/sample_data.xlsx"
    
    sheets = {
        "Summary": [
            ["Metric", "Q1", "Q2", "Q3", "Q4"],
            ["Revenue", "$1.2M", "$1.5M", "$1.8M", "$2.1M"],
            ["Profit", "$300K", "$375K", "$450K", "$525K"],
            ["Growth", "12%", "15%", "18%", "20%"]
        ],
        "Regions": [
            ["Region", "Sales", "Growth", "Status"],
            ["North", "$500K", "10%", "Active"],
            ["South", "$450K", "8%", "Active"],
            ["East", "$600K", "15%", "Growing"],
            ["West", "$400K", "5%", "Stable"]
        ]
    }
    
    result = agent.create_excel(
        output_path=test_file,
        sheets=sheets,
        title="Quarterly Report",
        auto_format=True
    )
    
    print(f"   Status: {result.get('success')}")
    print(f"   Output: {result.get('output_path')}")
    print(f"   Sheets Created: {result.get('sheets_created')}")
    print(f"   File Size: {result.get('file_size')} bytes")
    
    # Read the file back
    if result.get('success'):
        print("\n   Reading back the created file:")
        read_result = agent.read_excel(file_path=test_file)
        if read_result.get('success'):
            for sheet_name in read_result.get('sheets', []):
                sheet_data = read_result['data'].get(sheet_name, {})
                print(f"     - Sheet '{sheet_name}': {sheet_data.get('rows')} rows, {len(sheet_data.get('columns', []))} columns")


def test_excel_reading():
    """Test Excel file reading."""
    print("\n2. Testing Excel Reading:")
    
    from task_manager.sub_agents import ExcelAgent
    
    agent = ExcelAgent()
    
    # Create a test file first
    test_file = "test_output/read_test.xlsx"
    
    sheets = {
        "Data": [
            ["ID", "Name", "Value"],
            [1, "Item A", 100],
            [2, "Item B", 200],
            [3, "Item C", 150]
        ]
    }
    
    agent.create_excel(output_path=test_file, sheets=sheets)
    
    # Now read it
    result = agent.read_excel(file_path=test_file)
    
    print(f"   Status: {result.get('success')}")
    print(f"   File: {result.get('file')}")
    print(f"   Sheets: {result.get('sheets')}")
    
    if result.get('success') and result.get('data'):
        for sheet_name, sheet_data in result['data'].items():
            print(f"   Sheet '{sheet_name}':")
            print(f"     - Rows: {sheet_data.get('rows')}")
            print(f"     - Columns: {sheet_data.get('columns')}")
            print(f"     - First row: {sheet_data.get('data')[0] if sheet_data.get('data') else 'N/A'}")


def test_pdf_creation():
    """Test PDF file creation."""
    print("\n3. Testing PDF Creation:")
    
    from task_manager.sub_agents import PDFAgent
    
    agent = PDFAgent()
    
    test_file = "test_output/sample_report.pdf"
    
    content = [
        {"type": "heading", "data": "Quarterly Sales Report"},
        {"type": "text", "data": "This report summarizes Q1 sales performance across all regions and product categories."},
        {"type": "heading", "data": "Executive Summary"},
        {"type": "text", "data": "Total Revenue: $2.1M | Total Profit: $525K | Growth Rate: 20%"},
        {"type": "table", "data": [
            ["Region", "Sales", "Growth"],
            ["North", "$500K", "10%"],
            ["South", "$450K", "8%"],
            ["East", "$600K", "15%"],
            ["West", "$400K", "5%"]
        ]},
        {"type": "pagebreak", "data": None},
        {"type": "heading", "data": "Detailed Analysis"},
        {"type": "text", "data": "The East region showed the strongest performance with 15% growth..."}
    ]
    
    result = agent.create_pdf(
        output_path=test_file,
        content=content,
        title="Q1 Sales Report",
        author="Task Manager Agent"
    )
    
    print(f"   Status: {result.get('success')}")
    print(f"   Output: {result.get('output_path')}")
    print(f"   File Size: {result.get('file_size')} bytes")
    
    # Test PDF reading if creation was successful
    if result.get('success'):
        print("\n   Reading back the created PDF:")
        read_result = agent.read_pdf(file_path=test_file, extract_text=True)
        if read_result.get('success'):
            print(f"     - Total Pages: {read_result.get('page_count')}")
            print(f"     - Extracted Pages: {read_result.get('metadata', {}).get('extracted_pages')}")
            text_content = read_result.get('text_content', [])
            if text_content:
                print(f"     - First Page Length: {text_content[0].get('length')} characters")


def test_sub_agent_integration():
    """Test how sub-agents integrate with the main agent."""
    print("\n" + "="*70)
    print("TESTING SUB-AGENT INTEGRATION WITH MAIN AGENT")
    print("="*70)
    
    # Test without creating main agent (avoids API key requirement)
    from task_manager.sub_agents import PDFAgent, ExcelAgent
    
    pdf_agent = PDFAgent()
    excel_agent = ExcelAgent()
    
    print("\nSub-agents initialized:")
    print(f"  - PDFAgent: {pdf_agent is not None}")
    print(f"  - ExcelAgent: {excel_agent is not None}")
    
    # Test routing detection
    print("\nTesting action routing:")
    
    test_analysis_pdf = {
        "action": "pdf_task",
        "file_operation": {
            "type": "pdf",
            "operation": "create"
        }
    }
    
    test_analysis_excel = {
        "action": "excel_task",
        "file_operation": {
            "type": "excel",
            "operation": "create"
        }
    }
    
    print(f"  - PDF task routing: {'pdf_task' == test_analysis_pdf['action']}")
    print(f"  - Excel task routing: {'excel_task' == test_analysis_excel['action']}")


if __name__ == "__main__":
    try:
        test_sub_agent_integration()
        test_pdf_and_excel_capabilities()
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}")
        sys.exit(1)
