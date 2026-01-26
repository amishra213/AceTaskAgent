"""
Quick syntax check for new agents.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Checking imports...")

try:
    from task_manager.sub_agents.document_agent import DocumentAgent
    print("✓ DocumentAgent imported successfully")
except Exception as e:
    print(f"✗ DocumentAgent import failed: {e}")
    sys.exit(1)

try:
    from task_manager.sub_agents.mermaid_agent import MermaidAgent
    print("✓ MermaidAgent imported successfully")
except Exception as e:
    print(f"✗ MermaidAgent import failed: {e}")
    sys.exit(1)

try:
    from task_manager.sub_agents import DocumentAgent, MermaidAgent
    print("✓ Agents imported from __init__ successfully")
except Exception as e:
    print(f"✗ __init__ import failed: {e}")
    sys.exit(1)

print("\nCreating agent instances...")

try:
    doc_agent = DocumentAgent()
    print(f"✓ DocumentAgent instance created")
    print(f"  - Agent name: {doc_agent.agent_name}")
    print(f"  - Supported operations: {', '.join(doc_agent.supported_operations)}")
except Exception as e:
    print(f"✗ DocumentAgent instantiation failed: {e}")
    sys.exit(1)

try:
    mermaid_agent = MermaidAgent()
    print(f"✓ MermaidAgent instance created")
    print(f"  - Agent name: {mermaid_agent.agent_name}")
    print(f"  - Supported operations: {', '.join(mermaid_agent.supported_operations)}")
except Exception as e:
    print(f"✗ MermaidAgent instantiation failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("SUCCESS: All agents loaded and initialized correctly!")
print("=" * 60)
print("\nNext steps:")
print("1. Install python-docx for DOCX support: pip install python-docx")
print("2. Run test scripts:")
print("   - tests/test_document_agent.py")
print("   - tests/test_mermaid_agent.py")
print("3. Try the combined example:")
print("   - examples/combined_document_mermaid_example.py")
