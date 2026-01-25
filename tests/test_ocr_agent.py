"""
Test script for OCR and Image Agent functionality.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from task_manager.sub_agents import OCRImageAgent
from task_manager.utils.logger import get_logger

logger = get_logger(__name__)


def test_ocr_agent_initialization():
    """Test OCR agent initialization."""
    print("\n" + "="*60)
    print("Testing OCR Agent Initialization")
    print("="*60)
    
    try:
        agent = OCRImageAgent()
        print(f"✓ OCR Agent initialized successfully")
        print(f"  Available OCR engines: {agent.ocr_engines}")
        print(f"  Preferred engine: {agent.preferred_engine}")
        print(f"  PIL available: {agent.has_pil}")
        print(f"  pdf2image available: {agent.has_pdf2image}")
        print(f"  Supported operations: {agent.supported_operations}")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize OCR Agent: {str(e)}")
        return False


def test_ocr_capabilities():
    """Test OCR capabilities detection."""
    print("\n" + "="*60)
    print("Testing OCR Capabilities")
    print("="*60)
    
    agent = OCRImageAgent()
    
    if agent.ocr_engines:
        print(f"✓ OCR engines available: {', '.join(agent.ocr_engines)}")
    else:
        print("! No OCR engines available")
        print("  Install with:")
        print("    pip install pytesseract pillow")
        print("    pip install easyocr")
        print("    pip install paddlepaddle paddleocr")
    
    if agent.has_pdf2image:
        print("✓ pdf2image available for PDF to image conversion")
    else:
        print("! pdf2image not available")
        print("  Install with: pip install pdf2image")
    
    return True


def test_ocr_task_execution():
    """Test OCR task execution with dummy parameters."""
    print("\n" + "="*60)
    print("Testing OCR Task Execution")
    print("="*60)
    
    agent = OCRImageAgent()
    
    # Test with non-existent file to verify error handling
    result = agent.execute_task(
        operation="ocr_image",
        parameters={
            "image_path": "non_existent_image.png",
            "language": "eng"
        }
    )
    
    print(f"Test result: {result}")
    
    if not result.get('success'):
        print(f"✓ Error handling works correctly")
        print(f"  Error message: {result.get('error', 'Unknown error')}")
        return True
    else:
        print(f"✗ Unexpected success with non-existent file")
        return False


def test_agent_integration():
    """Test integration with main agent."""
    print("\n" + "="*60)
    print("Testing Agent Integration")
    print("="*60)
    
    try:
        from task_manager.core.agent import TaskManagerAgent
        from task_manager.config import AgentConfig
        
        config = AgentConfig()
        agent = TaskManagerAgent(
            objective="Test OCR integration",
            config=config
        )
        
        if hasattr(agent, 'ocr_image_agent'):
            print("✓ OCR Image Agent integrated into TaskManagerAgent")
            print(f"  OCR Agent type: {type(agent.ocr_image_agent)}")
            return True
        else:
            print("✗ OCR Image Agent not found in TaskManagerAgent")
            return False
    
    except Exception as e:
        print(f"✗ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("OCR AND IMAGE AGENT TEST SUITE")
    print("="*60)
    
    tests = [
        ("Agent Initialization", test_ocr_agent_initialization),
        ("OCR Capabilities", test_ocr_capabilities),
        ("Task Execution", test_ocr_task_execution),
        ("Agent Integration", test_agent_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
