"""
Quick test to verify operation alias normalization works.
"""

import sys
sys.path.insert(0, 'd:\\Projects\\TaskManager')

from task_manager.sub_agents.web_search_agent import WebSearchAgent
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_operation_aliases():
    """Test that various operation aliases work correctly."""
    agent = WebSearchAgent()
    
    # Test cases with different operation names and parameter aliases
    test_cases = [
        {
            'name': 'Standard search operation',
            'operation': 'search',
            'parameters': {'query': 'test query', 'max_results': 5}
        },
        {
            'name': 'web_search alias (the failing one)',
            'operation': 'web_search',
            'parameters': {'query': 'CPG supply chain trends', 'num_results': 5}
        },
        {
            'name': 'websearch alias',
            'operation': 'websearch',
            'parameters': {'query': 'test', 'max_results': 3}
        },
        {
            'name': 'search_web alias',
            'operation': 'search_web',
            'parameters': {'query': 'test', 'max_results': 3}
        }
    ]
    
    print("=" * 80)
    print("Testing Operation Alias Normalization")
    print("=" * 80)
    
    for test in test_cases:
        print(f"\n{test['name']}")
        print(f"Operation: {test['operation']}")
        print(f"Parameters: {test['parameters']}")
        print("-" * 40)
        
        try:
            result = agent.execute_task(
                operation=test['operation'],
                parameters=test['parameters']
            )
            
            if result.get('success') is not None:
                status = "✓ SUCCESS" if result.get('success') else "✗ FAILED"
                print(f"{status}")
                if not result.get('success'):
                    print(f"Error: {result.get('error', 'Unknown')}")
                else:
                    print(f"Results: {result.get('results_count', 0)}")
            else:
                print(f"⚠ WARNING: No success field in result")
                print(f"Result: {result}")
                
        except Exception as e:
            print(f"✗ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    test_operation_aliases()
