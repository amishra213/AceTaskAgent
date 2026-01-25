"""
Test script to demonstrate the Web Search Agent retry functionality.

This script tests the improved search with automatic retry on zero results.
"""

import sys
sys.path.insert(0, 'd:\\Projects\\TaskManager')

from task_manager.sub_agents.web_search_agent import WebSearchAgent
import logging

# Configure logging to see retry behavior
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_search_with_retry():
    """Test search with automatic retry on empty results."""
    agent = WebSearchAgent()
    
    print("=" * 80)
    print("TEST 1: Normal query that should return results")
    print("=" * 80)
    
    result1 = agent.search(
        query="Python programming tutorial",
        max_results=5,
        retry_on_empty=True,
        max_retries=3
    )
    
    print(f"\nSuccess: {result1.get('success')}")
    print(f"Results Count: {result1.get('results_count')}")
    print(f"Retry Count: {result1.get('retry_count', 0)}")
    if result1.get('attempts'):
        print("\nAttempts:")
        for attempt in result1.get('attempts', []):
            print(f"  {attempt}")
    
    print("\n" + "=" * 80)
    print("TEST 2: Query that might need reformulation")
    print("=" * 80)
    
    # This query might be too specific and need reformulation
    result2 = agent.search(
        query="very specific obscure topic that probably does not exist xyz123abc",
        max_results=5,
        retry_on_empty=True,
        max_retries=3
    )
    
    print(f"\nSuccess: {result2.get('success')}")
    print(f"Results Count: {result2.get('results_count')}")
    print(f"Retry Count: {result2.get('retry_count', 0)}")
    print(f"Error: {result2.get('error', 'None')}")
    
    if result2.get('attempts'):
        print("\nAttempts:")
        for attempt in result2.get('attempts', []):
            print(f"  Attempt {attempt.get('attempt')}: '{attempt.get('query')}' -> {attempt.get('results_count')} results")
    
    print("\n" + "=" * 80)
    print("TEST 3: Query with retry disabled")
    print("=" * 80)
    
    result3 = agent.search(
        query="another test query",
        max_results=5,
        retry_on_empty=False,  # Disable retry
        max_retries=0
    )
    
    print(f"\nSuccess: {result3.get('success')}")
    print(f"Results Count: {result3.get('results_count')}")
    print(f"Retry Count: {result3.get('retry_count', 0)}")


def test_deep_search_with_retry():
    """Test deep search which now uses improved search."""
    agent = WebSearchAgent()
    
    print("\n" + "=" * 80)
    print("TEST 4: Deep search with automatic retry")
    print("=" * 80)
    
    result = agent.deep_search(
        query="Karnataka districts India",
        max_results=3,
        max_depth=1
    )
    
    print(f"\nSuccess: {result.get('success')}")
    print(f"Pages Visited: {result.get('pages_visited', 0)}")
    print(f"Extracted Items: {len(result.get('extracted_data', []))}")
    print(f"Error: {result.get('error', 'None')}")
    
    if result.get('search_attempts'):
        print("\nSearch Attempts:")
        for attempt in result.get('search_attempts', []):
            print(f"  {attempt}")


if __name__ == "__main__":
    print("Testing Web Search Agent with Retry Functionality")
    print("=" * 80)
    
    try:
        test_search_with_retry()
        test_deep_search_with_retry()
        
        print("\n" + "=" * 80)
        print("✓ All tests completed!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
