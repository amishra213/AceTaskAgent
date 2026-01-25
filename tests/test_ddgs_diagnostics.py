"""
Diagnostic script to understand DDGS behavior and potential issues.
Tests:
1. Direct DDGS usage
2. Iterator consumption patterns
3. Error handling
4. Rate limiting detection
"""

import sys
sys.path.insert(0, 'd:\\Projects\\TaskManager')

import logging
import time

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_ddgs_direct():
    """Test DDGS library directly with various patterns."""
    print("=" * 80)
    print("TEST 1: Direct DDGS Usage")
    print("=" * 80)
    
    try:
        from ddgs import DDGS
        
        ddgs = DDGS()
        print("✓ DDGS initialized successfully")
        
        # Test 1: Simple query
        print("\nTest 1a: Simple query with list conversion")
        try:
            results = ddgs.text("Python programming", max_results=3)
            print(f"Type of results: {type(results)}")
            
            # Convert to list to consume iterator
            results_list = list(results)
            print(f"Results count: {len(results_list)}")
            
            if results_list:
                print("Sample result:")
                print(f"  Title: {results_list[0].get('title', 'N/A')}")
                print(f"  URL: {results_list[0].get('href', 'N/A')}")
                print(f"  Keys: {list(results_list[0].keys())}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 2: Enumerate pattern (like in the code)
        print("\nTest 1b: Enumerate pattern (like in actual code)")
        try:
            results = ddgs.text("Python programming", max_results=3)
            formatted_results = []
            
            for idx, result in enumerate(results, 1):
                print(f"  Processing result {idx}...")
                formatted_results.append({
                    "rank": idx,
                    "title": result.get('title', ''),
                    "url": result.get('href', ''),
                    "snippet": result.get('body', ''),
                })
            
            print(f"Formatted results count: {len(formatted_results)}")
            
        except Exception as e:
            print(f"✗ Error during enumeration: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 3: Empty results query
        print("\nTest 1c: Query that might return no results")
        try:
            results = ddgs.text("xyzabc123impossible456query789", max_results=5)
            results_list = list(results)
            print(f"Results count for impossible query: {len(results_list)}")
            
        except Exception as e:
            print(f"✗ Error with impossible query: {e}")
        
        # Test 4: Multiple rapid searches (rate limiting test)
        print("\nTest 1d: Multiple rapid searches (rate limit test)")
        for i in range(3):
            try:
                results = list(ddgs.text(f"test query {i}", max_results=2))
                print(f"  Search {i+1}: {len(results)} results")
                time.sleep(0.5)  # Small delay
            except Exception as e:
                print(f"  Search {i+1}: Error - {e}")
        
    except ImportError as e:
        print(f"✗ Failed to import DDGS: {e}")
        return


def test_web_search_agent():
    """Test the actual WebSearchAgent implementation."""
    print("\n" + "=" * 80)
    print("TEST 2: WebSearchAgent Implementation")
    print("=" * 80)
    
    try:
        from task_manager.sub_agents.web_search_agent import WebSearchAgent
        
        agent = WebSearchAgent()
        print("✓ WebSearchAgent initialized")
        
        # Test with normal query
        print("\nTest 2a: Normal search query")
        result = agent.search(query="Python programming", max_results=3, retry_on_empty=False)
        
        print(f"Success: {result.get('success')}")
        print(f"Results count: {result.get('results_count', 0)}")
        print(f"Error: {result.get('error', 'None')}")
        
        if result.get('results'):
            print(f"First result URL: {result['results'][0].get('url', 'N/A')}")
        
        # Test with difficult query
        print("\nTest 2b: Difficult query (with retry)")
        result2 = agent.search(
            query="very specific obscure topic xyz123",
            max_results=3,
            retry_on_empty=True,
            max_retries=2
        )
        
        print(f"Success: {result2.get('success')}")
        print(f"Results count: {result2.get('results_count', 0)}")
        print(f"Retry count: {result2.get('retry_count', 0)}")
        print(f"Error: {result2.get('error', 'None')}")
        
        if result2.get('attempts'):
            print("Attempts:")
            for att in result2.get('attempts', []):
                print(f"  {att}")
        
    except Exception as e:
        print(f"✗ Error testing WebSearchAgent: {e}")
        import traceback
        traceback.print_exc()


def test_captcha_detection():
    """Check if DDGS is returning CAPTCHA pages."""
    print("\n" + "=" * 80)
    print("TEST 3: CAPTCHA/Bot Detection")
    print("=" * 80)
    
    try:
        from ddgs import DDGS
        import requests
        
        ddgs = DDGS()
        results = list(ddgs.text("test", max_results=1))
        
        if not results:
            print("⚠ Warning: No results returned - possible bot detection")
            return
        
        # Try to fetch the first URL to see if it's accessible
        if results:
            test_url = results[0].get('href', '')
            print(f"Testing URL: {test_url}")
            
            try:
                response = requests.get(
                    test_url,
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'},
                    timeout=5
                )
                print(f"HTTP Status: {response.status_code}")
                
                if 'captcha' in response.text.lower() or 'robot' in response.text.lower():
                    print("⚠ Warning: CAPTCHA detected in response")
                else:
                    print("✓ URL accessible, no obvious CAPTCHA")
                    
            except Exception as e:
                print(f"Error accessing URL: {e}")
        
    except Exception as e:
        print(f"Error in CAPTCHA detection: {e}")


if __name__ == "__main__":
    print("DDGS Diagnostic Tests")
    print("=" * 80)
    
    test_ddgs_direct()
    test_web_search_agent()
    test_captcha_detection()
    
    print("\n" + "=" * 80)
    print("Diagnostic tests completed!")
    print("=" * 80)
