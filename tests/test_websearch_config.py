"""
Test web search with different backends configured via environment variables.
"""

import sys
import os
sys.path.insert(0, 'd:\\Projects\\TaskManager')

# Set environment variables before importing
os.environ['WEBSEARCH_BACKEND'] = 'api'  # Test with API backend
os.environ['WEBSEARCH_REGION'] = 'wt-wt'  # Worldwide region
os.environ['WEBSEARCH_TIMEOUT'] = '15'

from task_manager.sub_agents.web_search_agent import WebSearchAgent
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_backend_configuration():
    """Test that backend configuration is loaded from environment."""
    print("=" * 80)
    print("Test: Backend Configuration from Environment")
    print("=" * 80)
    
    agent = WebSearchAgent()
    
    print(f"\n✓ Agent initialized")
    print(f"  - Default backend: {agent.default_backend}")
    print(f"  - Default region: {agent.default_region}")
    print(f"  - Search timeout: {agent.search_timeout}")
    print(f"  - Custom URL: {agent.custom_search_url}")
    
    # Verify values match environment
    assert agent.default_backend == 'api', f"Expected 'api', got '{agent.default_backend}'"
    assert agent.default_region == 'wt-wt', f"Expected 'wt-wt', got '{agent.default_region}'"
    assert agent.search_timeout == 15, f"Expected 15, got {agent.search_timeout}"
    
    print("\n✓ All environment configurations loaded correctly!")


def test_search_with_different_backends():
    """Test searching with different backends."""
    print("\n" + "=" * 80)
    print("Test: Search with Different Backends")
    print("=" * 80)
    
    agent = WebSearchAgent()
    query = "Python programming tutorial"
    
    backends_to_test = ['html', 'api', 'lite']
    
    for backend in backends_to_test:
        print(f"\nTesting backend='{backend}'...")
        
        try:
            result = agent.search(
                query=query,
                max_results=3,
                backend=backend,
                retry_on_empty=False  # Disable retry for faster testing
            )
            
            if result.get('success'):
                count = result.get('results_count', 0)
                print(f"  ✓ Success: {count} results")
                if count > 0:
                    first_result = result['results'][0]
                    print(f"    First: {first_result.get('title', 'N/A')[:60]}...")
            else:
                print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"  ✗ Exception: {str(e)[:60]}")


def test_default_vs_explicit_backend():
    """Test that explicit backend parameter overrides default."""
    print("\n" + "=" * 80)
    print("Test: Explicit Backend Override")
    print("=" * 80)
    
    # Agent has default backend='api' from environment
    agent = WebSearchAgent()
    
    print(f"\nAgent default backend: {agent.default_backend}")
    
    # Override with explicit parameter
    print("\nSearching with explicit backend='html' (should override default)...")
    result = agent.search(
        query="test",
        max_results=2,
        backend='html',  # Explicit override
        retry_on_empty=False
    )
    
    if result.get('success'):
        print(f"  ✓ Success with explicit backend='html'")
        print(f"  Results: {result.get('results_count', 0)}")
    else:
        print(f"  ✗ Failed: {result.get('error')}")
    
    # Use default backend
    print(f"\nSearching with default backend (should use '{agent.default_backend}')...")
    result = agent.search(
        query="test",
        max_results=2,
        retry_on_empty=False
    )
    
    if result.get('success'):
        print(f"  ✓ Success with default backend")
        print(f"  Results: {result.get('results_count', 0)}")
    else:
        print(f"  ✗ Failed: {result.get('error')}")


def test_region_configuration():
    """Test region/language configuration."""
    print("\n" + "=" * 80)
    print("Test: Region Configuration")
    print("=" * 80)
    
    agent = WebSearchAgent()
    
    print(f"\nDefault region from env: {agent.default_region}")
    
    # Test with different regions
    regions = ['wt-wt', 'us', 'uk']
    
    for region in regions:
        print(f"\nSearching with region='{region}'...")
        result = agent.search(
            query="news",
            max_results=2,
            language=region,
            retry_on_empty=False
        )
        
        if result.get('success'):
            print(f"  ✓ Success: {result.get('results_count', 0)} results")
        else:
            print(f"  ✗ Failed: {result.get('error', 'Unknown')[:50]}")


if __name__ == "__main__":
    print("Web Search Backend Configuration Tests")
    print("=" * 80)
    
    try:
        test_backend_configuration()
        test_search_with_different_backends()
        test_default_vs_explicit_backend()
        test_region_configuration()
        
        print("\n" + "=" * 80)
        print("✓ All tests completed!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
