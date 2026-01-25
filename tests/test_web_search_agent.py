"""
Test Web Search Sub-Agent Integration

Tests the WebSearchAgent with the main TaskManagerAgent.
"""

import logging
from task_manager.sub_agents import WebSearchAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_web_search_agent():
    """Test web search agent functionality."""
    print("\n" + "="*70)
    print("WEB SEARCH SUB-AGENT TESTING")
    print("="*70)
    
    # Initialize agent
    agent = WebSearchAgent()
    print(f"\n✓ WebSearchAgent initialized")
    print(f"  Supported operations: {agent.supported_operations}")
    print(f"  Search library: {agent.search_lib}")
    
    # Test 1: Web Search
    print("\n" + "-"*70)
    print("TEST 1: Web Search")
    print("-"*70)
    
    search_result = agent.search(
        query="Python web scraping libraries",
        max_results=5
    )
    
    if search_result.get('success'):
        print(f"✓ Search successful")
        print(f"  Query: {search_result.get('query')}")
        print(f"  Results count: {search_result.get('results_count')}")
        if search_result.get('results'):
            print(f"  First result:")
            first = search_result['results'][0]
            print(f"    - Title: {first.get('title', 'N/A')[:60]}")
            print(f"    - URL: {first.get('url', 'N/A')[:60]}")
            print(f"    - Snippet: {first.get('snippet', 'N/A')[:80]}")
    else:
        print(f"✗ Search failed: {search_result.get('error')}")
    
    # Test 2: Web Scraping
    print("\n" + "-"*70)
    print("TEST 2: Web Scraping")
    print("-"*70)
    
    scrape_result = agent.scrape(
        url="https://www.example.com",
        extract_text=True,
        extract_links=True,
        extract_images=False
    )
    
    if scrape_result.get('success'):
        print(f"✓ Scraping successful")
        print(f"  URL: {scrape_result.get('url')}")
        print(f"  Title: {scrape_result.get('title', 'N/A')[:60]}")
        print(f"  Status code: {scrape_result.get('status_code')}")
        print(f"  Text length: {scrape_result.get('text_length', 0)} chars")
        print(f"  Links found: {scrape_result.get('links_count', 0)}")
        if scrape_result.get('links'):
            print(f"  First link: {scrape_result['links'][0].get('text', 'N/A')[:50]}")
    else:
        print(f"✗ Scraping failed: {scrape_result.get('error')}")
    
    # Test 3: Content Fetch
    print("\n" + "-"*70)
    print("TEST 3: Content Fetch")
    print("-"*70)
    
    fetch_result = agent.fetch(
        url="https://www.example.com",
        save_to_file=None
    )
    
    if fetch_result.get('success'):
        print(f"✓ Fetch successful")
        print(f"  URL: {fetch_result.get('url')}")
        print(f"  Status code: {fetch_result.get('status_code')}")
        print(f"  Content type: {fetch_result.get('content_type')}")
        print(f"  Content length: {fetch_result.get('content_length')} bytes")
        content_preview = fetch_result.get('content_preview', '')
        if content_preview:
            print(f"  Preview: {content_preview[:80]}")
    else:
        print(f"✗ Fetch failed: {fetch_result.get('error')}")
    
    # Test 4: Content Summarization
    print("\n" + "-"*70)
    print("TEST 4: Content Summarization")
    print("-"*70)
    
    summarize_result = agent.summarize(
        url="https://www.example.com",
        max_sentences=3
    )
    
    if summarize_result.get('success'):
        print(f"✓ Summarization successful")
        print(f"  URL: {summarize_result.get('url')}")
        print(f"  Title: {summarize_result.get('title', 'N/A')[:60]}")
        print(f"  Summary: {summarize_result.get('summary', 'N/A')[:150]}")
    else:
        print(f"✗ Summarization failed: {summarize_result.get('error')}")
    
    # Test 5: Execute Task Interface
    print("\n" + "-"*70)
    print("TEST 5: Execute Task Interface")
    print("-"*70)
    
    # Test search via execute_task
    search_via_execute = agent.execute_task(
        operation="search",
        parameters={
            "query": "AI agents",
            "max_results": 3
        }
    )
    
    if search_via_execute.get('success'):
        print(f"✓ Execute task (search) successful")
        print(f"  Results count: {search_via_execute.get('results_count')}")
    else:
        print(f"✗ Execute task (search) failed: {search_via_execute.get('error')}")
    
    # Test unknown operation
    unknown_op = agent.execute_task(
        operation="unknown_op",
        parameters={}
    )
    
    if not unknown_op.get('success'):
        print(f"✓ Unknown operation handled correctly")
        print(f"  Error: {unknown_op.get('error')}")
        print(f"  Supported: {unknown_op.get('supported_operations')}")
    
    print("\n" + "="*70)
    print("WEB SEARCH SUB-AGENT TESTING COMPLETED")
    print("="*70)


if __name__ == "__main__":
    test_web_search_agent()
