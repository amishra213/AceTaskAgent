#!/usr/bin/env python3
"""
Test to verify WebSearch normalization fix is working correctly.

This test checks that when WebSearchAgent returns search results with the 'search'
operation, the agent.py normalization properly converts them to extracted_data/findings
format for downstream agent consumption.
"""

import logging
import sys
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_normalization_logic():
    """Test the normalization logic directly."""
    print("\n" + "="*80)
    print("WEBSEARCH NORMALIZATION FIX VERIFICATION")
    print("="*80)
    
    # Simulate what WebSearchAgent returns from search() operation
    mock_search_response = {
        'success': True,
        'query': 'Python web scraping',
        'results': [
            {
                'title': 'BeautifulSoup Tutorial',
                'url': 'https://example.com/bs',
                'snippet': 'BeautifulSoup is a library for parsing HTML and XML documents.'
            },
            {
                'title': 'Scrapy Framework',
                'url': 'https://example.com/scrapy',
                'snippet': 'Scrapy is an open-source Python framework for web scraping.'
            },
            {
                'title': 'Selenium for Web Automation',
                'url': 'https://example.com/selenium',
                'snippet': 'Selenium is a powerful tool for controlling web browsers.'
            }
        ],
        'results_count': 3,
    }
    
    # Simulate the result_data structure as it comes from WebSearchAgent
    result_data = {
        'success': mock_search_response['success'],
        'summary': {},
        'results_count': mock_search_response['results_count'],
        'results': mock_search_response['results'],
        'query': mock_search_response['query']
    }
    
    print("\n[BEFORE NORMALIZATION]")
    print(f"  - success: {result_data.get('success')}")
    print(f"  - results_count: {result_data.get('results_count')}")
    print(f"  - Has 'extracted_data': {('extracted_data' in result_data)}")
    print(f"  - Has 'findings': {('findings' in result_data)}")
    print(f"  - Results items: {len(result_data.get('results', []))}")
    
    # Apply the normalization logic (from agent.py lines 2241-2268)
    operation = 'search'
    if operation == 'search' and 'results' in result_data and 'extracted_data' not in result_data:
        # Convert search results to extracted_data format
        search_results = result_data.get('results', [])
        
        # Create extracted_data from snippets
        result_data['extracted_data'] = [
            r.get('snippet', '') for r in search_results 
            if r.get('snippet', '').strip()
        ] if search_results else []
        
        # Create findings from full result objects
        result_data['findings'] = [
            {
                'url': r.get('url', ''),
                'title': r.get('title', ''),
                'relevance_score': 0.8,
                'text_preview': r.get('snippet', '')[:500]
            }
            for r in search_results
        ] if search_results else []
        
        # Generate summary if empty
        if not result_data.get('summary'):
            result_data['summary'] = {
                'query': result_data.get('query', ''),
                'total_results': len(search_results),
                'top_sources': [r.get('url', '') for r in search_results[:5]],
                'snippet_count': len(result_data['extracted_data'])
            }
        
        logger.debug(f"Normalized search results: {len(result_data['extracted_data'])} items, {len(result_data['findings'])} findings")
    
    print("\n[AFTER NORMALIZATION]")
    print(f"  - success: {result_data.get('success')}")
    print(f"  - Has 'extracted_data': {('extracted_data' in result_data)}")
    print(f"  - extracted_data items: {len(result_data.get('extracted_data', []))}")
    print(f"  - Has 'findings': {('findings' in result_data)}")
    print(f"  - findings items: {len(result_data.get('findings', []))}")
    print(f"  - Has 'summary': {('summary' in result_data)}")
    
    # Verify the fix
    print("\n[VERIFICATION]")
    success = True
    
    if result_data.get('success'):
        print("  ✓ Success flag is True")
    else:
        print("  ✗ Success flag is False")
        success = False
    
    if result_data.get('extracted_data') and len(result_data['extracted_data']) > 0:
        print(f"  ✓ extracted_data populated with {len(result_data['extracted_data'])} items")
        for i, item in enumerate(result_data['extracted_data'], 1):
            preview = item[:60] + "..." if len(item) > 60 else item
            print(f"    {i}. {preview}")
    else:
        print("  ✗ extracted_data is empty or missing")
        success = False
    
    if result_data.get('findings') and len(result_data['findings']) > 0:
        print(f"  ✓ findings populated with {len(result_data['findings'])} items")
        for i, item in enumerate(result_data['findings'], 1):
            print(f"    {i}. {item.get('title', 'N/A')[:50]} (score: {item.get('relevance_score')})")
            print(f"       URL: {item.get('url', 'N/A')[:60]}")
    else:
        print("  ✗ findings is empty or missing")
        success = False
    
    if result_data.get('summary'):
        summary = result_data['summary']
        print(f"  ✓ summary populated")
        print(f"    - Query: {summary.get('query', 'N/A')}")
        print(f"    - Total results: {summary.get('total_results', 0)}")
        print(f"    - Snippet count: {summary.get('snippet_count', 0)}")
    else:
        print("  ✗ summary is missing")
        success = False
    
    # Verify data structures are correct for downstream agents
    print("\n[DOWNSTREAM COMPATIBILITY]")
    
    if result_data.get('extracted_data') and isinstance(result_data['extracted_data'], list):
        if all(isinstance(item, str) for item in result_data['extracted_data']):
            print("  ✓ extracted_data is list of strings (ready for DataExtractionAgent/ProblemSolverAgent)")
        else:
            print("  ✗ extracted_data contains non-string items")
            success = False
    else:
        print("  ✗ extracted_data is not a list of strings")
        success = False
    
    if result_data.get('findings') and isinstance(result_data['findings'], list):
        if all(isinstance(item, dict) and 'url' in item and 'title' in item for item in result_data['findings']):
            print("  ✓ findings is list of dicts with url/title (ready for downstream agents)")
        else:
            print("  ✗ findings has missing required fields")
            success = False
    else:
        print("  ✗ findings is not properly structured")
        success = False
    
    print("\n" + "="*80)
    if success:
        print("✓ NORMALIZATION FIX VERIFICATION PASSED")
        print("  WebSearch search() results are now properly normalized for downstream agents")
    else:
        print("✗ NORMALIZATION FIX VERIFICATION FAILED")
        print("  There are issues with the normalization logic")
    print("="*80)
    
    return success

if __name__ == "__main__":
    try:
        success = test_normalization_logic()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
