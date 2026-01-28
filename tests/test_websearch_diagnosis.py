#!/usr/bin/env python3
"""
Diagnostic script to test web search functionality directly.
"""

import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def test_ddgs_direct():
    """Test DDGS library directly."""
    logger.info("="*80)
    logger.info("TEST 1: Direct DDGS Import and Search")
    logger.info("="*80)
    
    try:
        from ddgs import DDGS
        logger.info("✓ DDGS imported successfully")
        
        ddgs = DDGS()
        logger.info("✓ DDGS instance created")
        
        # Test simple search
        query = "supply chain trends CPG 2026"
        logger.info(f"\n→ Testing search with query: '{query}'")
        
        # Try with different parameters
        logger.info("  Attempting: ddgs.text(query, max_results=5, region='wt-wt')")
        results = ddgs.text(query, max_results=5, region='wt-wt')
        
        # Convert generator to list
        results_list = list(results)
        logger.info(f"✓ Got {len(results_list)} results")
        
        if results_list:
            logger.info("\n  Results (first 3):")
            for idx, result in enumerate(results_list[:3], 1):
                logger.info(f"    [{idx}] {result.get('title', 'N/A')[:60]}")
                logger.info(f"        URL: {result.get('href', 'N/A')}")
                logger.info(f"        Body: {result.get('body', 'N/A')[:80]}")
        else:
            logger.warning("✗ No results returned!")
            return False
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ Failed to import DDGS: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Error in DDGS test: {e}", exc_info=True)
        return False

def test_web_search_agent():
    """Test the WebSearchAgent directly."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: WebSearchAgent.search()")
    logger.info("="*80)
    
    try:
        from task_manager.sub_agents.web_search_agent import WebSearchAgent
        
        agent = WebSearchAgent()
        logger.info(f"✓ WebSearchAgent created (search_lib={agent.search_lib})")
        
        # Test simple search
        query = "supply chain trends CPG 2026"
        logger.info(f"\n→ Testing agent.search('{query}', max_results=5)")
        
        result = agent.search(query, max_results=5, language='en')
        
        logger.info(f"✓ Search returned (success={result.get('success')})")
        logger.info(f"  - results_count: {result.get('results_count', 0)}")
        logger.info(f"  - error: {result.get('error', 'None')}")
        
        if result.get('success'):
            results = result.get('results', [])
            logger.info(f"\n  Got {len(results)} results:")
            for idx, r in enumerate(results[:3], 1):
                logger.info(f"    [{idx}] {r.get('title', 'N/A')[:60]}")
                logger.info(f"        URL: {r.get('url', 'N/A')}")
        else:
            logger.warning(f"✗ Search failed: {result.get('error')}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error testing WebSearchAgent: {e}", exc_info=True)
        return False

def test_deep_search():
    """Test the deep_search method."""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: WebSearchAgent.deep_search()")
    logger.info("="*80)
    
    try:
        from task_manager.sub_agents.web_search_agent import WebSearchAgent
        
        agent = WebSearchAgent()
        logger.info(f"✓ WebSearchAgent created")
        
        # Test deep search
        query = "supply chain trends CPG"
        logger.info(f"\n→ Testing agent.deep_search('{query}', max_results=2, max_depth=1)")
        
        result = agent.deep_search(query, max_results=2, max_depth=1)
        
        logger.info(f"✓ Deep search returned (success={result.get('success')})")
        logger.info(f"  - pages_visited: {result.get('pages_visited', 0)}")
        logger.info(f"  - extracted_data count: {len(result.get('extracted_data', []))}")
        logger.info(f"  - findings count: {len(result.get('findings', []))}")
        logger.info(f"  - error: {result.get('error', 'None')}")
        
        if not result.get('success'):
            logger.warning(f"✗ Deep search failed: {result.get('error')}")
            logger.info(f"  Details: {result}")
            return False
        
        if not result.get('extracted_data'):
            logger.warning("✗ No data extracted!")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error testing deep_search: {e}", exc_info=True)
        return False

def main():
    """Run all diagnostics."""
    logger.info("WEBSEARCH AGENT DIAGNOSTIC TESTS")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info("")
    
    results = []
    
    # Test 1: Direct DDGS
    results.append(("DDGS Direct", test_ddgs_direct()))
    
    # Test 2: WebSearchAgent
    results.append(("WebSearchAgent", test_web_search_agent()))
    
    # Test 3: Deep Search
    results.append(("Deep Search", test_deep_search()))
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {status}: {name}")
    
    all_passed = all(p for _, p in results)
    logger.info("="*80)
    
    if all_passed:
        logger.info("✓ All tests passed!")
        return 0
    else:
        logger.warning("✗ Some tests failed. Check logs above for details.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
