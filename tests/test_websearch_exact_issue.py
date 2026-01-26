#!/usr/bin/env python3
"""
Specific test to reproduce the exact issue from the user's error log.
Tests the exact query: "1.1.1: Research and identify latest supply chain trends in CPG industry"
"""

import logging
import sys
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_exact_query():
    """Test with the exact query from the error log."""
    print("\n" + "="*80)
    print("REPRODUCING EXACT ISSUE FROM ERROR LOG")
    print("="*80)
    
    try:
        from task_manager.sub_agents.web_search_agent import WebSearchAgent
        
        agent = WebSearchAgent()
        print(f"[OK] WebSearchAgent initialized")
        print(f"  - Default backend: {agent.default_backend}")
        print(f"  - Default region: {agent.default_region}")
        print(f"  - Search library: {agent.search_lib}")
        
        # Use the exact query from error log
        exact_query = "1.1.1: Research and identify latest supply chain trends in CPG industry"
        
        print(f"\nSearching with exact query:")
        print(f"  '{exact_query}'")
        
        result = agent.search(
            query=exact_query,
            max_results=5
        )
        
        print(f"\n[SEARCH RESULT]")
        print(f"  - success: {result.get('success')}")
        print(f"  - results_count: {result.get('results_count')}")
        print(f"  - has 'results' key: {'results' in result}")
        
        if 'results' in result:
            results = result.get('results', [])
            print(f"  - results length: {len(results)}")
            print(f"  - results is empty: {len(results) == 0}")
            
            if len(results) > 0:
                print(f"\n  First result:")
                print(f"    - title: {results[0].get('title', 'N/A')[:50]}")
                print(f"    - url: {results[0].get('url', 'N/A')[:80]}")
                print(f"    - snippet: {results[0].get('snippet', 'N/A')[:80]}")
            else:
                print(f"\n  [ERROR] Results list is EMPTY!")
                if 'error' in result:
                    print(f"  Error message: {result.get('error')}")
        
        # Test the exact task scenario
        print(f"\n" + "="*80)
        print("TESTING WITH STANDARDIZED EXECUTE_TASK (AS CALLED BY AGENT.PY)")
        print("="*80)
        
        from task_manager.models import AgentExecutionRequest
        
        request: AgentExecutionRequest = {
            "task_id": "test_task_001",
            "task_description": exact_query,
            "task_type": "atomic",
            "operation": "search",
            "parameters": {
                "query": exact_query,
                "max_results": 5
            },
            "input_data": {},
            "temp_folder": str(Path(__file__).parent.parent / "temp_folder"),
            "output_folder": str(Path(__file__).parent.parent / "output_folder"),
            "cache_enabled": False,
            "blackboard": [],
            "relevant_entries": [],
            "max_retries": 3
        }
        
        print(f"Calling execute_task with standardized request...")
        response = agent.execute_task(request)
        
        print(f"\n[RESPONSE STRUCTURE]")
        print(f"  - status: {response.get('status')}")
        print(f"  - success: {response.get('success')}")
        
        if 'result' in response:
            result_dict = response['result']
            print(f"\n[RESULT DICT]")
            print(f"  - keys: {list(result_dict.keys())}")
            print(f"  - results_count: {result_dict.get('results_count')}")
            print(f"  - results length: {len(result_dict.get('results', []))}")
            
            # Check if results are empty
            if result_dict.get('results_count', 0) > 0:
                results = result_dict.get('results', [])
                if len(results) == 0:
                    print(f"\n  [ERROR] results_count says {result_dict.get('results_count')}, but results list is empty!")
                else:
                    print(f"\n  [OK] Got {len(results)} results")
            else:
                print(f"\n  [ERROR] results_count is 0 or missing!")
                
        return True
        
    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = test_exact_query()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
