#!/usr/bin/env python3
"""
Diagnostic test to identify why WebSearchAgent is returning empty results.
Tests the actual search library and backend configuration.
"""

import logging
import sys
import os
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_ddgs_installation():
    """Test if DDGS is properly installed."""
    print("\n" + "="*80)
    print("TEST 1: DDGS Library Installation")
    print("="*80)
    
    try:
        from ddgs import DDGS
        print("[OK] DDGS library imported successfully")
        
        # Try to instantiate
        ddgs = DDGS()
        print("[OK] DDGS instance created successfully")
        
        # Try a simple search
        print("\nAttempting a simple search...")
        results = list(ddgs.text("test", max_results=5))
        print(f"[OK] DDGS search executed successfully")
        print(f"  Results returned: {len(results)}")
        
        if results:
            print(f"  First result: {results[0]}")
            return True
        else:
            print("  [WARN] No results returned from DDGS")
            return False
            
    except ImportError as e:
        print(f"[ERROR] Failed to import DDGS: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Error during DDGS test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_web_search_agent_search_method():
    """Test WebSearchAgent.search() method directly."""
    print("\n" + "="*80)
    print("TEST 2: WebSearchAgent.search() Method")
    print("="*80)
    
    try:
        from task_manager.sub_agents.web_search_agent import WebSearchAgent
        
        agent = WebSearchAgent()
        print(f"[OK] WebSearchAgent initialized")
        print(f"  - Default backend: {agent.default_backend}")
        print(f"  - Default region: {agent.default_region}")
        print(f"  - Search library: {agent.search_lib}")
        print(f"  - Playwright available: {agent.playwright_available}")
        
        if not agent.search_lib:
            print("[ERROR] No search library detected in WebSearchAgent")
            return False
        
        # Test search
        print("\nAttempting search with query: 'Python programming'")
        result = agent.search(
            query="Python programming",
            max_results=5
        )
        
        print(f"\nSearch result:")
        print(f"  - success: {result.get('success')}")
        print(f"  - query: {result.get('query')}")
        print(f"  - results_count: {result.get('results_count')}")
        print(f"  - has 'results' key: {'results' in result}")
        print(f"  - results type: {type(result.get('results'))}")
        
        if 'results' in result:
            print(f"  - results length: {len(result.get('results', []))}")
            if result.get('results'):
                print(f"  - First result: {result['results'][0]}")
            else:
                print(f"  - results array is empty")
        
        if 'error' in result:
            print(f"  - error: {result.get('error')}")
        
        if result.get('success') and result.get('results_count', 0) > 0:
            return True
        else:
            print(f"[ERROR] Search did not return successful results")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error testing WebSearchAgent.search(): {e}")
        import traceback
        traceback.print_exc()
        return False

def test_web_search_agent_execute_task():
    """Test WebSearchAgent.execute_task() with standardized request."""
    print("\n" + "="*80)
    print("TEST 3: WebSearchAgent.execute_task() - Standardized Format")
    print("="*80)
    
    try:
        from task_manager.sub_agents.web_search_agent import WebSearchAgent
        from task_manager.models import AgentExecutionRequest
        
        agent = WebSearchAgent()
        print(f"[OK] WebSearchAgent initialized")
        
        # Create standardized request
        request: AgentExecutionRequest = {
            "task_id": "test_task_001",
            "task_description": "Research and identify latest supply chain trends in CPG industry",
            "task_type": "atomic",
            "operation": "search",
            "parameters": {
                "query": "latest supply chain trends CPG industry"
            },
            "input_data": {},
            "temp_folder": str(Path(__file__).parent.parent / "temp_folder"),
            "output_folder": str(Path(__file__).parent.parent / "output_folder"),
            "cache_enabled": False,
            "blackboard": [],
            "relevant_entries": [],
            "max_retries": 3
        }
        
        print(f"\nCalling execute_task with standardized request...")
        print(f"  - task_id: {request['task_id']}")
        print(f"  - operation: {request['operation']}")
        print(f"  - query: {request['parameters']['query']}")
        
        response = agent.execute_task(request)
        
        print(f"\nResponse received:")
        print(f"  - type: {type(response)}")
        print(f"  - keys: {list(response.keys()) if isinstance(response, dict) else 'N/A'}")
        print(f"  - success: {response.get('success')}")
        print(f"  - results_count: {response.get('results_count')}")
        
        if 'result' in response:
            result = response['result']
            print(f"  - result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
            print(f"  - result success: {result.get('success')}")
            print(f"  - result results_count: {result.get('results_count')}")
            if 'results' in result:
                print(f"  - result has 'results': {len(result.get('results', []))} items")
            if 'error' in result:
                print(f"  - result error: {result.get('error')}")
        
        if response.get('success') and response.get('results_count', 0) > 0:
            return True
        else:
            print(f"[ERROR] execute_task did not return successful results")
            if response.get('error'):
                print(f"  Error: {response.get('error')}")
            if 'result' in response and response['result'].get('error'):
                print(f"  Result error: {response['result'].get('error')}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error testing WebSearchAgent.execute_task(): {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backend_configuration():
    """Test backend configuration from environment."""
    print("\n" + "="*80)
    print("TEST 4: Backend Configuration")
    print("="*80)
    
    print(f"Environment variables:")
    print(f"  - WEBSEARCH_BACKEND: {os.getenv('WEBSEARCH_BACKEND', 'NOT SET')}")
    print(f"  - WEBSEARCH_REGION: {os.getenv('WEBSEARCH_REGION', 'NOT SET')}")
    print(f"  - WEBSEARCH_TIMEOUT: {os.getenv('WEBSEARCH_TIMEOUT', 'NOT SET')}")
    
    # Test direct DDGS with different backends
    print(f"\nTesting DDGS with different backends:")
    
    backends = ['api', 'html', 'lite']
    
    for backend in backends:
        try:
            from ddgs import DDGS
            print(f"\n  Testing backend: {backend}")
            ddgs = DDGS()
            results = list(ddgs.text("test query", max_results=2, backend=backend))
            print(f"    [OK] Backend '{backend}' returned {len(results)} results")
        except Exception as e:
            print(f"    [ERROR] Backend '{backend}' failed: {e}")
    
    return True

def main():
    """Run all diagnostic tests."""
    print("\n" + "="*80)
    print("WEB SEARCH AGENT DIAGNOSTIC TEST SUITE")
    print("="*80)
    
    results = {
        "DDGS Installation": test_ddgs_installation(),
        "WebSearchAgent.search()": test_web_search_agent_search_method(),
        "WebSearchAgent.execute_task()": test_web_search_agent_execute_task(),
        "Backend Configuration": test_backend_configuration(),
    }
    
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*80)
    if all_passed:
        print("[OK] ALL TESTS PASSED - Web search appears to be working correctly")
    else:
        print("[ERROR] SOME TESTS FAILED - Web search has issues that need fixing")
    print("="*80)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
