#!/usr/bin/env python3
"""
Test to verify results are properly passed through the standardized response format.
"""

import logging
import sys
import json
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_results_flow():
    """Test that results flow correctly from WebSearchAgent through standardized response."""
    print("\n" + "="*80)
    print("TESTING RESULTS FLOW THROUGH STANDARDIZED RESPONSE")
    print("="*80)
    
    try:
        from task_manager.sub_agents.web_search_agent import WebSearchAgent
        from task_manager.models import AgentExecutionRequest
        
        agent = WebSearchAgent()
        print(f"[OK] WebSearchAgent initialized")
        
        # Create standardized request  
        request: AgentExecutionRequest = {
            "task_id": "flow_test_001",
            "task_description": "Python programming",
            "task_type": "atomic",
            "operation": "search",
            "parameters": {
                "query": "Python programming",
                "max_results": 3
            },
            "input_data": {},
            "temp_folder": str(Path(__file__).parent.parent / "temp_folder"),
            "output_folder": str(Path(__file__).parent.parent / "output_folder"),
            "cache_enabled": False,
            "blackboard": [],
            "relevant_entries": [],
            "max_retries": 3
        }
        
        print(f"\nStep 1: Calling execute_task()")
        response = agent.execute_task(request)
        
        print(f"\nStep 2: Checking response structure")
        print(f"  - response['success']: {response.get('success')}")
        print(f"  - response['status']: {response.get('status')}")
        print(f"  - 'result' in response: {'result' in response}")
        
        if 'result' not in response:
            print(f"  [ERROR] 'result' key missing from response!")
            print(f"  Response keys: {list(response.keys())}")
            return False
        
        result = response['result']
        print(f"\nStep 3: Inspecting result dict")
        print(f"  - keys: {list(result.keys())}")
        print(f"  - 'results' in result: {'results' in result}")
        print(f"  - 'results_count' in result: {'results_count' in result}")
        print(f"  - results_count value: {result.get('results_count')}")
        
        if 'results' not in result:
            print(f"  [ERROR] 'results' key missing from result dict!")
            return False
        
        results_list = result.get('results', [])
        print(f"\nStep 4: Checking results list")
        print(f"  - len(results): {len(results_list)}")
        print(f"  - results is empty: {len(results_list) == 0}")
        
        if len(results_list) == 0:
            print(f"  [ERROR] Results list is EMPTY!")
            return False
        
        if len(results_list) > 0:
            first_result = results_list[0]
            print(f"\nStep 5: Inspecting first result")
            print(f"  - type: {type(first_result)}")
            print(f"  - keys: {list(first_result.keys()) if isinstance(first_result, dict) else 'N/A'}")
            if isinstance(first_result, dict):
                print(f"  - title: {first_result.get('title', 'N/A')[:50]}")
                print(f"  - url: {first_result.get('url', 'N/A')[:60]}")
                print(f"  - snippet: {first_result.get('snippet', 'N/A')[:60]}")
        
        # Now test the normalization logic from agent.py
        print(f"\nStep 6: Simulating agent.py normalization logic")
        print(f"  Simulating what agent.py does:")
        
        operation = 'search'
        result_data = {
            'success': response['success'],
            'summary': result.get('summary', ''),
            'results_count': result.get('results_count', 0),
            **result  # Merge all result fields
        }
        
        print(f"    - result_data keys after merge: {list(result_data.keys())}")
        print(f"    - result_data['results_count']: {result_data.get('results_count')}")
        print(f"    - len(result_data['results']): {len(result_data.get('results', []))}")
        
        # Apply normalization
        if operation == 'search' and 'results' in result_data and 'extracted_data' not in result_data:
            search_results = result_data.get('results', [])
            print(f"\n  Applying search normalization:")
            print(f"    - search_results count: {len(search_results)}")
            
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
            
            print(f"    - extracted_data items: {len(result_data['extracted_data'])}")
            print(f"    - findings items: {len(result_data['findings'])}")
            
            if len(result_data['extracted_data']) == 0:
                print(f"    [ERROR] extracted_data is EMPTY after normalization!")
                return False
            else:
                print(f"    [OK] Normalization successful, extracted {len(result_data['extracted_data'])} items")
        
        print(f"\n" + "="*80)
        print(f"[OK] FULL FLOW TEST PASSED - Results properly preserved through all layers")
        print(f"="*80)
        return True
        
    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = test_results_flow()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
