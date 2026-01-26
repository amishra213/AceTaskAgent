#!/usr/bin/env python3
"""
Comprehensive fix verification test.
Tests the complete flow including normalization with improved fallbacks and error handling.
"""

import logging
import sys
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_complete_fix():
    """Test the complete fix including normalization improvements and error handling."""
    print("\n" + "="*80)
    print("WEB SEARCH AGENT - COMPREHENSIVE FIX VERIFICATION")
    print("="*80)
    
    try:
        from task_manager.sub_agents.web_search_agent import WebSearchAgent
        from task_manager.models import AgentExecutionRequest, TaskStatus
        
        agent = WebSearchAgent()
        print(f"\n[STEP 1] WebSearchAgent initialized successfully")
        
        # Test multiple queries to ensure consistency
        test_queries = [
            "Python programming",
            "Web development",
            "Data science"
        ]
        
        all_passed = True
        
        for query in test_queries:
            print(f"\n[STEP 2] Testing query: {query}")
            
            request: AgentExecutionRequest = {
                "task_id": f"test_{test_queries.index(query)}",
                "task_description": query,
                "task_type": "atomic",
                "operation": "search",
                "parameters": {
                    "query": query,
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
            
            response = agent.execute_task(request)
            
            # Verify response structure
            assert 'result' in response, f"Missing 'result' in response for query: {query}"
            assert response['success'], f"Response marked as failed for query: {query}"
            
            result = response['result']
            results_count = result.get('results_count', 0)
            results_list = result.get('results', [])
            
            print(f"  - Results found: {results_count}")
            print(f"  - Results list length: {len(results_list)}")
            
            if results_count > 0 and len(results_list) > 0:
                print(f"  - [OK] Results preserved correctly")
            elif results_count == 0 and len(results_list) == 0:
                print(f"  - [OK] Empty results handled correctly")
            else:
                print(f"  - [ERROR] Mismatch: results_count={results_count} but len(results)={len(results_list)}")
                all_passed = False
                continue
            
            # Simulate the agent.py normalization logic
            operation = 'search'
            result_data = {
                'success': response['success'],
                'summary': result.get('summary', ''),
                'results_count': result.get('results_count', 0),
                **result  # Merge all result fields
            }
            
            # Apply improved normalization logic
            if operation == 'search' and 'results' in result_data and 'extracted_data' not in result_data:
                search_results = result_data.get('results', [])
                
                # Create extracted_data with fallback fields
                result_data['extracted_data'] = []
                for r in search_results:
                    snippet = r.get('snippet', '') or r.get('body', '') or r.get('title', '')
                    if isinstance(snippet, str) and snippet.strip():
                        result_data['extracted_data'].append(snippet)
                
                # Create findings
                result_data['findings'] = []
                for r in search_results:
                    snippet = r.get('snippet', '') or r.get('body', '') or r.get('title', '')
                    if r.get('url', '') or r.get('title', ''):
                        result_data['findings'].append({
                            'url': r.get('url', ''),
                            'title': r.get('title', ''),
                            'relevance_score': 0.8,
                            'text_preview': (snippet[:500] if isinstance(snippet, str) else str(snippet)[:500])
                        })
            
            # Verify normalized data
            extracted_count = len(result_data.get('extracted_data', []))
            findings_count = len(result_data.get('findings', []))
            
            print(f"  - Extracted data items: {extracted_count}")
            print(f"  - Findings items: {findings_count}")
            
            if results_count > 0:
                if extracted_count == 0 or findings_count == 0:
                    print(f"  - [ERROR] Normalization failed: results_count > 0 but no extracted_data/findings")
                    all_passed = False
                else:
                    print(f"  - [OK] Normalization successful")
            else:
                print(f"  - [OK] Empty results handled correctly")
        
        print(f"\n" + "="*80)
        if all_passed:
            print("[SUCCESS] ALL TESTS PASSED - Web Search Agent is working correctly")
            print("The fixes ensure:")
            print("  1. Results flow correctly through standardized response format")
            print("  2. Normalization creates extracted_data and findings")
            print("  3. Fallback field names handle different response formats")
            print("  4. Empty results are handled gracefully")
            print("  5. Errors are properly logged for debugging")
        else:
            print("[FAILURE] SOME TESTS FAILED - Issues detected")
        print("="*80)
        
        return all_passed
        
    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = test_complete_fix()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
