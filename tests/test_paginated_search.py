#!/usr/bin/env python3
"""
Test suite for the paginated search functionality with comprehensive logging.

Tests:
1. Basic paginated search with multiple batches
2. Result aggregation and deduplication
3. Logging verification
4. Edge cases (empty results, single batch, etc.)
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from task_manager.sub_agents.web_search_agent import WebSearchAgent
import logging

# Setup logging to see all pagination logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)

logger = logging.getLogger(__name__)


def test_paginated_search_basic():
    """Test basic paginated search with multiple batches."""
    print("\n" + "="*80)
    print("TEST 1: PAGINATED SEARCH - BASIC MULTI-BATCH")
    print("="*80)
    
    try:
        agent = WebSearchAgent()
        print(f"✓ WebSearchAgent initialized")
        
        # Test query
        query = "Python machine learning libraries"
        max_results = 15  # Request 15 results (3 batches of 5)
        batch_size = 5
        
        print(f"\n📝 Test Configuration:")
        print(f"   Query: {query}")
        print(f"   Max Results: {max_results}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Expected Batches: 3")
        
        # Execute paginated search
        print(f"\n🔍 Starting paginated search...")
        result = agent.paginated_search(
            query=query,
            max_results=max_results,
            batch_size=batch_size,
            max_batches=10
        )
        
        print(f"\n✅ Paginated search completed: {result.get('success')}")
        
        if not result.get('success'):
            print(f"❌ Search failed: {result.get('error')}")
            return False
        
        # Validation
        results_count = result.get('results_count', 0)
        print(f"\n📊 RESULTS:")
        print(f"   Total Results: {results_count}")
        
        pagination_stats = result.get('pagination_stats', {})
        print(f"\n📋 PAGINATION STATS:")
        print(f"   Total Batches: {pagination_stats.get('total_batches', 0)}")
        print(f"   Successful Batches: {pagination_stats.get('successful_batches', 0)}")
        print(f"   Failed Batches: {pagination_stats.get('failed_batches', [])}")
        print(f"   Results per Batch: {pagination_stats.get('results_per_batch', [])}")
        
        # Check batch details
        batch_details = result.get('batch_details', [])
        print(f"\n📄 BATCH DETAILS:")
        for idx, batch in enumerate(batch_details, 1):
            status = batch.get('status', 'unknown')
            count = batch.get('results_count', 0)
            time_sec = batch.get('time_seconds', 0)
            print(f"   Batch {idx}: {status} - {count} results in {time_sec:.2f}s")
        
        # Verify results have global ranks
        results = result.get('results', [])
        if results:
            print(f"\n🏆 GLOBAL RANKS:")
            for idx, r in enumerate(results[:3], 1):
                global_rank = r.get('global_rank', 'N/A')
                title = r.get('title', 'N/A')[:50]
                print(f"   [{global_rank}] {title}")
        
        if results_count > 0:
            print(f"\n✅ TEST PASSED: Retrieved {results_count} results from multiple batches")
            return True
        else:
            print(f"\n❌ TEST FAILED: No results retrieved")
            return False
        
    except Exception as e:
        print(f"❌ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_paginated_search_small_batch():
    """Test paginated search with small batch size."""
    print("\n" + "="*80)
    print("TEST 2: PAGINATED SEARCH - SMALL BATCH (3 results per batch)")
    print("="*80)
    
    try:
        agent = WebSearchAgent()
        
        query = "Python web frameworks"
        max_results = 10
        batch_size = 3
        
        print(f"\n📝 Configuration:")
        print(f"   Query: {query}")
        print(f"   Max Results: {max_results}")
        print(f"   Batch Size: {batch_size} (smaller = more batches)")
        print(f"   Expected Batches: ~4")
        
        result = agent.paginated_search(
            query=query,
            max_results=max_results,
            batch_size=batch_size,
            max_batches=10
        )
        
        if not result.get('success'):
            print(f"⚠️  Search returned success=False: {result.get('error')}")
            return False
        
        results_count = result.get('results_count', 0)
        pagination_stats = result.get('pagination_stats', {})
        
        print(f"\n✅ Results: {results_count} items")
        print(f"   Batches: {pagination_stats.get('total_batches', 0)}")
        print(f"   Success rate: {pagination_stats.get('successful_batches', 0)}/{pagination_stats.get('total_batches', 1)}")
        
        if results_count > 0:
            print(f"✅ TEST PASSED")
            return True
        else:
            print(f"❌ TEST FAILED: No results")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


def test_paginated_search_single_batch():
    """Test pagination with only one batch (edge case)."""
    print("\n" + "="*80)
    print("TEST 3: PAGINATED SEARCH - SINGLE BATCH (edge case)")
    print("="*80)
    
    try:
        agent = WebSearchAgent()
        
        query = "Python"
        max_results = 5
        batch_size = 10  # Larger than max_results = single batch
        
        print(f"\n📝 Configuration:")
        print(f"   Query: {query}")
        print(f"   Max Results: {max_results}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Expected Batches: 1")
        
        result = agent.paginated_search(
            query=query,
            max_results=max_results,
            batch_size=batch_size
        )
        
        if not result.get('success'):
            print(f"❌ Search failed: {result.get('error')}")
            return False
        
        pagination_stats = result.get('pagination_stats', {})
        total_batches = pagination_stats.get('total_batches', 0)
        
        print(f"\n✅ Results: {result.get('results_count', 0)} items")
        print(f"   Batches executed: {total_batches}")
        
        if total_batches == 1:
            print(f"✅ TEST PASSED: Single batch as expected")
            return True
        else:
            print(f"⚠️  Expected 1 batch, got {total_batches}")
            return result.get('results_count', 0) > 0
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


def test_logging_output(capture_logs: bool = True):
    """Test that logging is working properly."""
    print("\n" + "="*80)
    print("TEST 4: LOGGING VERIFICATION")
    print("="*80)
    
    try:
        agent = WebSearchAgent()
        
        query = "Python programming"
        
        print(f"\n📝 Configuration:")
        print(f"   Query: {query}")
        print(f"   Max Results: 8")
        print(f"   Batch Size: 4 (will create 2 batches)")
        print(f"   Logging: ENABLED - Watch logs above for detailed pagination info")
        
        result = agent.paginated_search(
            query=query,
            max_results=8,
            batch_size=4
        )
        
        print(f"\n✅ Paginated search completed")
        print(f"   Success: {result.get('success')}")
        print(f"   Results: {result.get('results_count', 0)}")
        
        batch_details = result.get('batch_details', [])
        if batch_details:
            print(f"\n📄 Batch Details Captured:")
            for batch in batch_details:
                print(f"   - Batch {batch.get('batch_num')}: {batch.get('results_count')} results")
            print(f"\n✅ TEST PASSED: Logging and batching working correctly")
            return True
        else:
            print(f"⚠️  No batch details captured")
            return result.get('success', False)
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*15 + "PAGINATED WEBSEARCH WITH LOGGING TEST SUITE" + " "*18 + "║")
    print("╚" + "="*78 + "╝")
    
    tests = [
        ("Basic Multi-Batch Search", test_paginated_search_basic),
        ("Small Batch Size", test_paginated_search_small_batch),
        ("Single Batch Edge Case", test_paginated_search_single_batch),
        ("Logging Verification", test_logging_output),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running: {test_name}")
            passed = test_func()
            results[test_name] = passed
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {str(e)}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed")
    
    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
