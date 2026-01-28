#!/usr/bin/env python3
"""
Real-world usage examples for the WebSearch Agent's paginated_search() method.

This file demonstrates practical patterns for:
1. Basic pagination
2. Handling different result set sizes
3. Processing paginated results
4. Error handling
5. Integration with other agents
"""

import sys
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from task_manager.sub_agents.web_search_agent import WebSearchAgent


# ============================================================================
# EXAMPLE 1: Basic Pagination - Get 15 Results
# ============================================================================

def example_1_basic_pagination():
    """
    Retrieve 15 results from a web search with default batch size of 5.
    This demonstrates the simplest usage pattern.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Pagination (15 results)")
    print("="*80 + "\n")
    
    agent = WebSearchAgent()
    
    result = agent.paginated_search(
        query="Python machine learning",
        max_results=15,
        batch_size=5
    )
    
    print(f"✓ Search completed successfully")
    print(f"  Results retrieved: {result['results_count']}")
    print(f"  Batches executed: {result['pagination_stats']['total_batches']}")
    print(f"  Success rate: {result['pagination_stats']['successful_batches']}/{result['pagination_stats']['total_batches']}")
    
    # Display results
    print(f"\nTop 5 Results:")
    for r in result['results'][:5]:
        print(f"  [{r['global_rank']}] {r['title'][:60]}")
        print(f"      {r['url'][:70]}")


# ============================================================================
# EXAMPLE 2: Small Batch Size - Fine-Grained Control
# ============================================================================

def example_2_small_batch_size():
    """
    Use a small batch size (2-3 results per batch) for fine-grained control.
    Useful when you want to process results incrementally or monitor progress.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Small Batch Size (3 results per batch)")
    print("="*80 + "\n")
    
    agent = WebSearchAgent()
    
    result = agent.paginated_search(
        query="Web development frameworks",
        max_results=9,
        batch_size=3,  # Smaller batches
        max_batches=5
    )
    
    print(f"✓ Search completed")
    print(f"  Total results: {result['results_count']}")
    print(f"  Batches: {result['pagination_stats']['total_batches']}")
    print(f"  Results per batch: {result['pagination_stats']['results_per_batch']}")
    
    # Group results by batch
    batch_num = 1
    for idx, r in enumerate(result['results'], 1):
        if (idx - 1) % 3 == 0:
            print(f"\nBatch {batch_num}:")
            batch_num += 1
        print(f"  [{r['global_rank']}] {r['title'][:50]}")


# ============================================================================
# EXAMPLE 3: Large Result Set - Comprehensive Research
# ============================================================================

def example_3_large_result_set():
    """
    Retrieve a large number of results (50+) for comprehensive research.
    Demonstrates how to aggregate and process many results.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Large Result Set (50 results)")
    print("="*80 + "\n")
    
    agent = WebSearchAgent()
    
    result = agent.paginated_search(
        query="Artificial intelligence applications",
        max_results=50,
        batch_size=10,  # Larger batches for efficiency
        max_batches=5
    )
    
    print(f"✓ Search completed")
    print(f"  Results retrieved: {result['results_count']}")
    print(f"  Batches executed: {result['pagination_stats']['total_batches']}")
    
    # Analyze results
    urls = set()
    sources = {}
    
    for r in result['results']:
        # Track unique URLs (for deduplication)
        urls.add(r['url'])
        
        # Count by source
        source = r.get('source', 'Unknown')
        sources[source] = sources.get(source, 0) + 1
    
    print(f"\nAnalysis:")
    print(f"  Unique URLs: {len(urls)}")
    print(f"  Total results: {len(result['results'])}")
    print(f"  Sources breakdown:")
    for source, count in sorted(sources.items(), key=lambda x: -x[1]):
        percentage = (count / len(result['results'])) * 100
        print(f"    - {source}: {count} ({percentage:.1f}%)")


# ============================================================================
# EXAMPLE 4: Error Handling & Recovery
# ============================================================================

def example_4_error_handling():
    """
    Demonstrate error handling for failed batches and retry logic.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Error Handling & Recovery")
    print("="*80 + "\n")
    
    agent = WebSearchAgent()
    
    # Try with a query that might have issues
    result = agent.paginated_search(
        query="very specific technical query",
        max_results=10,
        batch_size=3,
        retry_on_empty=True
    )
    
    if result['success']:
        stats = result['pagination_stats']
        print(f"✓ Search succeeded (graceful degradation)")
        print(f"  Results: {result['results_count']}")
        print(f"  Successful batches: {stats['successful_batches']}/{stats['total_batches']}")
        
        if stats['failed_batches']:
            print(f"  ⚠️  Failed batches: {stats['failed_batches']} (recovered from)")
        else:
            print(f"  ✓ All batches successful")
    else:
        print(f"✗ Search failed: {result.get('error', 'Unknown error')}")
        print(f"  Consider adjusting query or parameters")


# ============================================================================
# EXAMPLE 5: Using Different Search Backends
# ============================================================================

def example_5_different_backends():
    """
    Use different search backends for various geographic regions and engines.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Different Search Backends & Regions")
    print("="*80 + "\n")
    
    agent = WebSearchAgent()
    
    # Configuration options
    backends = [
        {"backend": "auto", "region": "wt-wt", "label": "Worldwide (Auto)"},
        {"backend": "auto", "region": "us", "label": "United States"},
        {"backend": "auto", "region": "uk", "label": "United Kingdom"},
    ]
    
    query = "Python programming"
    
    for config in backends[:1]:  # Run first one to avoid too many API calls
        print(f"\n{config['label']}:")
        print(f"  Backend: {config['backend']}, Region: {config['region']}")
        
        result = agent.paginated_search(
            query=query,
            max_results=5,
            batch_size=5,
            backend=config['backend'],
            language=config['region']
        )
        
        print(f"  Results: {result['results_count']}")
        if result['results']:
            print(f"  Top result: {result['results'][0]['title'][:60]}")


# ============================================================================
# EXAMPLE 6: Export Results to JSON
# ============================================================================

def example_6_export_to_json():
    """
    Retrieve paginated results and export to JSON for further processing.
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Export Results to JSON")
    print("="*80 + "\n")
    
    agent = WebSearchAgent()
    
    result = agent.paginated_search(
        query="Data science tools",
        max_results=10,
        batch_size=5
    )
    
    # Prepare data for export
    export_data = {
        "query": "Data science tools",
        "timestamp": str(result.get('timestamp', 'N/A')),
        "pagination": result['pagination_stats'],
        "results": result['results']
    }
    
    # Save to file
    output_file = Path(__file__).parent / "paginated_search_results.json"
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"✓ Results exported to {output_file}")
    print(f"  Results: {len(export_data['results'])}")
    print(f"  File size: {output_file.stat().st_size} bytes")


# ============================================================================
# EXAMPLE 7: Processing Results in Chunks
# ============================================================================

def example_7_process_in_chunks():
    """
    Retrieve paginated results and process them in chunks for batch operations.
    """
    print("\n" + "="*80)
    print("EXAMPLE 7: Process Results in Chunks")
    print("="*80 + "\n")
    
    agent = WebSearchAgent()
    
    result = agent.paginated_search(
        query="Software libraries",
        max_results=12,
        batch_size=3
    )
    
    # Process results in chunks of 4
    chunk_size = 4
    results = result['results']
    
    print(f"Processing {len(results)} results in chunks of {chunk_size}:\n")
    
    for chunk_idx in range(0, len(results), chunk_size):
        chunk = results[chunk_idx:chunk_idx + chunk_size]
        print(f"Chunk {chunk_idx // chunk_size + 1}:")
        for r in chunk:
            print(f"  - [{r['global_rank']}] {r['title'][:45]}")
        print()


# ============================================================================
# EXAMPLE 8: Combining with Deep Search
# ============================================================================

def example_8_combine_with_deep_search():
    """
    Use paginated_search to get many results, then deep_search for detailed content.
    """
    print("\n" + "="*80)
    print("EXAMPLE 8: Combine Paginated with Deep Search")
    print("="*80 + "\n")
    
    agent = WebSearchAgent()
    
    # First: Get initial results via pagination
    print("Step 1: Get paginated results (overview)")
    paginated = agent.paginated_search(
        query="Python web frameworks",
        max_results=10,
        batch_size=5
    )
    
    print(f"  Retrieved {paginated['results_count']} results")
    
    # Second: Deep search on top result for detailed content
    if paginated['results']:
        top_result = paginated['results'][0]
        print(f"\nStep 2: Deep search top result: {top_result['title'][:50]}")
        
        deep = agent.deep_search(
            query=top_result['title'],
            max_depth=2
        )
        
        print(f"  Deep search found {len(deep.get('results', []))} detailed results")
        print(f"  Extracted {deep.get('items_extracted', 0)} content items")


# ============================================================================
# EXAMPLE 9: Monitoring Pagination Progress
# ============================================================================

def example_9_monitor_progress():
    """
    Monitor pagination progress with real-time feedback.
    """
    print("\n" + "="*80)
    print("EXAMPLE 9: Monitor Pagination Progress")
    print("="*80 + "\n")
    
    agent = WebSearchAgent()
    
    result = agent.paginated_search(
        query="Blockchain applications",
        max_results=15,
        batch_size=5,
        max_batches=5
    )
    
    # Show progress
    stats = result['pagination_stats']
    print(f"✓ Pagination Complete\n")
    print(f"Target results: 15")
    print(f"Actual results: {result['results_count']}")
    print(f"Coverage: {(result['results_count'] / 15) * 100:.1f}%\n")
    
    print(f"Batch Performance:")
    times = []
    for idx, batch in enumerate(result['batch_details'], 1):
        time_sec = batch.get('time_seconds', 0)
        count = batch.get('results_count', 0)
        rate = count / time_sec if time_sec > 0 else 0
        times.append(time_sec)
        
        status_icon = "✓" if batch['status'] == 'success' else "✗"
        print(f"  {status_icon} Batch {idx}: {count} results in {time_sec:.2f}s ({rate:.1f} results/s)")
    
    avg_time = sum(times) / len(times) if times else 0
    total_time = sum(times)
    print(f"\nTiming Summary:")
    print(f"  Average per batch: {avg_time:.2f}s")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Overall rate: {result['results_count'] / total_time:.2f} results/s")


# ============================================================================
# EXAMPLE 10: Using with AgentExecutionRequest (Advanced)
# ============================================================================

def example_10_agent_execution_request():
    """
    Use paginated_search via the execute_task() interface (advanced).
    """
    print("\n" + "="*80)
    print("EXAMPLE 10: Using execute_task() with Dictionary (Advanced)")
    print("="*80 + "\n")
    
    agent = WebSearchAgent()
    
    # Create execution request as dictionary
    request = {
        "operation": "paginated_search",
        "parameters": {
            "query": "Cloud computing platforms",
            "max_results": 20,
            "batch_size": 5,
            "max_batches": 5,
            "backend": "auto"
        }
    }
    
    print(f"✓ Created execution request")
    print(f"  Operation: {request['operation']}")
    print(f"  Query: {request['parameters']['query']}")
    
    # Execute
    response = agent.execute_task(request)
    
    print(f"\n✓ Execution completed")
    print(f"  Results: {response.get('results_count', 0)}")
    print(f"  Status: {'Success' if response.get('success', False) else 'Failed'}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run all examples."""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*15 + "PAGINATED WEBSEARCH - REAL-WORLD USAGE EXAMPLES" + " "*15 + "║")
    print("╚" + "="*78 + "╝")
    
    examples = [
        ("Basic Pagination", example_1_basic_pagination),
        ("Small Batch Size", example_2_small_batch_size),
        ("Large Result Set", example_3_large_result_set),
        ("Error Handling", example_4_error_handling),
        ("Different Backends", example_5_different_backends),
        ("Export to JSON", example_6_export_to_json),
        ("Process in Chunks", example_7_process_in_chunks),
        ("Combine with Deep Search", example_8_combine_with_deep_search),
        ("Monitor Progress", example_9_monitor_progress),
        ("AgentExecutionRequest", example_10_agent_execution_request),
    ]
    
    # Run first 3 examples to avoid excessive API calls
    for idx, (name, func) in enumerate(examples[:3], 1):
        try:
            print(f"\n🔹 Example {idx}: {name}")
            func()
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
    
    # Show remaining examples as code-only
    print("\n" + "="*80)
    print("Additional Examples (Code-Only - Uncomment to Run)")
    print("="*80)
    for idx, (name, func) in enumerate(examples[3:], 4):
        print(f"  {idx}. {name} - example_{idx}_{name.lower().replace(' ', '_')}()")
    
    print("\n✅ Example execution complete!")
    print("   (Run first 3 examples to avoid excessive API calls)")
    print("   Uncomment additional examples in main() to run them")


if __name__ == "__main__":
    main()
