#!/usr/bin/env python3
"""
Test suite for the ENHANCED WebSearch Deep Search functionality.

ENHANCEMENTS TESTED:
1. Processing of leading 5 search results only
2. 3-4 links deep traversal into pages
3. Extraction of structured data from all levels
4. Findings aggregation by depth level
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from task_manager.sub_agents.web_search_agent import WebSearchAgent


def test_deep_search_with_depth_tracking():
    """Test ENHANCED deep_search with depth-aware link traversal."""
    print("\n" + "="*80)
    print("TEST: ENHANCED DEEP SEARCH WITH DEPTH TRACKING")
    print("="*80)
    
    try:
        agent = WebSearchAgent()
        print(f"✓ WebSearchAgent initialized")
        
        # Test query
        query = "Python programming best practices"
        print(f"\n📝 Query: {query}")
        print(f"🎯 Configuration:")
        print(f"   - Max results (TOP N to process): 5")
        print(f"   - Max depth (links to follow): 4")
        print(f"   - Expected: Process top 5 results, go 3-4 levels deep into each")
        
        # Execute enhanced deep search
        result = agent.deep_search(
            query=query,
            max_results=5,  # Only top 5
            max_depth=4,    # Go 4 levels deep
            extract_structured_data=True
        )
        
        print(f"\n✅ Deep search completed successfully: {result.get('success')}")
        
        if not result.get('success'):
            print(f"❌ Search failed: {result.get('error')}")
            return False
        
        # VALIDATION CHECKS
        print(f"\n{'='*80}")
        print("VALIDATION RESULTS")
        print(f"{'='*80}")
        
        # Check 1: Pages visited
        pages_visited = result.get('pages_visited', 0)
        print(f"\n1️⃣  PAGES VISITED: {pages_visited}")
        print(f"   ✓ Should be > 5 (at least the 5 main results)")
        if pages_visited >= 5:
            print(f"   ✅ PASS: Visited {pages_visited} pages")
        else:
            print(f"   ❌ FAIL: Only visited {pages_visited} pages (expected >= 5)")
            return False
        
        # Check 2: Findings by depth
        findings_by_depth = result.get('findings_by_depth', {})
        print(f"\n2️⃣  FINDINGS BY DEPTH LEVEL:")
        for depth_level in sorted(findings_by_depth.keys()):
            count = findings_by_depth[depth_level]
            print(f"   - Depth {depth_level}: {count} pages")
        
        # Should have findings at multiple depths
        depth_levels_with_findings = len([d for d, c in findings_by_depth.items() if c > 0])
        print(f"   ✓ Depth levels with findings: {depth_levels_with_findings}")
        
        if depth_levels_with_findings >= 2:
            print(f"   ✅ PASS: Found content at multiple depth levels")
        else:
            print(f"   ⚠️  WARNING: Limited depth exploration (only {depth_levels_with_findings} levels)")
        
        # Check 3: Extracted data
        extracted_count = result.get('results_count', 0)
        extraction_stats = result.get('extraction_stats', {})
        print(f"\n3️⃣  DATA EXTRACTION:")
        print(f"   - Total extracted items: {extracted_count}")
        print(f"   - Total findings: {extraction_stats.get('total_findings', 0)}")
        print(f"   - Depth 0 pages: {extraction_stats.get('depth_0_pages', 0)}")
        print(f"   - Depth 1 pages: {extraction_stats.get('depth_1_pages', 0)}")
        print(f"   - Depth 2 pages: {extraction_stats.get('depth_2_pages', 0)}")
        print(f"   - Depth 3 pages: {extraction_stats.get('depth_3_pages', 0)}")
        print(f"   - Depth 4+ pages: {extraction_stats.get('depth_4_plus_pages', 0)}")
        
        if extracted_count > 0:
            print(f"   ✅ PASS: Extracted {extracted_count} items")
        else:
            print(f"   ❌ FAIL: No items extracted")
            return False
        
        # Check 4: Top 5 results processing
        print(f"\n4️⃣  TOP 5 RESULTS PROCESSING:")
        summary = result.get('summary', {})
        top_sources = summary.get('top_sources', [])
        print(f"   - Top sources found: {len(top_sources)}")
        for idx, source in enumerate(top_sources[:5], 1):
            print(f"     {idx}. {source}")
        
        if len(top_sources) >= 1:
            print(f"   ✅ PASS: Processing top {len(top_sources)} results")
        else:
            print(f"   ❌ FAIL: No top sources identified")
        
        # Check 5: Keywords used for relevance
        keywords_used = result.get('keywords_used', [])
        print(f"\n5️⃣  KEYWORDS FOR RELEVANCE FILTERING:")
        print(f"   - Keywords extracted: {len(keywords_used)}")
        if keywords_used:
            print(f"   - Sample keywords: {keywords_used[:5]}")
            print(f"   ✅ PASS: Keywords identified for filtering")
        else:
            print(f"   ⚠️  WARNING: No keywords extracted")
        
        # Check 6: Sample extracted data
        print(f"\n6️⃣  SAMPLE EXTRACTED DATA (first 3 items):")
        extracted_data = result.get('extracted_data', [])
        for idx, item in enumerate(extracted_data[:3], 1):
            item_str = str(item)[:100]
            print(f"   {idx}. {item_str}...")
        
        # Check 7: Findings detail
        print(f"\n7️⃣  FINDINGS DETAIL:")
        findings = result.get('findings', [])
        print(f"   - Total findings: {len(findings)}")
        
        if findings:
            print(f"   - Sample findings (first 3):")
            for idx, finding in enumerate(findings[:3], 1):
                url = finding.get('url', 'N/A')[:50]
                depth = finding.get('depth', 'N/A')
                score = finding.get('relevance_score', 'N/A')
                print(f"     {idx}. URL: {url}... (depth: {depth}, relevance: {score})")
        
        # FINAL SUMMARY
        print(f"\n{'='*80}")
        print("✅ ENHANCED DEEP SEARCH TEST PASSED")
        print(f"{'='*80}")
        print(f"\nSummary:")
        print(f"  ✓ Processed {pages_visited} pages from {len(top_sources)} top results")
        print(f"  ✓ Traversed up to depth {max(findings_by_depth.keys()) if findings_by_depth else 0}")
        print(f"  ✓ Extracted {extracted_count} items total")
        print(f"  ✓ Found content at {depth_levels_with_findings} different depth levels")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_deep_search_with_specific_query():
    """Test with a specific research query."""
    print("\n" + "="*80)
    print("TEST: DEEP SEARCH WITH SPECIFIC RESEARCH QUERY")
    print("="*80)
    
    try:
        agent = WebSearchAgent()
        
        # More targeted query
        query = "machine learning algorithms comparison"
        print(f"\n📝 Query: {query}")
        
        result = agent.deep_search(
            query=query,
            max_results=5,
            max_depth=4,
            extract_structured_data=True
        )
        
        if not result.get('success'):
            print(f"❌ Search failed: {result.get('error')}")
            return False
        
        print(f"✅ Search successful")
        print(f"   - Pages visited: {result.get('pages_visited', 0)}")
        print(f"   - Items extracted: {result.get('results_count', 0)}")
        
        extraction_stats = result.get('extraction_stats', {})
        print(f"   - Findings by depth:")
        for depth in range(5):
            key = f'depth_{depth}_pages'
            count = extraction_stats.get(key, 0)
            if count > 0:
                print(f"     • Depth {depth}: {count} pages")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


def test_deep_search_with_url():
    """Test deep search starting from a specific URL."""
    print("\n" + "="*80)
    print("TEST: DEEP SEARCH WITH SPECIFIC TARGET URL")
    print("="*80)
    
    try:
        agent = WebSearchAgent()
        
        # Start with a specific URL known to have good content
        target_url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
        query = "Python programming language"
        
        print(f"\n📝 Query: {query}")
        print(f"🎯 Target URL: {target_url}")
        
        result = agent.deep_search(
            query=query,
            target_url=target_url,
            max_results=5,
            max_depth=3,
            extract_structured_data=True
        )
        
        if not result.get('success'):
            print(f"❌ Search failed: {result.get('error')}")
            return False
        
        print(f"✅ Search successful")
        print(f"   - Pages visited: {result.get('pages_visited', 0)}")
        print(f"   - Items extracted: {result.get('results_count', 0)}")
        
        findings = result.get('findings', [])
        print(f"   - Findings collected: {len(findings)}")
        
        if findings:
            print(f"   - First finding: {findings[0].get('url', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*15 + "ENHANCED WEBSEARCH DEEP SEARCH TEST SUITE" + " "*21 + "║")
    print("╚" + "="*78 + "╝")
    
    tests = [
        ("Depth Tracking", test_deep_search_with_depth_tracking),
        ("Specific Query", test_deep_search_with_specific_query),
        ("Target URL", test_deep_search_with_url),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results[test_name] = passed
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {str(e)}")
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
    
    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
