"""Quick test for Karnataka CEG deep search."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task_manager.sub_agents.web_search_agent import WebSearchAgent
import json

agent = WebSearchAgent()

print("Testing deep_search on Karnataka CEG website...")
print("=" * 60)

result = agent.deep_search(
    query="list of districts in Karnataka",
    target_url="https://ceg.karnataka.gov.in/en",
    max_results=3,
    max_depth=2,
    extract_structured_data=True
)

print(f"Success: {result.get('success')}")
print(f"Pages Visited: {result.get('pages_visited')}")
print()

if result.get('summary'):
    print("Summary:")
    for k, v in result.get('summary', {}).items():
        print(f"  - {k}: {v}")
print()

if result.get('extracted_data'):
    print(f"Extracted Data ({len(result['extracted_data'])} items):")
    for i, item in enumerate(result['extracted_data'][:30]):
        print(f"  {i+1}. {str(item)[:100]}")
else:
    print("No structured data extracted")

print()
print("Findings:")
for f in result.get('findings', [])[:5]:
    url = f.get('url', 'N/A')
    title = f.get('title', 'N/A')
    score = f.get('relevance_score', 0)
    preview = f.get('text_preview', '')[:300]
    
    print(f"  URL: {url[:80]}")
    print(f"  Title: {title[:60]}")
    print(f"  Relevance: {score}")
    if preview:
        print(f"  Preview: {preview}...")
    print()

print("=" * 60)
