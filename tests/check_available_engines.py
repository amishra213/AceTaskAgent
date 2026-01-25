"""Check actual available search engines in DDGS."""
from ddgs import DDGS
import logging

logging.basicConfig(level=logging.WARNING)  # Reduce noise

print("=" * 80)
print("Available DDGS Search Engines")
print("=" * 80)

ddgs = DDGS()

# Get available engines for text search
try:
    # Try to access internal engine list
    available_engines = ddgs._get_engines('text')
    print(f"\nAvailable text search engines: {available_engines}")
except Exception as e:
    print(f"\nCouldn't get engines directly: {e}")

# Test individual engines
print("\nTesting individual engines:")
engines_to_test = [
    'auto',
    'duckduckgo',
    'google',
    'brave',
    'wikipedia',
    'yahoo',
    'yandex',
    'mojeek',
    'grokipedia'
]

for engine in engines_to_test:
    try:
        results = ddgs.text("test", max_results=1, backend=engine)
        results_list = list(results)
        print(f"  ✓ {engine:15s} - {len(results_list)} results")
    except Exception as e:
        error_msg = str(e)[:60]
        print(f"  ✗ {engine:15s} - {error_msg}")

print("\n" + "=" * 80)
