"""Check available DDGS engines and backends."""
from ddgs import DDGS, engines
import inspect

print("=" * 80)
print("DDGS Search Backends Investigation")
print("=" * 80)

# Check available engines
print("\nAvailable engine classes:")
for name in dir(engines):
    obj = getattr(engines, name)
    if inspect.isclass(obj) and not name.startswith('_'):
        print(f"  - {name}")

# Check DDGS methods
print("\nDDGS instance methods:")
ddgs = DDGS()
for method in ['text', 'images', 'videos', 'news']:
    if hasattr(ddgs, method):
        print(f"  - {method}()")

# Test different backends
print("\nTesting backends:")
backends_to_test = ['html', 'api', 'lite']

for backend in backends_to_test:
    try:
        ddgs = DDGS()
        results = ddgs.text("test", max_results=1, backend=backend)
        results_list = list(results)
        print(f"  ✓ backend='{backend}': {len(results_list)} results")
    except Exception as e:
        print(f"  ✗ backend='{backend}': {str(e)[:60]}")

# Check what engines are used by default
print("\nChecking internal engines:")
try:
    ddgs = DDGS()
    engines_list = ddgs._get_engines('text')
    print(f"  Default text search engines: {engines_list}")
except Exception as e:
    print(f"  Could not get engines: {e}")

print("\n" + "=" * 80)
