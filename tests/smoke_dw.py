# TODO: review this file
"""Quick smoke test — double-well only, with memory tracking."""
import os, resource, sys, time

os.environ['MPLBACKEND'] = 'Agg'
os.environ['SFI_BENCH_MODE'] = 'quick'

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'examples'))

t0 = time.perf_counter()
exec(open(os.path.join(ROOT, 'examples', 'benchmarks', 'doublewell_bench.py')).read())
dt = time.perf_counter() - t0
peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f'\nPeak RSS: {peak_mb:.0f} MB, wall time: {dt:.0f}s')
print('=== SMOKE TEST PASSED ===')
