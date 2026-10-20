# TODO: review this file
import json
rows = [json.loads(l) for l in open('benchmark_results/lorenz_10x/sweep.jsonl')]
def is_nan(x):
    return x is None or (isinstance(x, float) and x != x)
def missing(method):
    return [r for r in rows if r['method']==method and (r.get('error') or is_nan(r.get('NMSE_force')))]
for m in ['baseline', 'sfi_v1']:
    miss = missing(m)
    print(f"{m}: {len(miss)} missing/failed/NaN")
    if miss:
        print(f"  Example: {miss[0]}")
