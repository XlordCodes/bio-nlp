import json
import numpy as np

with open("evaluation_output/evaluation_summary.json") as f:
    data = json.load(f)

trunc_rates, mm2_ids, edlib_ids = [], [], []
for ex in data["per_example"]:
    op = ex["operational_metrics"]
    trunc = op["chunks_without_eos"] / max(op["num_chunks"], 1)
    mm2 = ex.get("minimap2", {}).get("identity")
    if mm2 is None:
        continue
    trunc_rates.append(trunc)
    mm2_ids.append(mm2)
    edlib_ids.append(ex["alignment"]["identity"])

trunc_rates, mm2_ids, edlib_ids = map(np.array, (trunc_rates, mm2_ids, edlib_ids))
gaps = edlib_ids - mm2_ids

print(f"n = {len(trunc_rates)}")
print(f"corr(trunc_rate, mm2_identity)  = {np.corrcoef(trunc_rates, mm2_ids)[0,1]:.3f}")
print(f"corr(trunc_rate, edlib-mm2 gap) = {np.corrcoef(trunc_rates, gaps)[0,1]:.3f}")

zero_mask = trunc_rates == 0
if zero_mask.any():
    print(f"\n{zero_mask.sum()} example(s) with trunc_rate == 0:")
    print(f"  mean edlib_identity = {edlib_ids[zero_mask].mean():.3f}")
    print(f"  mean mm2_identity   = {mm2_ids[zero_mask].mean():.3f}")
else:
    print("\nNo trunc_rate==0 examples in this run -- can't isolate a truncation-free baseline this way.")