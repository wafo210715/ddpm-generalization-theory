"""Split the ward renders so the page opens fast and fetches the rest on demand.

All 24 wards at 16 layers is tens of megabytes.  Loading that up front makes the
page unusable, so the renders are split into parts: an index plus one file per
part.  The page loads the index and the first part, and injects the others as
same-origin scripts when a ward in them is chosen.
"""
import json, math, pathlib
REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "data/hcmc/processed/preview2025/wards.json"
DST = REPO.parent / "discussion"
TARGET_MB = 9.0

d = json.load(open(SRC))
sizes = {k: len(json.dumps(v, separators=(",", ":"))) / 1e6 for k, v in d.items()}
order = sorted(d, key=lambda k: -sizes[k])

parts, load = [], []
for k in order:                                   # greedy bin packing by size
    placed = False
    for i, p in enumerate(parts):
        if load[i] + sizes[k] <= TARGET_MB:
            p.append(k); load[i] += sizes[k]; placed = True; break
    if not placed:
        parts.append([k]); load.append(sizes[k])

index = {}
for i, ks in enumerate(parts, start=1):
    for k in ks:
        index[k] = i
    chunk = {k: d[k] for k in ks}
    js = ("window.HCMC_WARDS=Object.assign(window.HCMC_WARDS||{},"
          + json.dumps(chunk, separators=(",", ":")) + ");"
          + f"window.dispatchEvent(new CustomEvent('wards-part',{{detail:{i}}}));")
    (DST / f"hcmc_wards_data_{i}.js").write_text(js)
    print(f"  part {i}: {load[i-1]:5.1f} MB  {len(ks)} wards")

meta = {k: {kk: v[kk] for kk in ("km2", "tiles_covering", "block", "px", "m_per_px", "cells")}
        for k, v in d.items()}
(DST / "hcmc_wards_index.js").write_text(
    "window.HCMC_WARD_INDEX=" + json.dumps({"part": index, "meta": meta,
     "parts": len(parts)}, separators=(",", ":")) + ";")
print(f"\nindex + {len(parts)} parts, {sum(load):.1f} MB total")
