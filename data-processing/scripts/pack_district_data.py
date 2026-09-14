"""Split the district renders into files that fit the artifact per-file limit."""
import json, pathlib
REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "data/hcmc/processed/preview2025/districts.json"
DST = REPO.parent / "discussion"

d = json.load(open(SRC))
keys = list(d)
half = (len(keys) + 1) // 2
parts = [keys[:half], keys[half:]]
for i, ks in enumerate(parts):
    chunk = {k: d[k] for k in ks}
    js = ("window.HCMC_DISTRICTS=Object.assign(window.HCMC_DISTRICTS||{},"
          + json.dumps(chunk, separators=(",", ":")) + ");")
    p = DST / f"hcmc_districts_data_{chr(97+i)}.js"
    p.write_text(js)
    print(f"  {p.name}  {len(js)/1e6:5.1f} MB  {ks}")
