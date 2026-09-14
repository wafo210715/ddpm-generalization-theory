"""Bundle the three preview products into the one file the page loads."""
import json, pathlib
REPO = pathlib.Path(__file__).resolve().parent.parent
P = REPO / "data/hcmc/processed/preview2025"
DST = REPO.parent / "discussion/hcmc_layers_data.js"

mos = json.load(open(P / "mosaic.json"))
til = json.load(open(P / "tiles.json"))
var = json.load(open(P / "variance_split.json"))

# tiles with at least 100 buildings -- the trainable set
trainable = var.get("n_built_tiles") or 2881

out = {"mosaic": mos, "tiles": til, "variance": var,
       "trainable": f"{trainable:,}"}
js = "window.HCMC=" + json.dumps(out, separators=(",", ":")) + ";"
DST.write_text(js)
print(f"wrote {DST.name}  {len(js)/1e6:.2f} MB")
print(f"  mosaic layers : {len(mos['mosaic'])}")
print(f"  gallery tiles : {len(til['gallery'])}")
print(f"  resolutions   : {var['resolutions']}")
