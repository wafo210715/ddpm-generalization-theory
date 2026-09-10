#!/usr/bin/env python3
"""Build an interactive stack explorer: turn a cell over, and break it on purpose.

The static 3-D picture shows what a cell is. This one exists so the structure
can be destroyed a piece at a time and the damage watched, which is the only
way some of it becomes visible.

Two controls, and each answers a different question.

  Unit size.  Coarsen the grid and watch which channels stop changing. The
  footprint sheet loses its alleys immediately; the ring-based sheets look the
  same at 64 units as at 8, because their meaning was never at that scale.

  Shuffle a sheet.  Move that sheet's values to random positions. This keeps
  its histogram exactly and destroys two things at once: where its own values
  sit relative to each other, and where they sit relative to the other five
  sheets. The live numbers underneath separate those two casualties --- Moran's
  I for the first, the channel correlations for the second.

The second one is the point of checkpoint B10b. Shuffling every sheet
independently gives a cell whose six histograms are all still perfect and whose
six-number combinations are nonsense. If per-channel marginals were the whole
check, that cell would pass.

Usage:
    python gen_stack_explorer.py --demo_npz /path/to/demo3.npz
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DDPM = os.path.join(REPO, "ddpm-win-master")
if DDPM not in sys.path:
    sys.path.insert(0, DDPM)

CSS = open(os.path.join(HERE, "theme.css"), encoding="utf-8").read()
FONTS = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=Archivo:wght@600;700&family=Source+Sans+3:wght@400;600&'
         'family=IBM+Plex+Mono:wght@500;600&display=swap">')
PLOTLY = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.32.0/plotly.min.js"

PLAIN = {
    "building_binary": "is there a building on this unit",
    "building_density": "how much building within 100 m",
    "accessibility_gravity": "how easy to reach shops by road",
    "poi_count": "how many shops within 300 m",
    "poi_entropy": "how mixed those shops are",
    "road_density": "how much street within 100 m",
}

EXTRA_CSS = """
.wrap{display:grid;grid-template-columns:260px 1fr;gap:26px;align-items:start;}
@media (max-width:900px){.wrap{grid-template-columns:1fr}}
.panel{background:var(--surface);border:1px solid var(--line);padding:16px;}
.panel h3{margin:0 0 4px;font:600 13px/1.3 "Archivo",sans-serif;}
.panel .hint{margin:0 0 14px;font-size:12px;color:var(--muted);line-height:1.45;}
.panel label{display:block;font:500 11px/1 "IBM Plex Mono",monospace;
  letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:16px 0 6px;}
.seg{display:flex;gap:4px;flex-wrap:wrap;}
.seg button{flex:1;min-width:44px;padding:6px 4px;font:500 12px "IBM Plex Mono",monospace;
  background:var(--bg);border:1px solid var(--line);color:var(--ink);cursor:pointer;}
.seg button[aria-pressed="true"]{background:var(--spine);border-color:var(--spine);color:#fff;}
.chrow{display:flex;align-items:center;gap:8px;margin:5px 0;}
.chrow .nm{flex:1;font-size:12px;}
.chrow button{padding:4px 9px;font:500 11px "IBM Plex Mono",monospace;
  background:var(--bg);border:1px solid var(--line);color:var(--ink);cursor:pointer;}
.chrow button.on{background:var(--gate);border-color:var(--gate);color:#fff;}
.wide{width:100%;padding:9px;margin-top:12px;font:600 12px "IBM Plex Mono",monospace;
  background:var(--bg);border:1px solid var(--line);color:var(--ink);cursor:pointer;}
.wide.warn{border-color:var(--gate);color:var(--gate);}
#plot{width:100%;height:560px;}
table.live{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;
  font-size:13px;margin-top:6px;}
table.live th,table.live td{padding:5px 8px 5px 0;border-bottom:1px solid var(--line);
  text-align:right;}
table.live th:first-child,table.live td:first-child{text-align:left;}
table.live th{font:600 10px/1.4 "IBM Plex Mono",monospace;letter-spacing:.08em;
  text-transform:uppercase;color:var(--muted);}
td.drop{color:var(--gate);font-weight:600;}
td.keep{color:var(--muted);}
.mono{font:500 12px "IBM Plex Mono",monospace;}
"""


def pool(a, factor):
    n = a.shape[-1] // factor
    return a[..., :n * factor, :n * factor].reshape(
        *a.shape[:-2], n, factor, n, factor).mean(axis=(-3, -1))


def build(demo_npz, out_path, size=64):
    d = np.load(demo_npz, allow_pickle=True)
    names = [str(x) for x in d["names"]]
    real = d["real_std"][0]
    gen = d["gen_std"][0]
    if real.shape[-1] > size:
        real = pool(real, real.shape[-1] // size)
    if gen.shape[-1] > size:
        gen = pool(gen, gen.shape[-1] // size)

    both = np.concatenate([real.ravel(), gen.ravel()])
    cmin, cmax = float(np.percentile(both, 1)), float(np.percentile(both, 99))

    payload = {
        "names": names,
        "plain": [PLAIN.get(n, n) for n in names],
        "size": int(real.shape[-1]),
        "cmin": round(cmin, 4), "cmax": round(cmax, 4),
        "real": [[[round(float(v), 3) for v in row] for row in grid] for grid in real],
        "gen": [[[round(float(v), 3) for v in row] for row in grid] for grid in gen],
    }

    body = f"""<main class="page doc" style="max-width:1160px">
  <header class="masthead" style="max-width:none">
    <p class="eyebrow">interactive · companion to Data in, data out</p>
    <h1>Turn a cell over</h1>
    <p class="lede">The same six sheets, but you can break them. Coarsen the
    units and watch which sheets stop caring. Shuffle a sheet and watch what
    survives — its histogram does, everything about where its values sit does
    not. Drag to rotate, scroll to zoom.</p>
  </header>

  <div class="wrap">
    <div>
      <div class="panel">
        <h3>Which cell</h3>
        <p class="hint">The real tile, or one the demo model invented.</p>
        <div class="seg" id="src">
          <button data-src="real" aria-pressed="true">real</button>
          <button data-src="gen" aria-pressed="false">generated</button>
        </div>

        <label>units across the cell</label>
        <p class="hint" style="margin:-8px 0 8px">1.2 km divided this many ways.
        64 units is 18.8 m each; 8 units is 150 m each.</p>
        <div class="seg" id="res"></div>

        <label>shuffle a sheet</label>
        <p class="hint" style="margin:-8px 0 8px">Moves that sheet's values to
        random spots. Its histogram cannot change. Everything about where they
        sit does.</p>
        <div id="chans"></div>

        <button class="wide warn" id="shuffleAll">shuffle every sheet</button>
        <button class="wide" id="reset">put everything back</button>
      </div>
    </div>

    <div>
      <div id="plot"></div>
      <h3 style="margin:22px 0 2px;font:600 15px/1.3 Archivo,sans-serif">
        what each shuffle costs</h3>
      <p style="margin:0 0 8px;font-size:13px;color:var(--muted);max-width:70ch">
        <b>Clumping</b> is Moran's I: does a unit resemble the ones next to it.
        Shuffling drops it to zero, because it is the only column here that is
        about <em>where</em>. <b>Agreement with the other sheets</b> is how well
        this sheet's values line up with the other five at the same unit —
        shuffling destroys that too, and no per-sheet histogram would notice.
      </p>
      <table class="live">
        <thead><tr><th>sheet</th><th>clumping now</th><th>was</th>
          <th>agreement now</th><th>was</th></tr></thead>
        <tbody id="stats"></tbody>
      </table>
      <div class="callout" style="margin-top:18px">
        Shuffle every sheet. All six histograms stay exactly as they were, and
        every six-number combination in the cell becomes nonsense — dense
        building sitting on zero roads, high shop-mix with no shops. That is
        what checkpoint <b>B10b</b> is for: per-channel marginals cannot see it.
      </div>
    </div>
  </div>

  <footer style="max-width:none">
    Built by <code>discussion/pipeline_map/gen_stack_explorer.py</code> from the
    real HCMC store, pooled to {payload["size"]} × {payload["size"]}. Values are
    standardized, so a colour means the same number on the real cell and the
    generated one.
  </footer>
</main>

<script src="{PLOTLY}"></script>
<script>
const D = {json.dumps(payload)};
const MAGMA = [[0,"#000004"],[0.125,"#180f3d"],[0.25,"#440f76"],[0.375,"#721f81"],
  [0.5,"#9e2f7f"],[0.625,"#cd4071"],[0.75,"#f1605d"],[0.875,"#fd9668"],[1,"#fcfdbf"]];
const N = D.names.length;
let src = "real", res = D.size;
let shuffled = new Array(N).fill(false);
let seeds = new Array(N).fill(0);

// A cheap deterministic shuffle so a sheet looks the same each time you press
// the button, and so "put everything back" is exact.
function rng(seed) {{
  let s = seed >>> 0 || 1;
  return () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296;
}}
function poolGrid(g, factor) {{
  if (factor <= 1) return g;
  const n = Math.floor(g.length / factor), out = [];
  for (let r = 0; r < n; r++) {{
    const row = [];
    for (let c = 0; c < n; c++) {{
      let s = 0;
      for (let i = 0; i < factor; i++)
        for (let j = 0; j < factor; j++) s += g[r * factor + i][c * factor + j];
      row.push(s / (factor * factor));
    }}
    out.push(row);
  }}
  return out;
}}
function shuffleGrid(g, seed) {{
  const flat = [].concat(...g), rand = rng(seed + 7);
  for (let i = flat.length - 1; i > 0; i--) {{
    const j = Math.floor(rand() * (i + 1));
    [flat[i], flat[j]] = [flat[j], flat[i]];
  }}
  const n = g.length, out = [];
  for (let r = 0; r < n; r++) out.push(flat.slice(r * n, (r + 1) * n));
  return out;
}}
function grids(applyShuffle) {{
  const factor = Math.round(D.size / res);
  return D[src].map((g, k) => {{
    let out = poolGrid(g, factor);
    if (applyShuffle && shuffled[k]) out = shuffleGrid(out, seeds[k]);
    return out;
  }});
}}

// --- the two numbers under the plot ------------------------------------
function moransI(g) {{
  const n = g.length, flat = [].concat(...g);
  const mean = flat.reduce((a, b) => a + b, 0) / flat.length;
  let varr = 0;
  for (const v of flat) varr += (v - mean) ** 2;
  if (varr < 1e-12) return 0;
  let num = 0;
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {{
    let sum = 0, cnt = 0;
    for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {{
      if (!dr && !dc) continue;
      const rr = r + dr, cc = c + dc;
      if (rr < 0 || cc < 0 || rr >= n || cc >= n) continue;
      sum += g[rr][cc] - mean; cnt++;
    }}
    if (cnt) num += (g[r][c] - mean) * (sum / cnt);
  }}
  return num / varr;
}}
function corr(a, b) {{
  const x = [].concat(...a), y = [].concat(...b), n = x.length;
  const mx = x.reduce((p, q) => p + q, 0) / n, my = y.reduce((p, q) => p + q, 0) / n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) {{
    sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; syy += (y[i] - my) ** 2;
  }}
  return (sxx < 1e-12 || syy < 1e-12) ? 0 : sxy / Math.sqrt(sxx * syy);
}}
// How well one sheet lines up with the other five at the same unit.
function agreement(all, k) {{
  let total = 0;
  for (let j = 0; j < all.length; j++) if (j !== k) total += Math.abs(corr(all[k], all[j]));
  return total / (all.length - 1);
}}

function draw() {{
  const now = grids(true), base = grids(false);
  const n = now[0].length;
  const axis = Array.from({{length: n}}, (_, i) => i);
  const GAP = Math.max(n * 0.42, 6);

  const traces = now.map((g, k) => ({{
    type: "surface", showscale: k === 0, name: D.names[k],
    x: axis, y: axis,
    z: g.map(row => row.map(() => k * GAP)),
    surfacecolor: g, colorscale: MAGMA, cmin: D.cmin, cmax: D.cmax,
    hovertemplate: D.names[k] + "<br>value %{{surfacecolor:.2f}}<extra></extra>",
    colorbar: {{title: {{text: "standardized<br>value", font: {{size: 10}}}},
                len: .7, thickness: 12}},
    contours: {{x: {{highlight: false}}, y: {{highlight: false}}, z: {{highlight: false}}}},
  }}));

  Plotly.react("plot", traces, {{
    margin: {{l: 0, r: 0, t: 0, b: 0}},
    paper_bgcolor: "rgba(0,0,0,0)",
    scene: {{
      xaxis: {{visible: false}}, yaxis: {{visible: false}}, zaxis: {{visible: false}},
      aspectmode: "manual", aspectratio: {{x: 1, y: 1, z: .85}},
      camera: {{eye: {{x: 1.5, y: -1.5, z: .9}}}},
    }},
  }}, {{displayModeBar: false, responsive: true}});

  const rows = now.map((g, k) => {{
    const mi = moransI(g), mi0 = moransI(base[k]);
    const ag = agreement(now, k), ag0 = agreement(base, k);
    const dropped = shuffled[k];
    return `<tr>
      <td>${{D.names[k]}}</td>
      <td class="${{dropped ? "drop" : "keep"}}">${{mi.toFixed(3)}}</td>
      <td class="keep">${{mi0.toFixed(3)}}</td>
      <td class="${{dropped ? "drop" : "keep"}}">${{ag.toFixed(3)}}</td>
      <td class="keep">${{ag0.toFixed(3)}}</td></tr>`;
  }});
  document.getElementById("stats").innerHTML = rows.join("");
}}

// --- controls -----------------------------------------------------------
const resBox = document.getElementById("res");
[8, 16, 32, D.size].filter((v, i, a) => a.indexOf(v) === i && v <= D.size)
  .forEach(v => {{
    const b = document.createElement("button");
    b.textContent = v; b.dataset.res = v;
    b.setAttribute("aria-pressed", String(v === res));
    b.onclick = () => {{
      res = v;
      [...resBox.children].forEach(c =>
        c.setAttribute("aria-pressed", String(+c.dataset.res === v)));
      draw();
    }};
    resBox.appendChild(b);
  }});

const chans = document.getElementById("chans");
D.names.forEach((nm, k) => {{
  const row = document.createElement("div");
  row.className = "chrow";
  row.innerHTML = `<span class="nm" title="${{D.plain[k]}}">${{nm}}</span>`;
  const b = document.createElement("button");
  b.textContent = "shuffle";
  b.onclick = () => {{
    shuffled[k] = !shuffled[k];
    if (shuffled[k]) seeds[k] = (seeds[k] + 1) * 31 + k;
    b.className = shuffled[k] ? "on" : "";
    b.textContent = shuffled[k] ? "shuffled" : "shuffle";
    draw();
  }};
  row.appendChild(b);
  chans.appendChild(row);
}});

document.getElementById("shuffleAll").onclick = () => {{
  shuffled = shuffled.map(() => true);
  seeds = seeds.map((s, k) => (s + 1) * 31 + k);
  [...chans.querySelectorAll("button")].forEach(b => {{
    b.className = "on"; b.textContent = "shuffled";
  }});
  draw();
}};
document.getElementById("reset").onclick = () => {{
  shuffled = shuffled.map(() => false);
  [...chans.querySelectorAll("button")].forEach(b => {{
    b.className = ""; b.textContent = "shuffle";
  }});
  draw();
}};
document.querySelectorAll("#src button").forEach(b => {{
  b.onclick = () => {{
    src = b.dataset.src;
    document.querySelectorAll("#src button").forEach(o =>
      o.setAttribute("aria-pressed", String(o === b)));
    draw();
  }};
}});

draw();
</script>"""

    head = (f'<title>Turn a cell over</title>\n<style>{CSS}\n{EXTRA_CSS}</style>\n'
            f'{FONTS}\n')
    doc = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
           '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
           + head + '</head>\n<body>\n' + body + '\n</body>\n</html>\n')
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"Wrote {out_path} ({os.path.getsize(out_path) / 1024:.0f} kB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo_npz", required=True)
    parser.add_argument("--out", default=os.path.join(REPO, "discussion",
                                                      "stack_explorer.html"))
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()
    build(args.demo_npz, args.out, args.size)
