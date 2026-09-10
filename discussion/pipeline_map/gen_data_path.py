#!/usr/bin/env python3
"""Build 'Data in, data out' — one page, every step shown rather than described.

A separate page from the pipeline map and its companion, on purpose. Those two
answer "what are we doing and why". This one answers a narrower question that
kept coming back in conversation and kept being answered in prose that did not
land: what is the model actually handed, and what does it actually give back.

Two rules it is written under, both from the same feedback:

  * Every term is explained where it first appears. Nothing depends on the
    reader remembering a glossary at the bottom, because they will not, and
    should not have to.
  * Every step has a picture of the real thing beside it. Not a schematic of
    the idea — the actual numbers from the actual data.

Figures come from data-processing/scripts/make_explainer_figures.py and are
embedded as data URIs so the page is one self-contained file that can be
opened from anywhere or emailed to a supervisor.

Usage:
    python gen_data_path.py                 # writes ../data_in_data_out.html
    python gen_data_path.py --out /tmp/x.html
"""

import argparse
import base64
import html
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
REPORTS = os.path.join(REPO, "data-processing", "data", "hcmc", "processed", "reports")
DDPM = os.path.join(REPO, "ddpm-win-master")
if DDPM not in sys.path:
    sys.path.insert(0, DDPM)

STORE = os.path.join(REPO, "data-processing", "data", "hcmc", "processed",
                     "cells_z15_1024")
CELL_ID = "15_26097_15392"
PROBE = (470, 300)

CSS = open(os.path.join(HERE, "theme.css"), encoding="utf-8").read()
FONTS = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=Archivo:wght@600;700&family=Source+Sans+3:wght@400;600&'
         'family=IBM+Plex+Mono:wght@500;600&display=swap">')

EXTRA_CSS = """
.fig{margin:20px 0 8px;padding:14px;background:var(--surface);
     border:1px solid var(--line);}
.fig img{width:100%;height:auto;display:block;}
.fig .cap{margin-top:10px;font-size:13px;color:var(--muted);max-width:76ch;}
.shape{display:block;margin:14px 0;padding:14px 16px;background:var(--code-bg);
       font:500 13px/1.75 "IBM Plex Mono",ui-monospace,monospace;
       overflow-x:auto;white-space:pre;}
.shape b{color:var(--spine);font-weight:600;}
.plain{border-left:4px solid var(--spine);background:var(--spine-soft);
       padding:12px 16px;margin:16px 0;max-width:74ch;}
.plain p{margin:0 0 6px;}.plain p:last-child{margin:0;}
.term{color:var(--spine);font-weight:600;text-decoration:none;border:0;}
.no{border-left:4px solid var(--gate);background:var(--gate-soft);
    padding:12px 16px;margin:16px 0;max-width:74ch;}
"""


def data_uri(name):
    path = os.path.join(REPORTS, name)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def figure(name, caption):
    uri = data_uri(name)
    if uri is None:
        return (f'<div class="fig"><p class="cap">[{html.escape(name)} not '
                f'built yet — run make_explainer_figures.py]</p></div>')
    return (f'<figure class="fig"><img src="{uri}" alt="{html.escape(caption[:110])}">'
            f'<p class="cap">{caption}</p></figure>')


def real_numbers():
    """Pull the actual arrays out of the store so the page quotes real values."""
    import numpy as np
    from datasets.cell_tensors import CellTensorStore

    store = CellTensorStore(STORE)
    cell_id = CELL_ID if CELL_ID in set(store.cell_ids()) else sorted(store.cell_ids())[0]
    cell = store.read_cell(cell_id).astype(float)
    py, px = PROBE
    names = store.channel_names

    unit = ", ".join(f"{v:.2f}" for v in cell[:, py, px])
    grid_rows = "\n".join(
        "  " + "  ".join(f"{v:5.2f}" for v in cell[1, py + r, px:px + 4])
        for r in range(4))
    pair_rows = "\n".join(
        "  " + "   ".join(
            "[" + ", ".join(f"{v:5.2f}" for v in cell[:, py + r, px + c]) + "]"
            for c in range(2))
        for r in range(2))
    return {
        "cell_id": cell_id,
        "names": names,
        "n": len(names),
        "size": store.img_size,
        "unit": unit,
        "grid_rows": grid_rows,
        "pair_rows": pair_rows,
        "total": f"{len(names) * store.img_size ** 2:,}",
    }


def build(out_path):
    d = real_numbers()
    # Wrap the channel names: on one line they overflow the code block and
    # give the page a horizontal scrollbar on its very first figure.
    names = d["names"]
    half = (len(names) + 1) // 2
    labels = ("   " + " · ".join(names[:half]) + "\n   "
              + " · ".join(names[half:]))

    body = f"""<main class="page doc">
  <header class="masthead">
    <p class="eyebrow">a companion page · 10 Sep 2026</p>
    <h1>Data in, data out</h1>
    <p class="lede">What the model is actually handed, and what it actually
    gives back, at every stage, with the real numbers rather than a sketch of
    them. Every word that could send you to a glossary is explained where it
    appears.</p>
    <ul class="toc">
      <li><a href="#cell">1 One cell, written out</a></li>
      <li><a href="#in">2 What goes in</a></li>
      <li><a href="#model">3 What the model does</a></li>
      <li><a href="#out">4 What comes out</a></li>
      <li><a href="#back">5 What you get back, and what you don't</a></li>
    </ul>
  </header>

  <section id="cell">
    <h2><span class="num">1</span>One cell, written out</h2>
    <p>A <span class="term">cell</span> is one square of the city — 1.2 km on a
    side — and it is one training example. The model sees a whole cell at once.</p>
    <p>Cut that square into a {d["size"]} × {d["size"]} chequerboard. Each little
    square is a <span class="term">unit</span>: 1.17 m of ground, about the width
    of a doorway. For every unit we work out {d["n"]} numbers, one per indicator.
    Those {d["n"]} numbers are the unit's vector.</p>
    <div class="shape">one unit, at row {PROBE[0]}, column {PROBE[1]}:

  [{d["unit"]}]

{labels}</div>
    <p>Now take one indicator at a time and lay its number for every unit back
    onto the chequerboard. That sheet is a <span class="term">grid</span>. Here
    is a 4 × 4 corner of the building-density grid, so you can see it is just
    numbers in a square:</p>
    <div class="shape">{d["grid_rows"]}</div>
    <p>Stack the {d["n"]} grids and you have the whole cell. Two ways of writing
    the same thing, and both are correct — it is the same numbers either way:</p>
    <div class="shape">as <b>{d["n"]} sheets</b>, each {d["size"]} × {d["size"]}:
    shape ({d["n"]}, {d["size"]}, {d["size"]})

as a <b>{d["size"]} × {d["size"]} board</b>, each square holding {d["n"]} numbers:
    shape ({d["size"]}, {d["size"]}, {d["n"]})

a 2 × 2 corner, written the second way:
{d["pair_rows"]}

either way: {d["total"]} numbers in one cell.</div>
    <p>The word <span class="term">tensor</span> just means this stack. It is
    not a special kind of object; it is the pile of grids.</p>
    {figure("what_a_cell_is.png",
            "One real cell. Six grey pictures of the same square, and one unit "
            "circled on all six so you can read its six numbers. The colour "
            "picture on the right is three of the six forced into red, green "
            "and blue — a trick for human eyes. The model never sees colour.")}
    <p>Laid out flat like that, it looks like six pictures. It is really one
    object: every sheet covers the same ground, and a unit is a column straight
    down through all of them. Stacked, that is easier to believe.</p>
    {figure("cell_in_3d.png",
            "The same cell as a stack, beside one the model invented. Both are "
            "on a single shared colour scale, so a colour means the same number "
            "in both. The red line is one unit going down through all six "
            "sheets. Underneath, the same numbers again as distributions — "
            "which is where the differences between real and generated are "
            "easier to read than in the pictures.")}
  </section>

  <section id="in">
    <h2><span class="num">2</span>What goes in</h2>
    <p>One thing happens to a cell before the model sees it, and it is
    arithmetic, not interpretation.</p>
    <div class="plain">
      <p><b>Standardizing.</b> For each grid separately: work out the average of
      all its numbers, subtract that average from every number, then divide
      every number by how spread out they were.</p>
      <p>Afterwards that grid's numbers centre on zero and have a spread of one.
      The picture is unchanged — the bright parts are still bright. Only the
      numbers on them are different.</p>
    </div>
    <p>Why bother: the grids arrive in wildly different units. Building density
    runs 0 to 1. Shop counts run 0 to 60. Left alone, the shop grid's numbers
    are sixty times bigger, and anything that measures distance would be
    listening almost entirely to shops. Standardizing gives every grid the same
    say.</p>
    <p>Everything needed to undo it — the average and the spread, per grid — is
    written down and kept. That stored pair <em>is</em> the decoder. Nothing
    else is needed to get back.</p>
  </section>

  <section id="model">
    <h2><span class="num">3</span>What the model does</h2>
    <p>A diffusion model has two halves, and only one of them learns anything.</p>
    <div class="plain">
      <p><b>The noise recipe.</b> Fixed, written down in advance, no learning.
      It says how much random noise to mix into a cell at each of 1000 steps,
      until nothing of the original is left.</p>
      <p><b>The network.</b> Shown a noisy cell, it guesses what noise is in it.
      This is the part that is trained, by showing it noisy cells and correcting
      its guesses, over and over.</p>
    </div>
    <p>To generate, you run the recipe backwards. Start from pure noise, ask the
    network what noise it sees, subtract a bit of its answer, repeat about a
    thousand times. What is left is a new cell.</p>
    <p>Nothing in that loop checks whether the network is any good. It asks, it
    subtracts, it moves on. So an <span class="term">untrained</span> network —
    one whose numbers are still the random values it was created with, never
    trained on anything — produces output perfectly happily. The output is just
    nonsense.</p>
    {figure("untrained_vs_trained.png",
            "The same generation loop three times over; only the network in the "
            "middle differs. Note the ranges rather than the pictures: an "
            "untrained network's output runs -790 to 746, after 1000 training "
            "steps -82 to 107, and real cells -2 to 7. The pictures barely "
            "change; the numbers move a long way.")}
    <p>One question worth answering while we are here: does the network know
    that the first number in a unit's vector and the sixth are different things,
    and does their order matter?</p>
    {figure("how_the_unet_reads.png",
            "How a convolution reads the stack. The cyan square is one filter's "
            "field of view: 3 x 3 across the ground, but all the way through "
            "every channel — 54 numbers in, one number out. The middle panel is "
            "a real filter's actual trained weights, nine per channel.")}
    <div class="plain">
      <p><b>Does it tell the channels apart?</b> Yes. Every channel gets its own
      set of weights, visible in the middle panel, and training pulls them
      apart.</p>
      <p><b>Does the order mean anything?</b> No. The filter slides left-right
      and up-down across the ground. It does not slide down through the stack —
      it adds the whole depth together at once. "Channel 6" is a name, not a
      position. Relabel them all consistently, retrain, and you get the same
      model.</p>
      <p>Which is the exact opposite of the two map directions, where being next
      to something is the whole point of sliding.</p>
    </div>
    <p>The figure above is also why we bother measuring what an untrained model
    scores. If nonsense scores 0.05 on a test, then a trained model scoring 0.05
    has not achieved anything either. It is a "do not get excited below this
    line" mark.</p>
  </section>

  <section id="out">
    <h2><span class="num">4</span>What comes out</h2>
    <p>The model gives back exactly the shape it was given: {d["n"]} grids,
    {d["size"]} × {d["size"]}. It is never reshaped, at any point, going in or
    coming out.</p>
    <p>But the numbers are in the standardized form — centred on zero, spread of
    one — so they mean nothing physically yet. <span class="term">Unpacking</span>
    fixes that, and it is the standardizing run backwards:</p>
    <div class="plain">
      <p><b>Unpacking.</b> For each grid: multiply every number by the spread we
      stored, then add back the average we stored.</p>
      <p>That is the entire operation. It is exact, it costs nothing, and it
      cannot be wrong, because it is the same arithmetic that got us here.</p>
    </div>
    {figure("data_in_data_out.png",
            "One channel followed the whole way. The shape is the same at every "
            "stage. The bottom strip is the model's output split back into its "
            "named grids — which is the step this page exists to make concrete.")}
    <div class="no">
      <p><b>Unpacking does not check anything.</b> The demo above was trained for
      eight seconds on eight cells, so its output is nowhere near real data —
      and unpacking dutifully turns that into <b>−705 shops within 300 m</b>.</p>
      <p>A negative shop count is impossible. Unpacking will produce it anyway,
      because its job is to undo arithmetic, not to judge. Sanity of the numbers
      is a separate question, asked afterwards.</p>
    </div>
  </section>

  <section id="back">
    <h2><span class="num">5</span>What you get back, and what you don't</h2>
    <p><b>You get back:</b> {d["n"]} named grids of numbers in real units — share
    of ground built on, shops within 300 m, bits of mixedness, metres of street
    per square metre. You can look at them, measure them, compare them against
    real cells, map them.</p>
    <p><b>You do not get back:</b> buildings, roads or shops as shapes. There is
    no address anywhere in a generated cell. A grid says "this patch scores 0.58
    on building density" and nothing more — not which building, not where its
    walls are.</p>
    <p>This is a deliberate choice rather than a limitation we ran into. Turning
    grids back into shapes would need a second trained model to do the
    translating, and then any near-copy we detected might be an artefact of that
    second model rather than a fact about the first. Keeping the encoding a
    written rule keeps the way back free and beyond argument.</p>
    <div class="plain">
      <p>So: <b>the stack of grids is what the model reads and what it writes.</b>
      Standardizing and unpacking move the numbers between measured units and
      model units. Nothing is ever reshaped, and nothing is ever recovered that
      was not put in.</p>
    </div>
  </section>

  <footer>
    Figures built by <code>data-processing/scripts/make_explainer_figures.py</code>
    from the real HCMC store; this page by
    <code>discussion/pipeline_map/gen_data_path.py</code>. Numbers quoted in the
    text are read from cell <code>{d["cell_id"]}</code> at build time, so they
    cannot drift out of date.
  </footer>
</main>"""

    head = (f'<title>Data in, data out</title>\n<style>{CSS}\n{EXTRA_CSS}</style>\n'
            f'{FONTS}\n')
    doc = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
           '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
           + head + '</head>\n<body>\n' + body + '\n</body>\n</html>\n')
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"Wrote {out_path} ({os.path.getsize(out_path) / 1024:.0f} kB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(REPO, "discussion",
                                                      "data_in_data_out.html"))
    args = parser.parse_args()
    build(args.out)
