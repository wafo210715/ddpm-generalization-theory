"""Count OSM features by tag over the HCMC study-area polygon.

Answers the "data source feasibility" column of the vitality factor tables with
measured counts instead of guesses.  Uses Overpass `out count`, which returns a
number rather than geometry, so the whole battery costs a fraction of a fetch.
"""
import json, sys, time, pathlib
import geopandas as gpd
import requests

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
CACHE = REPO / "data/hcmc/processed/reports/osm_tag_coverage.json"
URL = "https://overpass-api.de/api/interpreter"
UA = {"User-Agent": "hcmc-vitality-research/1.0 (academic; coverage audit)"}


def poly_filter() -> str:
    g = gpd.read_file(REPO / "data/hcmc/raw/hcmc_boundary_full.gpkg")
    geom = g.geometry.union_all().simplify(0.002)
    coords = geom.exterior.coords
    return " ".join(f"{lat:.4f} {lon:.4f}" for lon, lat in coords)


# label -> Overpass statement, without the trailing (poly:...) which is added below
QUERIES = [
    # built form ------------------------------------------------------------
    ("building_all",        'way["building"]'),
    ("building_levels",     'way["building"]["building:levels"]'),
    ("building_height",     'way["building"]["height"]'),
    ("building_start_date", 'way["building"]["start_date"]'),
    ("building_typed",      'way["building"]["building"!~"^(yes|house)$"]'),
    # land use --------------------------------------------------------------
    ("landuse_all",         'way["landuse"]'),
    ("landuse_urban",       'way["landuse"~"^(residential|commercial|retail|industrial)$"]'),
    # street network --------------------------------------------------------
    ("highway_all",         'way["highway"]'),
    ("highway_named",       'way["highway"]["name"]'),
    ("highway_lit",         'way["highway"]["lit"]'),
    ("highway_sidewalk_tag", 'way["highway"]["sidewalk"]'),
    ("footway_sidewalk",    'way["footway"="sidewalk"]'),
    # transit ---------------------------------------------------------------
    ("bus_stop",            'node["highway"="bus_stop"]'),
    ("pt_platform",         'node["public_transport"="platform"]'),
    ("rail_station",        'node["railway"~"^(station|subway_entrance|halt)$"]'),
    # green and water -------------------------------------------------------
    ("park",                'way["leisure"="park"]'),
    ("green_landuse",       'way["landuse"~"^(grass|forest|meadow|village_green|recreation_ground)$"]'),
    ("water_natural",       'way["natural"="water"]'),
    ("waterway",            'way["waterway"]'),
    # destinations ----------------------------------------------------------
    ("poi_nodes",           'node["amenity"]'),
    ("poi_shop_nodes",      'node["shop"]'),
]


def run_batch(batch, poly):
    body = "[out:json][timeout:300];\n"
    for _, stmt in batch:
        body += f"{stmt}(poly:\"{poly}\");\nout count;\n"
    r = requests.post(URL, data={"data": body}, headers=UA, timeout=400)
    r.raise_for_status()
    elements = r.json()["elements"]
    if len(elements) != len(batch):
        raise RuntimeError(f"expected {len(batch)} counts, got {len(elements)}")
    return {label: int(el["tags"]["total"]) for (label, _), el in zip(batch, elements)}


def main():
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    poly = poly_filter()
    todo = [q for q in QUERIES if q[0] not in results]
    print(f"{len(results)} cached, {len(todo)} to fetch", flush=True)
    for i in range(0, len(todo), 4):
        batch = todo[i:i + 4]
        labels = ", ".join(l for l, _ in batch)
        for attempt in range(4):
            try:
                results.update(run_batch(batch, poly))
                print(f"  ok: {labels}", flush=True)
                break
            except Exception as e:
                wait = 30 * (attempt + 1)
                print(f"  retry {attempt+1} ({labels}): {type(e).__name__} {e}; sleep {wait}s", flush=True)
                time.sleep(wait)
        else:
            print(f"  FAILED: {labels}", flush=True)
        CACHE.write_text(json.dumps(results, indent=2))
        time.sleep(5)
    CACHE.write_text(json.dumps(results, indent=2))
    for label, _ in QUERIES:
        print(f"{label:24s} {results.get(label, 'MISSING')}")


if __name__ == "__main__":
    main()
