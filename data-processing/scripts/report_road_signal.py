"""Do the road-derived factors carry within-tile structure, or are they conditions?

Road density, intersection density and space syntax are the three best-covered
BE factors in HCMC OSM.  Coverage alone does not make a grid channel: a layer
that is near-constant across a tile is a cell-level condition (B10's test).
This measures the field on a 32x32 sub-grid inside each fetched tile.
"""
import geopandas as gpd, numpy as np, pathlib

TILES = pathlib.Path("data/hcmc/raw/tiles")
idx = gpd.read_file("data/hcmc/raw/hcmc_tiles_z15.gpkg").set_index("cell_id")
blds = gpd.read_file("data/hcmc/raw/hcmc_buildings.gpkg").to_crs(3405)

N = 32
print(f"{'tile':18s} {'bld':>6s} {'road km':>8s} {'km/km2':>7s} {'nodes3+':>8s} "
      f"{'/km2':>6s} {'zero%':>6s} {'CV':>5s}")
rows = []
for d in sorted(TILES.iterdir()):
    f = d / "roads.gpkg"
    if not f.exists():
        continue
    tile = idx.loc[d.name]
    poly = gpd.GeoSeries([tile.geometry], crs=4326).to_crs(3405).iloc[0]
    roads = gpd.read_file(f).to_crs(3405)
    roads = roads[roads.intersects(poly)].copy()
    roads["geometry"] = roads.geometry.intersection(poly)
    roads = roads[~roads.is_empty]
    if roads.empty:
        continue

    minx, miny, maxx, maxy = poly.bounds
    area_km2 = poly.area / 1e6
    total_km = roads.length.sum() / 1000

    # intersections: endpoints shared by 3+ distinct road ways
    from collections import Counter
    ends = Counter()
    for geom in roads.geometry:
        parts = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
        for p in parts:
            cs = list(p.coords)
            for c in (cs[0], cs[-1]):
                ends[(round(c[0], 1), round(c[1], 1))] += 1
    nodes3 = sum(1 for v in ends.values() if v >= 3)

    # 32x32 field of road length density
    xs = np.linspace(minx, maxx, N + 1)
    ys = np.linspace(miny, maxy, N + 1)
    from shapely.geometry import box
    cells = [box(xs[i], ys[j], xs[i+1], ys[j+1]) for j in range(N) for i in range(N)]
    cg = gpd.GeoDataFrame(geometry=cells, crs=3405)
    j = gpd.sjoin(roads[["geometry"]], cg, how="inner", predicate="intersects")
    lens = np.zeros(len(cells))
    for ci, grp in j.groupby("index_right"):
        lens[ci] = grp.geometry.intersection(cells[ci]).length.sum()
    zero = (lens == 0).mean() * 100
    cv = lens.std() / lens.mean() if lens.mean() > 0 else float("nan")

    nb = len(blds[blds.intersects(poly)])
    print(f"{d.name:18s} {nb:6d} {total_km:8.1f} {total_km/area_km2:7.1f} "
          f"{nodes3:8d} {nodes3/area_km2:6.1f} {zero:5.0f}% {cv:5.2f}")
    rows.append((nb, total_km / area_km2, nodes3 / area_km2, zero, cv))

r = np.array([x for x in rows if x[0] >= 100])
if len(r):
    print(f"\nbuilt-up tiles only (n={len(r)}): road {r[:,1].mean():.1f} km/km2, "
          f"intersections {r[:,2].mean():.0f}/km2, empty sub-cells {r[:,3].mean():.0f}%, CV {r[:,4].mean():.2f}")
