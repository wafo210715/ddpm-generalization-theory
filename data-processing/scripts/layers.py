"""The indicator layers, each computed at its own measurement scale.

One module so the city mosaic, the tile renders and the marginals all call the
same function -- the same discipline the descriptor follows at B10b and E3.

Two scales per indicator, which the earlier preview collapsed and should not
have.  The MEASUREMENT SCALE is where a value is computed: a disc radius, a ring,
a decay length, or the polygon itself.  The SAMPLING UNIT is where it is read:
the pixel.  Reading a smooth field at more points records more samples of the
same field; it does not change what the field means.  So the radii below are
part of the indicator's definition, and the raster size is not.

Layers follow the "Have it" rows of discussion/hcmc_data_landscape.html.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import uniform_filter, distance_transform_edt

# measurement scales, metres
R_DENSITY   = 100.0    # disc for building / road / intersection density
R_RING      = 300.0    # ring for place count and entropy
R_TRANSIT   = 500.0    # decay length for transit accessibility
R_FORM      = 150.0    # disc for mean footprint size (grain)

LAYERS = [
    ("building_binary",  "BE", "is there a building on this unit"),
    ("building_density", "BE", "footprint area within 100 m"),
    ("building_height",  "BE", "height of the building here, metres"),
    ("far",              "BE", "floor area within 100 m over disc area"),
    ("building_form",    "BE", "mean footprint size within 150 m, metres squared"),
    ("road_density",     "BE", "street length within 100 m, km per km2"),
    ("intersection",     "BE", "junctions within 100 m, per km2"),
    ("transit_access",   "BE", "gravity to stops, 500 m decay"),
    ("poi_count",        "BE", "places within a 300 m ring, per km2"),
    ("poi_entropy",      "BE", "Shannon over functional classes in the ring"),
    ("green",            "BE", "vegetation fraction"),
    ("dist_water",       "BE", "distance to water, metres"),
    ("dist_park",        "BE", "distance to a park, metres"),
    ("dist_centre",      "BE", "distance to the historic centre, metres"),
    ("severance",        "BE", "distance to rail or motorway, metres"),
    ("population",       "HU", "residents per hectare"),
]
PENDING = [
    ("perception",       "HU", "needs the Mapillary pull and the vision model"),
    ("age_sex",          "HU", "needs the WorldPop age-sex rasters"),
]

CENTRE = (106.6983, 10.7769)     # Ben Thanh, the historic centre


def _disc(a, res, r):
    """Mean of `a` over a disc of radius r, approximated by a square box."""
    k = max(1, int(round(2 * r / res)))
    return uniform_filter(a.astype("float32"), size=k, mode="nearest")


def _tree_count(tree, px, py, r, shape):
    n = tree.query_ball_point(np.c_[px.ravel(), py.ravel()], r, return_length=True)
    return n.reshape(shape).astype("float32")


def compute(ctx, minx, miny, maxx, maxy, nx, ny=None):
    """Every layer on an nx by ny grid over the given bounds, in metres (EPSG:3405).

    The city mosaic runs this once over the whole study area, so the disc and
    ring neighbourhoods never straddle a tile edge; the tile renders run it per
    tile at full resolution.  Same function either way.
    """
    ny = ny or nx
    res = (maxx - minx) / nx
    ys, xs = np.mgrid[0:ny, 0:nx]
    px = minx + (xs + 0.5) * res
    py = maxy - (ys + 0.5) * res
    out = {}
    area_disc_km2 = np.pi * (R_DENSITY ** 2) / 1e6

    # --- built form, rasterized from polygons at the sampling unit -------------
    b = ctx["burn_buildings"](minx, miny, maxx, maxy, nx, ny)
    h = ctx["burn_heights"](minx, miny, maxx, maxy, nx, ny)
    out["building_binary"] = (b > 0).astype("float32")
    out["building_density"] = _disc(out["building_binary"], res, R_DENSITY)
    out["building_height"] = h
    out["far"] = _disc(h / 3.0, res, R_DENSITY)          # storeys, averaged over the disc
    out["building_form"] = _disc(ctx["burn_footprint_area"](minx, miny, maxx, maxy, nx, ny), res, R_FORM)

    # --- network ---------------------------------------------------------------
    r = ctx["burn_roads"](minx, miny, maxx, maxy, nx, ny)
    out["road_density"] = _disc(r, res, R_DENSITY) * (res / 1000.0) / (res * res / 1e6)
    out["intersection"] = _tree_count(ctx["junc_tree"], px, py, R_DENSITY, (ny, nx)) / area_disc_km2

    # --- destinations ----------------------------------------------------------
    out["poi_count"] = _tree_count(ctx["poi_tree"], px, py, R_RING, (ny, nx)) / (np.pi * R_RING ** 2 / 1e6)
    out["poi_entropy"] = ctx["entropy"](px, py, R_RING)

    # --- accessibility: a gravity field, not a presence mask -------------------
    d, _ = ctx["stop_tree"].query(np.c_[px.ravel(), py.ravel()], k=8)
    out["transit_access"] = np.exp(-d / R_TRANSIT).sum(axis=1).reshape(ny, nx).astype("float32")

    # --- surfaces and distances ------------------------------------------------
    out["green"] = ctx["read_green"](minx, miny, maxx, maxy, nx, ny)
    for name, key in [("dist_water", "water_tree"), ("dist_park", "park_tree"),
                      ("severance", "sever_tree")]:
        t = ctx.get(key)
        if t is None:
            out[name] = np.full((ny, nx), np.nan, "float32")
        else:
            dd, _ = t.query(np.c_[px.ravel(), py.ravel()], k=1)
            out[name] = dd.reshape(ny, nx).astype("float32")
    cx, cy = ctx["centre_utm"]
    out["dist_centre"] = np.hypot(px - cx, py - cy).astype("float32")

    out["population"] = ctx["read_pop"](minx, miny, maxx, maxy, nx, ny)
    return out
