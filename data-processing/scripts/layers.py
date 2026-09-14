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
from scipy.ndimage import uniform_filter, zoom

# B7: every channel is computed at its own scale first and resampled last.
# A quantity defined over a 300 m ring is smooth at 1.17 m, so evaluating it at
# every unit records the same field more times at real cost.  SAMPLES_PER_SCALE
# fixes how finely each is computed before being read onto the common grid --
# four samples across its own measurement scale, matching the pilot's 17x17.
SAMPLES_PER_SCALE = 4.0

def native_grid(extent_m, scale_m, n, cap=None):
    """Grid size for a quantity whose measurement scale is scale_m."""
    if scale_m is None:
        return n                       # sharp: computed at the sampling unit
    k = int(np.ceil(SAMPLES_PER_SCALE * extent_m / scale_m))
    return max(8, min(n, cap or n, k))

def up(a, nx, ny):
    """Read a coarse field onto the ny x nx common grid."""
    if a.shape == (ny, nx):
        return a.astype("float32")
    z = zoom(a, (ny / a.shape[0], nx / a.shape[1]), order=1).astype("float32")
    return z[:ny, :nx]

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
    """Every layer on an nx by ny grid, each computed at its own scale first.

    The city mosaic runs this once over the whole study area, so no
    neighbourhood straddles a tile edge; the tile renders run it per tile at the
    sampling unit.  Same function either way.
    """
    ny = ny or nx
    ext = maxx - minx
    res = ext / nx
    out = {}

    ext_y = maxy - miny

    def grid(scale, cap=None):
        """Native grid for a quantity at `scale`, keeping the output aspect."""
        gx = native_grid(ext, scale, nx, cap)
        gy = max(8, min(ny, int(round(gx * ext_y / ext))))
        r = ext / gx
        ys, xs = np.mgrid[0:gy, 0:gx]
        return gx, gy, r, minx + (xs + 0.5) * r, maxy - (ys + 0.5) * (ext_y / gy)

    def counts(tree, scale, r_query, norm):
        gx, gy, _, px, py = grid(scale)
        k = tree.query_ball_point(np.c_[px.ravel(), py.ravel()], r_query, return_length=True)
        return up((k.reshape(gy, gx) / norm).astype("float32"), nx, ny)

    # --- sharp: defined by the polygon, so read at the sampling unit ----------
    b = ctx["burn_buildings"](minx, miny, maxx, maxy, nx, ny)
    out["building_binary"] = (b > 0).astype("float32")
    out["building_height"] = ctx["burn_heights"](minx, miny, maxx, maxy, nx, ny)

    # --- neighbourhood quantities: computed coarse, read fine -----------------
    gxD, gyD, rD, _, _ = grid(R_DENSITY)
    bD = (ctx["burn_buildings"](minx, miny, maxx, maxy, gxD, gyD) > 0).astype("float32")
    hD = ctx["burn_heights"](minx, miny, maxx, maxy, gxD, gyD)
    rdD = ctx["burn_roads"](minx, miny, maxx, maxy, gxD, gyD)
    out["building_density"] = up(_disc(bD, rD, R_DENSITY), nx, ny)
    out["far"] = up(_disc(hD / 3.0, rD, R_DENSITY), nx, ny)
    out["road_density"] = up(_disc(rdD, rD, R_DENSITY) / (rD / 1000.0), nx, ny)

    gxF, gyF, rF, _, _ = grid(R_FORM)
    out["building_form"] = up(_disc(ctx["burn_footprint_area"](minx, miny, maxx, maxy, gxF, gyF),
                                    rF, R_FORM), nx, ny)

    area_disc_km2 = np.pi * (R_DENSITY ** 2) / 1e6
    out["intersection"] = counts(ctx["junc_tree"], R_DENSITY, R_DENSITY, area_disc_km2)
    out["poi_count"] = counts(ctx["poi_tree"], R_RING, R_RING, np.pi * R_RING ** 2 / 1e6)

    gxE, gyE, _, pxE, pyE = grid(R_RING)
    out["poi_entropy"] = up(ctx["entropy"](pxE, pyE, R_RING), nx, ny)

    gxT, gyT, _, pxT, pyT = grid(R_TRANSIT)
    d, _ = ctx["stop_tree"].query(np.c_[pxT.ravel(), pyT.ravel()], k=8)
    out["transit_access"] = up(np.exp(-d / R_TRANSIT).sum(axis=1).reshape(gyT, gxT).astype("float32"), nx, ny)

    # --- surfaces and distances ----------------------------------------------
    gxG, gyG, _, _, _ = grid(10.0)
    out["green"] = up(ctx["read_green"](minx, miny, maxx, maxy, gxG, gyG), nx, ny)

    for name, key, scale in [("dist_water", "water_tree", 100.0),
                             ("dist_park", "park_tree", 300.0),
                             ("severance", "sever_tree", 300.0)]:
        t = ctx.get(key)
        gx, gy, _, px, py = grid(scale)
        if t is None:
            out[name] = np.full((ny, nx), np.nan, "float32")
        else:
            dd, _ = t.query(np.c_[px.ravel(), py.ravel()], k=1)
            out[name] = up(dd.reshape(gy, gx).astype("float32"), nx, ny)

    gxC, gyC, _, px, py = grid(1000.0)
    cx, cy = ctx["centre_utm"]
    out["dist_centre"] = up(np.hypot(px - cx, py - cy).astype("float32"), nx, ny)

    gxP, gyP, _, _, _ = grid(100.0)
    out["population"] = up(ctx["read_pop"](minx, miny, maxx, maxy, gxP, gyP), nx, ny)
    return out
