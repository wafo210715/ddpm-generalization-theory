"""The area data is fetched over is the union of the tiles, not the boundary.

Clipping source data to the municipal polygon leaves every tile that straddles
the boundary half empty: 885 of 5 170 tiles, 699 km2 of blank inside the tile
set.  A cell is a training example, and a cell with a straight empty edge
teaches the model that cities have straight empty edges.

So the boundary's only job is to choose which tiles are in the set.  Once chosen,
each tile is filled completely.  The study area becomes the union of those tiles
-- a pixelated version of the municipality -- plus a buffer, because indicators
measured over a ring or a disc read past the outer edge of the set.
"""
import pathlib
import geopandas as gpd
from shapely.ops import unary_union

REPO = pathlib.Path(__file__).resolve().parent.parent
RAW = REPO / "data/hcmc/raw"
BUFFER_M = 1000.0        # beyond the tile union, so edge neighbourhoods are real


def main():
    tiles = gpd.read_file(RAW / "hcmc_tiles_z15_2025.gpkg")
    bnd = gpd.read_file(RAW / "hcmc_boundary_2025.gpkg")
    tm = tiles.to_crs(3405)

    union = unary_union([g.buffer(0) for g in tm.geometry])
    fetch = union.buffer(BUFFER_M)

    gpd.GeoDataFrame({"name": ["tile union"]}, geometry=[union], crs=3405) \
        .to_crs(4326).to_file(RAW / "hcmc_tile_union.gpkg", driver="GPKG")
    gpd.GeoDataFrame({"name": ["tile union + buffer"]}, geometry=[fetch], crs=3405) \
        .to_crs(4326).to_file(RAW / "hcmc_fetch_area.gpkg", driver="GPKG")

    b_km2 = bnd.to_crs(3405).area.sum() / 1e6
    print(f"  municipality      : {b_km2:9,.1f} km2   (chooses which tiles)")
    print(f"  tile union        : {union.area/1e6:9,.1f} km2   ({len(tiles):,} whole tiles)")
    print(f"  union + {BUFFER_M:.0f} m buffer: {fetch.area/1e6:9,.1f} km2   (what data is fetched over)")
    print(f"  extra vs boundary : {union.area/1e6 - b_km2:9,.1f} km2   "
          f"({100*(union.area/1e6/b_km2 - 1):.1f} % more)")
    w, s, e, n = gpd.GeoSeries([fetch], crs=3405).to_crs(4326).total_bounds
    print(f"\n  fetch bbox: {w:.3f} {s:.3f} {e:.3f} {n:.3f}")


if __name__ == "__main__":
    main()
