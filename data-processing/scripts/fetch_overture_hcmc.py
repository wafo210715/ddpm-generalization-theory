"""Pull the Overture Maps layers for HCMC and clip them to the study area.

Overture merges Meta, Microsoft, Esri and OSM.  For HCMC that matters enormously:
OSM holds 19,670 POIs and 86,418 building footprints, while Overture returns
431,295 places and about 4.1 million buildings over the same bounding box.  The
POI sparsity that B6 measured is an OSM artefact, not a fact about the city.

Writes GeoParquet to data/hcmc/raw/overture/<layer>.parquet, clipped to the
municipality polygon.  Queried anonymously over public S3; no credentials.
"""
import pathlib, sys
import duckdb

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / "data/hcmc/raw/overture_tu"
BOUNDARY = REPO / "data/hcmc/raw/hcmc_fetch_area.gpkg"   # the tile union, not the municipality
REL = "s3://overturemaps-us-west-2/release/2026-08-19.0"
# the 2025 municipality: mainland plus Con Dao
W, S, E, N = 106.05, 8.50, 107.62, 11.56

LAYERS = {
    "places":    ("theme=places/type=place",       "id, names.primary AS name, categories.primary AS category, categories.alternate AS categories_alt, confidence, geometry"),
    "buildings": ("theme=buildings/type=building", "id, subtype, class, height, num_floors, roof_shape, geometry"),
    # Added 23 Sep (issue #2, round 1): which provider drew each outline.  Overture
    # conflates OSM, Esri, Google and Microsoft in a fixed order, and Google's
    # confidence separates its high- and low-precision tiers.  Same filter as
    # "buildings", so the ids match row for row.
    "building_sources": ("theme=buildings/type=building",
                         "id, sources[1].dataset AS source, sources[1].confidence AS source_confidence, "
                         "sources[1].update_time AS source_update_time, len(sources) AS n_sources"),
    "land_use":  ("theme=base/type=land_use",      "id, subtype, class, names.primary AS name, geometry"),
    "land_cover":("theme=base/type=land_cover",    "id, subtype, cartography.min_zoom AS min_zoom, geometry"),
    "water":     ("theme=base/type=water",         "id, subtype, class, names.primary AS name, geometry"),
    "infrastructure": ("theme=base/type=infrastructure", "id, subtype, class, names.primary AS name, geometry"),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.sql("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial;")
    con.sql("SET s3_region='us-west-2'; SET enable_progress_bar=false;")
    con.sql(f"CREATE TABLE boundary AS SELECT ST_SetCRS(ST_Union_Agg(geom), 'OGC:CRS84') AS g FROM ST_Read('{BOUNDARY}')")

    only = sys.argv[1:] or list(LAYERS)
    for name in only:
        path, cols = LAYERS[name]
        dst = OUT / f"{name}.parquet"
        if dst.exists():
            print(f"{name}: exists, skipping", flush=True)
            continue
        print(f"{name}: querying ...", flush=True)
        con.sql(f"""
            COPY (
              SELECT {cols}
              FROM read_parquet('{REL}/{path}/*', filename=false, hive_partitioning=1) o,
                   boundary b
              WHERE o.bbox.xmin BETWEEN {W} AND {E}
                AND o.bbox.ymin BETWEEN {S} AND {N}
                AND ST_Intersects(o.geometry, b.g)
            ) TO '{dst}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        n = con.sql(f"SELECT count(*) FROM read_parquet('{dst}')").fetchone()[0]
        print(f"  {name}: {n:,} features -> {dst.stat().st_size/1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
