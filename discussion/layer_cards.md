# Layer cards · what's decided and measured

| item | decision / finding | source |
|---|---|---|
| cells | 3 010 = 3 020 tiles with ≥ 100 Overture buildings − 10 on Côn Đảo | `build_b8.py cells` |
| pixel grid | each cell on its exact web-Mercator tile, 256 px = 4.69 × 4.66 m; no overlap between cells | `build_b8.py` |
| `building_binary` | 0/1, pixel-centre rule; built and checked: 9.09 % cover in raster and vector | `b8_256/layers/` |
| provider seam | footprints are 57 % Microsoft, 39 % Google, 4 % OSM; Google draws 0.45× Microsoft's size in the same cells. Affects every size-based measure; single provider or not is still open | `report_building_sources.py` |
| 256 px limit | cover is exact, but 2/3 of separate buildings fuse; half of all gaps are < 1.2 m roof seams, not alleys | `report_building_footprints.py` |
| wetland | the water layer counts 396 km² of wetland (mangrove) as water; open | `figure_base_map.py` |
