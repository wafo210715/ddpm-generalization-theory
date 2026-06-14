#!/usr/bin/env python3
"""Visualize POI spatial distribution: nodes vs ways."""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORT_DIR = "C:/sd/ddpm-generalization-theory/data-processing/data/hcmc/reports"

# Load data
nodes_lon, nodes_lat, nodes_type = [], [], []
with open(f"{REPORT_DIR}/_poi_nodes.csv", encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        nodes_lon.append(float(row['lon']))
        nodes_lat.append(float(row['lat']))
        nodes_type.append(row['type'])

ways_lon, ways_lat, ways_type = [], [], []
with open(f"{REPORT_DIR}/_poi_ways.csv", encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        ways_lon.append(float(row['lon']))
        ways_lat.append(float(row['lat']))
        ways_type.append(row['type'])

# Type colors
type_colors = {
    'amenity': '#e74c3c',
    'shop': '#3498db',
    'office': '#f39c12',
    'tourism': '#9b59b6',
    'leisure': '#2ecc71',
}

fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# --- Panel 1: Nodes only ---
ax = axes[0]
for ptype, color in type_colors.items():
    lon = [l for l, t in zip(nodes_lon, nodes_type) if t == ptype]
    lat = [l for l, t in zip(nodes_lat, nodes_type) if t == ptype]
    if lon:
        ax.scatter(lon, lat, s=0.3, c=color, alpha=0.6, label=f'{ptype} ({len(lon)})')
ax.set_title(f'POI Nodes: {len(nodes_lon):,}', fontsize=14, fontweight='bold')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend(fontsize=8, markerscale=10, loc='upper left')
ax.set_aspect('equal')

# --- Panel 2: Ways (centroids) only ---
ax = axes[1]
for ptype, color in type_colors.items():
    lon = [l for l, t in zip(ways_lon, ways_type) if t == ptype]
    lat = [l for l, t in zip(ways_lat, ways_type) if t == ptype]
    if lon:
        ax.scatter(lon, lat, s=1, c=color, alpha=0.6, label=f'{ptype} ({len(lon)})', marker='s')
ax.set_title(f'POI Ways (centroids): {len(ways_lon):,}', fontsize=14, fontweight='bold')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend(fontsize=8, markerscale=10, loc='upper left')
ax.set_aspect('equal')

# --- Panel 3: Combined with density comparison ---
ax = axes[2]
# Nodes as small dots
ax.scatter(nodes_lon, nodes_lat, s=0.2, c='#3498db', alpha=0.4, label=f'Nodes ({len(nodes_lon):,})')
# Ways as slightly larger squares
ax.scatter(ways_lon, ways_lat, s=1, c='#e74c3c', alpha=0.6, marker='s', label=f'Ways ({len(ways_lon):,})')
ax.set_title(f'Combined: {len(nodes_lon) + len(ways_lon):,} POIs', fontsize=14, fontweight='bold')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend(fontsize=9, markerscale=10, loc='upper left')
ax.set_aspect('equal')

# Add summary text
total = len(nodes_lon) + len(ways_lon)
node_pct = 100 * len(nodes_lon) / total
way_pct = 100 * len(ways_lon) / total
fig.suptitle(
    f'HCMC POI Spatial Distribution — Nodes vs Ways\n'
    f'BBox: (106.58, 10.70) to (106.85, 10.90) | Area: 653 km² | Density: {total/653:.0f} POIs/km²',
    fontsize=13, y=1.02
)

plt.tight_layout()
fig.savefig(f"{REPORT_DIR}/hcmc_poi_nodes_vs_ways.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {REPORT_DIR}/hcmc_poi_nodes_vs_ways.png")

# --- Also create a density heatmap comparison ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, (lons, lats, title) in zip(axes, [
    (nodes_lon, nodes_lat, f'Node POIs ({len(nodes_lon):,})'),
    (ways_lon, ways_lat, f'Way POIs ({len(ways_lon):,})'),
]):
    h = ax.hist2d(lons, lats, bins=60, cmap='hot_r', range=[[106.58, 106.85], [10.70, 10.90]])
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal')
    plt.colorbar(h[3], ax=ax, shrink=0.8, label='Count per cell')

fig.suptitle('POI Density Heatmaps — Nodes vs Ways', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(f"{REPORT_DIR}/hcmc_poi_density_comparison.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {REPORT_DIR}/hcmc_poi_density_comparison.png")
