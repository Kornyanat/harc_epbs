#!/usr/bin/env python3

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from scipy.ndimage import gaussian_filter
import matplotlib.gridspec as gridspec

def geo_filter(df, min_lat, max_lat, min_lon, max_lon):
    """Filters the DataFrame based on a given bounding box."""
    return df.filter(
        (pl.col("mid_lat").is_between(min_lat, max_lat)) &
        (pl.col("mid_long").is_between(min_lon, max_lon))
    )

def split_bounding_box(min_lat, max_lat, min_lon, max_lon, lat_divisions, lon_divisions):
    lat_step = (max_lat - min_lat) / lat_divisions
    lon_step = (max_lon - min_lon) / lon_divisions
    return [
        (min_lat + i * lat_step, min_lat + (i + 1) * lat_step,
         min_lon + j * lon_step, min_lon + (j + 1) * lon_step)
        for i in range(lat_divisions) for j in range(lon_divisions)
    ]

def plot_boxes(boxes, df, output_folder, lat_divisions, lon_divisions):
    fig = plt.figure(figsize=(20 * lon_divisions, 5 * lat_divisions))
    gs = gridspec.GridSpec(lat_divisions, lon_divisions, figure=fig)
    
    for idx, box in enumerate(boxes):
        box_min_lat, box_max_lat, box_min_lon, box_max_lon = box
        row_idx = idx // lon_divisions
        col_idx = idx % lon_divisions

        # Create a grid for each subplot spot with 1 row and 2 columns
        inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[row_idx, col_idx], width_ratios=[1, 3])
        
        ax_map = fig.add_subplot(inner_gs[0], projection=ccrs.PlateCarree())
        ax_dist_time = fig.add_subplot(inner_gs[1])
        
        df_box = geo_filter(df, box_min_lat, box_max_lat, box_min_lon, box_max_lon)
        
        lats = df_box["mid_lat"].to_numpy()
        lons = df_box["mid_long"].to_numpy()
        times = df_box["date"].dt.epoch("s").to_numpy()
        distances = df_box["dist_Km"].to_numpy()

        num_spots = len(df_box)
        buffer = 1.1
        ax_map.set_extent([box_min_lon - buffer, box_max_lon + buffer, 
                           box_min_lat - buffer, box_max_lat + buffer], crs=ccrs.PlateCarree())

        ax_map.set_facecolor("black")
        ax_map.add_feature(cfeature.COASTLINE, edgecolor="white")
        ax_map.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="white")
        ax_map.add_feature(cfeature.STATES, linestyle=":", edgecolor="white")

        ax_map.plot(
            [box_min_lon, box_max_lon, box_max_lon, box_min_lon, box_min_lon],
            [box_min_lat, box_min_lat, box_max_lat, box_max_lat, box_min_lat],
            color="cyan", linewidth=2, transform=ccrs.PlateCarree()
        )

        if len(lats) > 0:
            bins = 200
            heatmap, xedges, yedges = np.histogram2d(lons, lats, bins=bins, 
                                                     range=[[box_min_lon, box_max_lon], [box_min_lat, box_max_lat]])
            heatmap = np.log1p(heatmap)
            heatmap = gaussian_filter(heatmap, sigma=2)

            ax_map.imshow(heatmap.T, extent=[box_min_lon, box_max_lon, box_min_lat, box_max_lat],
                          origin='lower', cmap='inferno', alpha=0.7, aspect='auto')
        
        if len(times) > 0:
            time_bins = 200
            dist_bins = 200
            heatmap_dist_time, time_edges, dist_edges = np.histogram2d(times, distances, 
                                                                        bins=[time_bins, dist_bins])
            ax_dist_time.imshow(heatmap_dist_time.T, origin='lower', aspect='auto', cmap='viridis', alpha=0.7,
                                extent=[time_edges.min(), time_edges.max(), dist_edges.min(), dist_edges.max()])
            ax_dist_time.set_xlabel("Time (Unix Timestamp)", fontsize=10)
            ax_dist_time.set_ylabel("Distance (km)", fontsize=10)
            ax_dist_time.set_title(f"Distance vs Time (Box {idx+1})", fontsize=10)

        ax_map.set_aspect('equal')
        ax_map.tick_params(axis="both", colors="white")
        ax_map.set_title(f'Box {idx+1}\nLat: ({box_min_lat:.2f}, {box_max_lat:.2f})\nLon: ({box_min_lon:.2f}, {box_max_lon:.2f})',
                         fontsize=10)
        ax_map.text(0.5, -0.1, f'Total Spots: {num_spots}', ha='center', va='center', 
                    transform=ax_map.transAxes, fontsize=10)
    
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, 'heatmap_and_dist_time.png')
    plt.tight_layout()
    plt.savefig(output_file, facecolor="white")
    print(f"Plot saved to {output_file}")
    plt.close()
    
min_lat, max_lat = 20, 55
min_lon, max_lon = -130, -60
lat_divisions, lon_divisions = 4, 4
output_folder = "output"

df = pl.read_parquet("cache/df_gen/2017-07-01_lat-90_90_lon-180_180_1.00MHz_30.00MHz.parquet")
df = df.with_columns(pl.col('date').cast(pl.Datetime))

boxes = split_bounding_box(min_lat, max_lat, min_lon, max_lon, lat_divisions, lon_divisions)

plot_boxes(boxes, df, output_folder, lat_divisions, lon_divisions)
