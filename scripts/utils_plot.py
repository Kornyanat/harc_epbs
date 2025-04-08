#!/usr/bin/env python3

import polars as pl
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.image import AxesImage
from matplotlib.colors import LogNorm
from matplotlib import cm
from scipy.ndimage import gaussian_filter
import numpy as np
import os
from utils_geo import *
from matplotlib.patches import Polygon


def plot_map(ax, df_box: pl.DataFrame, region: dict) -> None:
    """Plot the map heatmap of midpoints."""
    lat_min, lat_max = region['lat_lim']
    lon_min, lon_max = region['lon_lim']

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.set_facecolor("black")
    ax.add_feature(cfeature.COASTLINE, edgecolor="white")
    ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="white")
    ax.add_feature(cfeature.STATES, linestyle=":", edgecolor="white")

    lats = df_box['mid_lat'].to_numpy()
    lons = df_box['mid_long'].to_numpy()

    if len(lats) > 0:
        bins = 100
        heatmap, xedges, yedges = np.histogram2d(lons, lats, bins=bins,
                                                 range=[[lon_min, lon_max], [lat_min, lat_max]])

        heatmap = np.log1p(heatmap)
        heatmap = gaussian_filter(heatmap, sigma=2)

        ax.imshow(heatmap.T, extent=[lon_min, lon_max, lat_min, lat_max],
                  origin='lower', cmap='inferno', alpha=0.7, aspect='auto')

    ax.set_title('Spot Midpoint Locations (Geocentric Coordinates)', fontsize=12)


def plot_dist_vs_lon(ax, df_box: pl.DataFrame, altitude: float) -> AxesImage | None:
    """Plot distance vs. geomagnetic longitude heatmap."""
    times = df_box[f"geomag_lon_{altitude}"].to_numpy()
    distances = df_box["dist_Km"].to_numpy()

    if len(times) > 0:
        time_bin_size = 1
        dist_bin_size = 50

        time_edges = np.arange(times.min(), times.max() + time_bin_size, time_bin_size)
        dist_edges = np.arange(distances.min(), distances.max() + dist_bin_size, dist_bin_size)

        heatmap, _, _ = np.histogram2d(times, distances, bins=[time_edges, dist_edges])
        norm = LogNorm(vmin=np.nanmin(heatmap[heatmap > 0]), vmax=np.nanmax(heatmap))

        cmap = cm.viridis.copy()
        cmap.set_bad(color='black')

        img = ax.imshow(
            heatmap.T, origin='lower', aspect='auto', cmap=cmap, alpha=0.7,
            norm=norm, extent=[time_edges[0], time_edges[-1], dist_edges[0], dist_edges[-1]]
        )

        ax.set_xlabel("Geomagnetic Longitude", fontsize=10)
        ax.set_ylabel("Distance (km)", fontsize=10)
        ax.set_title("Distance vs Geomagnetic Longitude (Log Scale)", fontsize=12)

        return img
    return None


def plot_spot_count(ax, df_box: pl.DataFrame, altitude: float) -> None:
    """Plot total spot count vs. geomagnetic longitude."""
    lons = df_box[f"geomag_lon_{altitude}"].to_numpy()
    lon_bins = np.linspace(lons.min(), lons.max(), 50)
    spot_counts, _ = np.histogram(lons, bins=lon_bins)

    ax.plot(lon_bins[:-1], spot_counts, marker="o", linestyle='-', color='tab:blue')
    ax.set_xlabel("Geomagnetic Longitude", fontsize=10)
    ax.set_ylabel("Total Spot Count", fontsize=10)
    ax.set_title(f"Geomagnetic Longitude vs. Total Spot Counts at {altitude} km", fontsize=12)


def plot_text_box(ax, df: pl.DataFrame, date_col: str = "date") -> None:
    """Plot a text box with date and source info."""
    date_str = df[date_col][0].strftime("%Y-%m-%d") if len(df) > 0 else "No Data"
    percentages = calculate_source_percent(df)
    spot_count = df.height

    text_lines = [f"Date: {date_str}", f"Spot count: {spot_count}"]
    text_lines += [f"{source}: {percent:.1f}%" for source, percent in percentages.items()]
    text = "\n".join(text_lines)

    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12,
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
    ax.axis("off")


def plot_globe(ax, df, region, date_str):
    """Plot the globe with specific lat/lon points, a red dotted line, and overlay the map on top."""
    altitude=0
    # Plot the globe first
    # Plot a red dotted line along the equator (0° latitude)
    geomag_lons = np.linspace(-180, 180, 360)
    geomag_lats = np.linspace(0, 0, 360)
    line_lats, line_lons = convert_geomagnetic_to_geocentric(geomag_lats, geomag_lons, altitude, date_str)
    ax.plot(line_lons, line_lats, color='red', linestyle='dotted', linewidth=2, transform=ccrs.Geodetic(), alpha=0.75)

    ecuator_label_position = 148
    ecuator_height_adjust = 1
    # Add a label along the line — place it at longitude 0, latitude 2 to sit above the line
    ax.text(line_lons[ecuator_label_position], line_lats[ecuator_label_position] + ecuator_height_adjust, "0°", color='red', fontsize=10, ha='center',
            transform=ccrs.Geodetic())

    # Add gridlines and coastlines with orange landmass lines
    gl = ax.gridlines(draw_labels=True, color='gray', linewidth=0.5, linestyle='--')
    gl.xlocator = plt.MultipleLocator(30)  # Longitude lines every 30°
    gl.ylocator = plt.MultipleLocator(15)
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}
    
    # Modify the label positioning (move the labels inside the globe)
    gl.xlabels_top = False  # Hide the top labels
    gl.xlabels_bottom = True  # Show labels only at the bottom
    gl.ylabels_left = True  # Show labels on the left
    gl.ylabels_right = False  # Hide the right labels
    
    ax.coastlines(resolution='50m', color='blue', linewidth=1.5,alpha=0.5)

    # Show the whole globe (no extent set, so it's fully visible)
    ax.set_global()

    # Set the whole globe's background to white
    ax.set_facecolor('white')

    # Now, overlay the map (without limiting the globe's view)
    lat_min, lat_max = region['lat_lim']
    lon_min, lon_max = region['lon_lim']

    # Extract data for map plotting
    lats = df['mid_lat'].to_numpy()
    lons = df['mid_long'].to_numpy()

    if len(lats) > 0:
        bins = 100
        heatmap, xedges, yedges = np.histogram2d(lons, lats, bins=bins,
                                                 range=[[lon_min, lon_max], [lat_min, lat_max]])

        heatmap = np.log1p(heatmap)
        heatmap = gaussian_filter(heatmap, sigma=2)

        # Create a mask for the region
        mask = np.zeros_like(heatmap)
        mask[0:len(heatmap), 0:len(heatmap[0])] = 1  # Apply mask within region

        # Overlay the heatmap with solid black background inside the region
        ax.imshow(heatmap.T * mask.T, extent=[lon_min, lon_max, lat_min, lat_max],
                  origin='lower', cmap='inferno', alpha=1.0, aspect='auto', transform=ccrs.PlateCarree())

    ax.set_title('Spot Midpoints (Geographic Coordinates)', fontsize=14)

def plot_map_geomag(ax, df_box: pl.DataFrame, region: dict, alt: float, date_str: str) -> None:
    """Plot the map heatmap of midpoints in geomagnetic coordinates."""
    latlim, lonlim = convert_geocentric_to_geomagnetic(region['lat_lim'], region['lon_lim'], alt, date_str)
    lat_min, lat_max = latlim
    lon_min, lon_max = lonlim

    lats = df_box[f"geomag_lat_{alt}"].to_numpy()
    lons = df_box[f"geomag_lon_{alt}"].to_numpy()

#    lat_min = min(latlim[0], lats.min())
#    lat_max = max(latlim[1], lats.max())
#    lon_min = min(lonlim[0], lons.min())
#    lon_max = max(lonlim[1], lons.max())

    if len(lats) > 0:
        bins = 100
        heatmap, xedges, yedges = np.histogram2d(lons, lats, bins=bins,
                                                 range=[[lon_min, lon_max], [lat_min, lat_max]])

        # Mask zeros to show them as black
        heatmap = gaussian_filter(heatmap, sigma=2)
        heatmap_masked = np.ma.masked_where(heatmap == 0, np.log1p(heatmap) * 3)
#        heatmap_masked = np.ma.masked_where(heatmap == 0, np.log1p(heatmap))

        # Create custom colormap
        cmap = plt.cm.inferno.copy()
        cmap.set_bad(color='black')  # Set masked values (0s) to black

        ax.imshow(heatmap_masked.T, extent=[lon_min, lon_max, lat_min, lat_max],
                  origin='lower', cmap=cmap, alpha=1, aspect='auto')

    ax.set_title('Spot Midpoints (Geomagnetic Coordinates)', fontsize=14)
    ax.set_xlabel('Longitude (°)', fontsize=12)
    ax.set_ylabel('Latitude (°)', fontsize=12)
    
# Example usage for testing purposes:
if __name__ == "__main__":
    # Read the DataFrame
    df = pl.read_parquet("../cache/df_gen/2017-07-01_lat-30_30_lon-100_-30_0.00MHz_30.00MHz_dist0_20000km_altitudes_100_300.parquet")

    # Define region and altitude for plotting
    region = {
        'lat_lim': [-30, 30],  
        'lon_lim': [-100, -30]  
    }
    
    altitude = 300  # Example altitude

    # Create output folder if it doesn't exist
    output_folder = "../output"
    test_plots_folder = f"{output_folder}/tests"
    os.makedirs(test_plots_folder, exist_ok=True)

    # Plotting and saving individual plots
    # Plot Map
    fig1, ax1 = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    plot_map(ax1, df, region)
    fig1.savefig(f"{test_plots_folder}/plot_map.png")
    print(f"Saved plot_map.png")

    # Plot Distance vs Longitude
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    plot_dist_vs_lon(ax2, df, altitude)
    fig2.savefig(f"{test_plots_folder}/plot_dist_vs_lon.png")
    print(f"Saved plot_dist_vs_lon.png")

    # Plot Spot Count
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    plot_spot_count(ax3, df, altitude)
    fig3.savefig(f"{test_plots_folder}/plot_spot_count.png")
    print(f"Saved plot_spot_count.png")

    # Plot Text Box
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    plot_text_box(ax4, df)
    fig4.savefig(f"{test_plots_folder}/plot_text_box.png")
    print(f"Saved plot_text_box.png")

    # Plot Globe
    fig5, ax5 = plt.subplots(figsize=(6, 6), subplot_kw={'projection': ccrs.Orthographic(
        central_longitude=-60, central_latitude=0)})
    
    # Call the plot_globe function
    plot_globe(ax5, df, region, date_str='2017-03-01')
    
    # Save the plot
    fig5.savefig(f"{test_plots_folder}/plot_globe.png")
    print(f"Saved plot_globe.png")

    fig6, ax6 = plt.subplots(figsize=(8, 6))
    plot_map_geomag(ax6, df, region, altitude, date_str='2017-07-01')
    fig6.savefig(f"{test_plots_folder}/plot_map_geomag.png")
    print(f"Saved plot_map_geomag.png")