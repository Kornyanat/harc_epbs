#!/usr/bin/env python3

import polars as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib import cm
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from scipy.ndimage import gaussian_filter
from utils import *

class GeoMagPlotter:
    def __init__(self, 
                 df: pl.DataFrame, 
                 region: dict, 
                 altitude: float = 300, 
                 output_folder: str = '..?/output'):
        
        self.df = df
        self.region = region
        self.altitude = altitude
        self.output_folder = output_folder

    def geo_filter(self):
        """Filters the DataFrame based on given latitude and longitude."""
        return self.df.filter(
            (pl.col('mid_lat').is_between(*self.region['lat_lim'])) &
            (pl.col('mid_long').is_between(*self.region['lon_lim']))
        )
    
    def mag_filter(self):
        """Filters the DataFrame based on given latitude and longitude."""
        lat, lon = convert_geocentric_to_geomagnetic(self.region['lat_lim'], self.region['lon_lim'], altitude, date_str='2017-07-01')

        mag_region = {
            'lat_lim': lat,  
            'lon_lim': lon  
        }
        
        return self.df.filter(
            (pl.col(f"geomag_lat_{self.altitude}").is_between(*mag_region['lat_lim'])) &
            (pl.col(f"geomag_lon_{self.altitude}").is_between(*mag_region['lon_lim']))
        )

    def create_map_subplot(self, ax):
        """Create map subplot."""
        lat_min, lat_max = self.region['lat_lim']
        lon_min, lon_max = self.region['lon_lim']

        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.set_facecolor("black")
        ax.add_feature(cfeature.COASTLINE, edgecolor="white")
        ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="white")
        ax.add_feature(cfeature.STATES, linestyle=":", edgecolor="white")

        # Filter data based on coordinates
        df_box = self.geo_filter()
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
        ax.set_title(f'Spot Midpoint Locations (Geocentric Coordinates)', fontsize=12)
        
    def create_dist_lon_subplot(self, ax):
        """Create distance vs. time subplot with a log-scale colormap, with zero values in custom color."""
        df_box = self.mag_filter()
        times = df_box[f"geomag_lon_{self.altitude}"].to_numpy()
        distances = df_box["dist_Km"].to_numpy()
    
        if len(times) > 0:
            time_bin_size = 1  
            dist_bin_size = 50   
    
            # Define bin edges
            time_edges = np.arange(times.min(), times.max() + time_bin_size, time_bin_size)
            dist_edges = np.arange(distances.min(), distances.max() + dist_bin_size, dist_bin_size)
    
            # Compute 2D histogram
            heatmap_dist_time, _, _ = np.histogram2d(times, distances, bins=[time_edges, dist_edges])
    
            # Normalize the colormap with log scaling
            norm = LogNorm(vmin=np.nanmin(heatmap_dist_time[heatmap_dist_time > 0]), vmax=np.nanmax(heatmap_dist_time))
    
            # Create a colormap where zeros are represented by a specific color
            cmap = cm.viridis  # Use the 'viridis' colormap
            cmap.set_bad(color='black')  # Set zero/empty values to black or any color you prefer
    
            # Plot the heatmap
            img = ax.imshow(
                heatmap_dist_time.T, origin='lower', aspect='auto', cmap=cmap, alpha=0.7,
                norm=norm, extent=[time_edges[0], time_edges[-1], dist_edges[0], dist_edges[-1]]
            )
    
    
            # Labels and title
            ax.set_xlabel("Geomagnetic Longitude", fontsize=10)
            ax.set_ylabel("Distance (km)", fontsize=10)
            ax.set_title("Distance vs Geomagnetic Longtitude (Log Scale)", fontsize=12)

        return img

    def create_spot_count_subplot(self, ax):
        """Create 2D plot for geomagnetic longitude vs. total spot counts."""
        # Filter data based on geomagnetic region
        df_box = self.mag_filter()
    
        # Get geomagnetic longitude at the given altitude
        lons = df_box[f"geomag_lon_{self.altitude}"].to_numpy()
    
        # Define the bin edges for geomagnetic longitude (adjust as needed)
        lon_bins = np.linspace(lons.min(), lons.max(), 50)  # 50 bins
    
        # Create histogram (count spots in each bin)
        spot_counts, _ = np.histogram(lons, bins=lon_bins)
    
        # Plotting the geomagnetic longitude vs. total spot counts
        ax.plot(lon_bins[:-1], spot_counts, marker="o", linestyle='-', color='tab:blue')  # Plot excluding last bin edge
        ax.set_xlabel("Geomagnetic Longitude", fontsize=10)
        ax.set_ylabel("Total Spot Count", fontsize=10)
        ax.set_title(f"Geomagnetic Longitude vs. Total Spot Counts at {self.altitude} km Altitude", fontsize=12)

    def create_text_box_subplot(self, ax):
        """Create a text box subplot with date and data source percentages."""
        df_box = self.df  # Use the entire dataset
    
        # Extract the date (assuming it's the same for all rows)
        date_str = df_box["date"][0].strftime("%Y-%m-%d") if len(df_box) > 0 else "No Data"
    
        percentages = calculate_source_percent(df_box)
        spot_count  = df_box.height
        
        text_lines  = [f"Date: {date_str}"]
        text_lines.append(f"Spot count: {spot_count}")
        for source, percent in percentages.items():
            text_lines.append(f"{source}: {percent:.1f}%")
    
        text_content = "\n".join(text_lines)
    
        # Display text inside subplot
        ax.text(0.5, 0.5, text_content, ha="center", va="center",
                fontsize=12, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
    
        ax.axis("off")  # Hide axes
    
    def plot(self):
        """Generate the combined plot."""
        fig = plt.figure(figsize=(16, 10))  # Larger figure size for better spacing
    
        # Create GridSpec with equal height ratios, adding an extra column for the colorbar
        gs = gridspec.GridSpec(2, 3, width_ratios=[1, 2, 0.05], height_ratios=[1, 1])
    
        # Create subplots
        ax_map = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())  # Left column
        ax_dist_time = fig.add_subplot(gs[0, 1])  # Top-right plot
        ax_new = fig.add_subplot(gs[1, 1])  # Bottom-right plot
        ax_textbox = fig.add_subplot(gs[0, 0])  # Top-left text box
        ax_colorbar = fig.add_subplot(gs[0, 2])  # Colorbar spans all rows in the last column
    
        # Create the plots
        self.create_map_subplot(ax_map)  # Plot the first map
        self.img = self.create_dist_lon_subplot(ax_dist_time)  # Distance vs. geomagnetic longitude
        self.create_spot_count_subplot(ax_new)  # Spot count plot
        self.create_text_box_subplot(ax_textbox)  # Text box
    
        # Add colorbar in its own subplot
        cbar = plt.colorbar(self.img, cax=ax_colorbar, orientation='vertical')
        cbar.set_label("Spot Count (Log Density)", fontsize=10)
    
        # Adjust layout: Use `tight_layout` and explicitly control spacing
        plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Adjust spaces between subplots
    
        # Save the plot
        os.makedirs(self.output_folder, exist_ok=True)
        output_file = os.path.join(self.output_folder, f"geo_plot_{self.altitude}km.png")
        plt.savefig(output_file, facecolor="white")
        plt.close()
    
        print(f"Plot saved to {output_file}")
    


if __name__ == "__main__":
    # Example usage:
    df = pl.read_parquet("../cache/df_gen/2017-07-01_lat-30_30_lon-100_-30_6.00MHz_15.00MHz_dist0_3000km_altitudes_100_300.parquet")

    region = {
        'lat_lim': [-30, 30],  
        'lon_lim': [-100, -30]  
    }
    
    altitude = 300 
     
    output_folder = "../output"
    
    plotter = GeoMagPlotter(df, region, altitude, output_folder)
    plotter.plot()