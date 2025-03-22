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

class GeoMagPlotter:
    def __init__(self, 
                 df: pl.DataFrame, 
                 min_lat: float, 
                 max_lat: float, 
                 min_lon: float, 
                 max_lon: float, 
                 altitude: float = 300, 
                 output_folder: str = 'output'):
        
        self.df = df
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.altitude = altitude
        self.output_folder = output_folder
    
    def geo_filter(self):
        """Filters the DataFrame based on given latitude and longitude."""
        return self.df.filter(
            (pl.col(f"geomag_lat_{self.altitude}").is_between(self.min_lat, self.max_lat)) &
            (pl.col(f"geomag_lon_{self.altitude}").is_between(self.min_lon, self.max_lon))
        )

    def create_map_subplot(self, ax):
        """Create map subplot."""
        # Ensure you're using Cartopy's GeoAxes with a PlateCarree projection
        ax.set_extent([self.min_lon, self.max_lon, self.min_lat, self.max_lat], crs=ccrs.PlateCarree())
        ax.set_facecolor("black")
        ax.add_feature(cfeature.COASTLINE, edgecolor="white")
        ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="white")
        ax.add_feature(cfeature.STATES, linestyle=":", edgecolor="white")

        # Filter data based on coordinates
        df_box = self.df
#        lats = df_box[f"geomag_lat_{self.altitude}"].to_numpy()
#        lons = df_box[f"geomag_lon_{self.altitude}"].to_numpy()
        lats = df_box['mid_lat'].to_numpy()
        lons = df_box['mid_long'].to_numpy()

        if len(lats) > 0:
            bins = 100
            heatmap, xedges, yedges = np.histogram2d(lons, lats, bins=bins, 
                                                     range=[[self.min_lon, self.max_lon], [self.min_lat, self.max_lat]])
            
            heatmap = np.log1p(heatmap)
            heatmap = gaussian_filter(heatmap, sigma=2)
            
            ax.imshow(heatmap.T, extent=[self.min_lon, self.max_lon, self.min_lat, self.max_lat],
                      origin='lower', cmap='inferno', alpha=0.7, aspect='auto')
        ax.set_title(f'Magnetic Coordinates at {self.altitude} km Altitude', fontsize=12)
        
    def create_dist_time_subplot(self, ax):
        """Create distance vs. time subplot with a log-scale colormap, with zero values in custom color."""
        df_box = self.df
        times = df_box["date"].dt.epoch("s").to_numpy()
        distances = df_box["dist_Km"].to_numpy()
    
        if len(times) > 0:
            time_bin_size = 120  # 2 minutes in seconds
            dist_bin_size = 50   # 50 km
    
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
    
            # Add colorbar
            cbar = plt.colorbar(img, ax=ax, orientation='vertical')
            cbar.set_label("Log Density", fontsize=10)
    
            # Labels and title
            ax.set_xlabel("Time (Unix Timestamp)", fontsize=10)
            ax.set_ylabel("Distance (km)", fontsize=10)
            ax.set_title("Distance vs Time (Log Scale)", fontsize=12)
    
    def plot(self):
        """Generate the combined plot."""
        fig = plt.figure(figsize=(24, 8))  
        
        # GridSpec with 1 row, 2 columns: Right subplot is WIDER
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])

        # Create subplots
        ax_map = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())  # Left map
        ax_dist_time = fig.add_subplot(gs[1])  # Right distance vs. time

        self.create_map_subplot(ax_map)
        self.create_dist_time_subplot(ax_dist_time)

        # Create the output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
        
        output_file = os.path.join(self.output_folder, f"geo_plot_{self.altitude}km.png")
        plt.tight_layout()
        plt.savefig(output_file, facecolor="white")
        plt.close()
        
        print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    # Example usage:
    df = pl.read_parquet("cache/df_gen/2017-07-01_lat-30_30_lon-100_-30_6.00MHz_15.00MHz_dist0_3000km_altitudes_100_300.parquet")
    #df = df.with_columns(pl.col('date').cast(pl.Datetime))
    
    min_lat, max_lat = -30, 30
    min_lon, max_lon = -100, -30
    altitude = 300  # Choose a specific altitude from the columns
    output_folder = "output"
    
    plotter = GeoMagPlotter(df, min_lat, max_lat, min_lon, max_lon, altitude, output_folder)
    plotter.plot()