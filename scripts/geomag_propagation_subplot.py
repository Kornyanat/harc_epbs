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
from utils_geo import *
from utils_plot import *

class GeoMagPlotter:
    def __init__(self, 
                 df: pl.DataFrame, 
                 region: dict, 
                 altitude: float = 300, 
                 output_folder: str = '..?/output'):
        
        self.df            = df
        self.region        = region
        self.altitude      = altitude
        self.output_folder = output_folder
        self.date_str      = self.df['date'].dt.strftime('%Y-%m-%d')[0]

#    def geo_filter(self):
#        """Filters the DataFrame based on given latitude and longitude."""
#        return self.df.filter(
#            (pl.col('mid_lat').is_between(*self.region['lat_lim'])) &
#            (pl.col('mid_long').is_between(*self.region['lon_lim']))
#        )
    
#    def mag_filter(self):
#        """Filters the DataFrame based on given latitude and longitude."""
#        lat, lon = convert_geocentric_to_geomagnetic(self.region['lat_lim'], self.region['lon_lim'], self.altitude, date_str='2017-07-01')

#        mag_region = {
#            'lat_lim': lat,  
#            'lon_lim': lon  
#        }
        
#        return self.df.filter(
#            (pl.col(f"geomag_lat_{self.altitude}").is_between(*mag_region['lat_lim'])) &
#            (pl.col(f"geomag_lon_{self.altitude}").is_between(*mag_region['lon_lim']))
#        )
    
    def plot_1(self):
        """Generate the combined plot."""
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, width_ratios=[1, 2, 0.05], height_ratios=[1, 1])
    
        ax_map = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
        ax_dist_time = fig.add_subplot(gs[0, 1])
        ax_spot = fig.add_subplot(gs[1, 1])
        ax_text = fig.add_subplot(gs[0, 0])
        ax_colorbar = fig.add_subplot(gs[0, 2])
    
        # Filtered data
#        df_geo = self.geo_filter()
#        df_mag = self.mag_filter()
    
        plot_map(ax_map, self.df, self.region)
        self.img = plot_dist_vs_lon(ax_dist_time, self.df, self.altitude)
        plot_spot_count(ax_spot, self.df, self.altitude)
        plot_text_box(ax_text, self.df)
    
        if self.img is not None:
            cbar = plt.colorbar(self.img, cax=ax_colorbar, orientation='vertical')
            cbar.set_label("Spot Count (Log Density)", fontsize=10)
    
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        os.makedirs(self.output_folder, exist_ok=True)
        output_file = os.path.join(self.output_folder, f"geo_plot_{self.altitude}km.png")
        plt.savefig(output_file, facecolor="white")
        plt.close()
        print(f"Plot saved to {output_file}")

    def plot_2(self):
        """Generate the combined plot."""
        fig = plt.figure(figsize=(5, 10))
        gs = gridspec.GridSpec(2, 1, width_ratios=[1], height_ratios=[1, 1])
    
        ax_globe = fig.add_subplot(gs[0, 0], projection=ccrs.Orthographic(central_longitude=-60, central_latitude=0))
        ax_map_geomag = fig.add_subplot(gs[1, 0])
    
        # Filtered data
#        df_geo = self.geo_filter()
#        df_mag = self.mag_filter()

        fig.suptitle(f"{self.date_str}", fontsize=16)
        
        plot_globe(ax_globe, self.df, self.region, self.date_str)
        plot_map_geomag(ax_map_geomag, self.df, self.region, self.altitude, self.date_str)
    
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        os.makedirs(self.output_folder, exist_ok=True)
        output_file = os.path.join(self.output_folder, f"geo_globe_plot_{self.altitude}km.png")
        plt.savefig(output_file, facecolor="white")
        plt.close()
        print(f"Plot saved to {output_file}")



if __name__ == "__main__":
    # Example usage:
    df = pl.read_parquet("../cache/df_gen/2017-07-01_lat-30_30_lon-100_-30_6.00MHz_8.00MHz_dist0_20000km_altitudes_0_100_300.parquet")

    region = {
        'lat_lim': [-30, 30],  
        'lon_lim': [-100, -30]  
    }
    
    altitude = 0
    output_folder = "../output"
    
    plotter = GeoMagPlotter(df, region, altitude, output_folder)
    plotter.plot_1()
    plotter.plot_2()