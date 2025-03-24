import matplotlib.pyplot as plt
import os
import datetime
import matplotlib as mpl
import numpy as np
from numpy import genfromtxt
import pandas as pd
import polars as pl
import datetime
import seaborn as sns
from scipy.stats import pearsonr
import itertools
from scipy.interpolate import make_interp_spline, BSpline
from scipy.interpolate import interp1d
import bz2
import cartopy
from geopy.distance import geodesic
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import binned_statistic_2d

def read_csv_bz2(csv_bz2_path):
    with bz2.open(csv_bz2_path, 'rt') as f: 
        return pd.read_csv(f) 
    
def gen_midpoint(df):
  df['mid_lat'] = (df['tx_lat'] + df['rx_lat']) / 2
  df['mid_lon'] = (df['tx_long'] + df['rx_long']) / 2
  return df

def filter_df(df, min_lat, max_lat, min_lon, max_lon, freq_high, freq_low):
    # Step 1: Filter based on latitude and longitude
    filtered_df = df.filter(
        (pl.col('column_24') >= min_lat) & 
        (pl.col('column_24') <= max_lat) & 
        (pl.col('column_25') >= min_lon) & 
        (pl.col('column_25') <= max_lon)
    )

    # Step 2: Filter based on frequency range
#    min_df = filtered_df.filter(
#        (pl.col('freq') <= freq_high) & 
#        (pl.col('freq') >= freq_low)
#    )

    # Step 3: Filter based on distance
    df = filtered_df.filter(pl.col('column_23') <= 1000)

    return df

def filter_time(df, start_time, minu):

  df = df.with_columns(pl.col('column_1').str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S'))
  
  # Define the start time for the two-minute chunk
  #start_time = pd.Timestamp('2017-11-03 18:00:00')

  # Define the end time for the two-minute chunk
  end_time = start_time + pl.duration(minutes=minu)

  # Select the data within the two-minute range
  df = df.filter((pl.col('column_1') >= start_time) & (pl.col('column_1') < end_time))

  return df

def gen_mesh_data(df, num_bins, min_lat, max_lat, min_lon, max_lon, start_time, freq_high, freq_low, minu):

  # filtering
  #df = gen_midpoint(df)
  df = filter_time(df, start_time, minu)
  df = filter_df(df, min_lat, max_lat, min_lon, max_lon, freq_high, freq_low)

  # Pull tx and rx midpoint data
  lat_data  = df.get_column('column_24')
  lon_data  = df.get_column('column_25')
  dist_data = df.get_column('column_23')

  # Adjust these values to change the resolution
  num_bins_lon = num_bins  # Number of bins in the longitude direction
  num_bins_lat = num_bins  # Number of bins in the latitude direction

  # Create the bins for longitude and latitude
  lon_bins = np.linspace(min_lon, max_lon, num_bins_lon + 1)
  lat_bins = np.linspace(min_lat, max_lat, num_bins_lat + 1)

  # Determine the values at the 10th and 30th percentiles of dist_data
  #percentile_10 = np.percentile(dist_data, 10)
  #percentile_30 = np.percentile(dist_data, 30)
  #percentile = np.percentile(dist_data, 20)

  # Create masks for data between 10th and 30th percentiles
  #mask_10_to_30 = (dist_data >= percentile_10) & (dist_data <= percentile_30)
  #mask = (dist_data <= percentile)

  # Apply binned_statistic_2d to the masked data
  #statistic, lon_edges, lat_edges, _ = binned_statistic_2d(
  #  lon_data[mask], lat_data[mask], dist_data[mask],
  #  statistic='mean', bins=[lon_bins, lat_bins]
  #)
  
  statistic, lon_edges, lat_edges, _ = binned_statistic_2d(
    lon_data, lat_data, dist_data,
    statistic='mean', bins=[lon_bins, lat_bins]
  )

  return lon_edges, lat_edges, statistic

def plot_map(lon_edges, lat_edges, statistic, min_lat, max_lat, min_lon, max_lon, filename, start_time, start_time_name):

  # Create the plot
  fig, ax = plt.subplots(figsize=(20, 15), subplot_kw={'projection': ccrs.PlateCarree()})

  # Add features to the map
  ax.add_feature(cfeature.COASTLINE)
  ax.add_feature(cfeature.BORDERS)
  ax.add_feature(cfeature.STATES, linestyle=':')

  # Set extent to cover the continental US
  ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

  # Use pcolormesh to plot the aggregated data
  # Note: lon_edges and lat_edges define the bin edges; statistic defines the bin values
  mesh = ax.pcolormesh(lon_edges, lat_edges, statistic.T, transform=ccrs.PlateCarree(), cmap='viridis')

  # Add a colorbar
  cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, aspect=50)
  cbar.set_label('Average Distance')

  # Add title
  plt.title('US Map with Aggregated Dist Data ' + str(start_time_name))

  # Show the plot
  plt.savefig(filename)
  plt.close()
  
  
num_bins   = 50
min_lat    = 20
max_lat    = 50
min_lon    = -130
max_lon    = -60
start_time_pl = pl.datetime(2024, 5, 1, 0, 0, 0)
end_time_pl   = pl.datetime(2024, 5, 1, 23, 30, 0)
start_time = pd.Timestamp('2024-05-01 00:00:00')
end_time   = pd.Timestamp('2024-05-01 23:30:00')
freq_high  = 8000
freq_low   = 6000
minu       = 15
frames_dir = 'frames/'

df = pl.read_csv("2024-05-01_PSK_14.csv", has_header=False)

filter_time(df, start_time, minu)
# Generate plots

frame_number = 0
while start_time <= end_time:
    filename = os.path.join(frames_dir, f'plot_{frame_number:04d}.png')
    lon_edges, lat_edges, statistic = gen_mesh_data(df, num_bins, min_lat, max_lat, min_lon, max_lon, start_time_pl, freq_high, freq_low, minu)
    plot_map(lon_edges, lat_edges, statistic, min_lat, max_lat, min_lon, max_lon, filename, start_time_pl, start_time)
    start_time_pl += pl.duration(minutes=minu)
    start_time += pd.Timedelta(minutes=minu)
    frame_number += 1