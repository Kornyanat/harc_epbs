import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import numpy as np
import datetime
import bz2
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def filter_df(df, min_lat, max_lat, min_lon, max_lon, freq_high, freq_low):

  filtered_df = df[
    (df[23] >= min_lat) &
    (df[23] <= max_lat) &
    (df[24] >= min_lon) &
    (df[24] <= max_lon)
  ]

#  min_df = filtered_df[
#    (filtered_df['freq'] <= freq_high) &
#    (filtered_df['freq'] >= freq_low)
#  ]

  df = filtered_df[
    (filtered_df[22] <= 1000)
  ]

  return df

def filter_time(df, start_time, minu):

  # Define the start time for the two-minute chunk
  #start_time = pd.Timestamp('2017-11-03 18:00:00')

  # Define the end time for the two-minute chunk
  end_time = start_time + pd.Timedelta(minutes=minu)

  # Select the data within the two-minute range
  df = df[(df[0] >= start_time) & (df[0] < end_time)]

  return df

def split_bounding_box(min_lat, max_lat, min_lon, max_lon, lat_divisions, lon_divisions):
    lat_step = (max_lat - min_lat) / lat_divisions
    lon_step = (max_lon - min_lon) / lon_divisions

    boxes = []

    for i in range(lat_divisions):
        for j in range(lon_divisions):
            box_min_lat = min_lat + i * lat_step
            box_max_lat = min_lat + (i + 1) * lat_step
            box_min_lon = min_lon + j * lon_step
            box_max_lon = min_lon + (j + 1) * lon_step

            boxes.append((box_min_lat, box_max_lat, box_min_lon, box_max_lon))

    return boxes

def plot_boxes(boxes, df, freq_high, freq_low):


  # Create main grid of subplots using GridSpec
  fig = plt.figure(figsize=(20 * lon_divisions, 5 * lat_divisions))
  gs = gridspec.GridSpec(lat_divisions, lon_divisions)

  for idx, box in enumerate(boxes):
      box_min_lat, box_max_lat, box_min_lon, box_max_lon = box
      row_idx = idx // lon_divisions
      col_idx = idx % lon_divisions

      # Create a grid for each subplot spot with 1 row and 2 columns
      inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[row_idx, col_idx], width_ratios=[1, 3])

      # Create the map plot on the left
      ax_map = fig.add_subplot(inner_gs[0], projection=ccrs.PlateCarree())
      ax_map.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
      ax_map.add_feature(cfeature.COASTLINE)
      ax_map.add_feature(cfeature.BORDERS, linestyle=':')
      ax_map.add_feature(cfeature.STATES, linestyle=':')
      ax_map.plot([box_min_lon, box_max_lon, box_max_lon, box_min_lon, box_min_lon],
                  [box_min_lat, box_min_lat, box_max_lat, box_max_lat, box_min_lat],
                  color='red', transform=ccrs.PlateCarree())
      ax_map.set_title(f'Box {idx+1}\nLat: ({box_min_lat:.2f}, {box_max_lat:.2f}), Lon: ({box_min_lon:.2f}, {box_max_lon:.2f})')

      # Create the additional plot on the right

      df_p = filter_df(df, box_min_lat, box_max_lat, box_min_lon, box_max_lon, freq_high, freq_low)

      ax_additional = fig.add_subplot(inner_gs[1])
      ax_additional.hist2d(df_p[0], df_p[22], bins = 400)
      ax_additional.set_title('Additional Plot')
      ax_additional.set_xlabel('Longitude')
      ax_additional.set_ylabel('Latitude')

  plt.tight_layout()
  plt.show()

min_lat = 20
max_lat = 55
min_lon = -130
max_lon = -60
lat_divisions = 4
lon_divisions = 4
freq_high  = 15000
freq_low   = 13000
minu = 30

# Load the CSV file into a DataFrame
df = pd.read_csv('2024-05-01_PSK_14.csv', header=None)
df[0] = pd.to_datetime(df[0])

start_time = pd.Timestamp('2024-05-01 00:30:00')
df = filter_time(df, start_time, minu)

boxes = split_bounding_box(min_lat, max_lat, min_lon, max_lon, lat_divisions, lon_divisions)
plot_boxes(boxes, df, freq_high, freq_low)



