#!/usr/bin/env python3

from datetime import date
import scripts 
from scripts.regions import REGIONS
from scripts.utils import *
from scripts.df_gen import *
from scripts.geomag_propagation_subplot import GeoMagPlotter
import os


if __name__ == "__main__":
    # Example usage:
#    df = pl.read_parquet("../cache/df_gen/2017-07-01 12:00:00_2017-07-01 23:59:59_6.00MHz_8.00MHz_dist0_20000km_0_100_300km.parquet")

    region_name = 'Equatorial America'
    region      = REGIONS['Equatorial America']
    altitude    = 0

    steps = get_sunset_time_steps(date(2017, 7, 1), region, altitude)
    for dt in steps:
        print(dt)

    freq_range = {
        'min_freq': 6000000,  # Example minimum frequency (0 MHz)
        'max_freq': 8000000  # Example maximum frequency (30 MHz)
    }

    distance_range = {
        'min_dist': 0,    # Example minimum distance in kilometers
        'max_dist': 20000  # Example maximum distance in kilometers
    }

    loader = HDF5PolarsLoader(
        data_dir="data/madrigal", 
        sDate=steps[0],
        eDate=steps[-1],
        region_name=region_name, 
        freq_range=freq_range,
        distance_range=distance_range,
        altitudes=altitude,  # Altitudes passed here
        use_cache=True
    )

    # Clear cache and load the dataframe (it will use the cache if available and `use_cache=True`)
    df = loader.get_dataframe()  # Load the data

    # Print the processed data
    print(f"Processed data for {steps[0]} - {steps[-1]}:")
    print(df)
    
    output_folder = "output"
    
    # Loop through each time step and create a new output folder for each section
    for i, start_dt in enumerate(steps[:-1]):
        end_dt = steps[i + 1]

        # Convert start_dt and end_dt to timezone-naive datetimes (remove timezone info)
        start_dt_naive = start_dt.replace(tzinfo=None)
        end_dt_naive = end_dt.replace(tzinfo=None)

        # Filter the Polars DataFrame for the current time step
        df_filtered = df.filter((df["date"] >= start_dt_naive) & (df["date"] < end_dt_naive))

        # Create a subfolder for the current time step
        subfolder = f"{start_dt_naive.strftime('%Y-%m-%d %H:%M:%S')}_{end_dt_naive.strftime('%Y-%m-%d %H:%M:%S')}"
        os.makedirs(f"{output_folder}/{subfolder}", exist_ok=True)

        # Plot and save for this specific time step
        plotter = GeoMagPlotter(df_filtered, region, altitude, f"{output_folder}/{subfolder}")
        plotter.plot_1()
        plotter.plot_2()