#!/usr/bin/env python3

from datetime import date
import scripts 
from scripts.regions import REGIONS
from scripts.utils import *
from scripts.df_gen import *


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
        data_dir="../data/madrigal", 
        sDate=steps[0],
        eDate=steps[-1],
        region_name=region_name, 
        freq_range=freq_range,
        distance_range=distance_range,
        altitudes=altitude,  # Altitudes passed here
        use_cache=True
    )

    # Clear cache and load the dataframe (it will use the cache if available and `use_cache=True`)
    loader.clear_cache()  # Clear cache first
    df = loader.get_dataframe()  # Load the data

    # Print the processed data
    print(f"Processed data for {sDate} - {eDate}:")
    print(df)
    
    output_folder = "../output"
    
    plotter = GeoMagPlotter(df, region, altitude, output_folder)
    plotter.plot_1()
    plotter.plot_2()