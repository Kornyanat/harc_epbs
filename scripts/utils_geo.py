#!/usr/bin/env python3

import polars as pl
from datetime import datetime
import apexpy

def add_geomagnetic_columns(df: pl.DataFrame, date_str: str, altitudes: list) -> pl.DataFrame:
    # Convert 'mid_lat' and 'mid_long' to lists
    geocen_lats = df['mid_lat'].to_list()
    geocen_lons = df['mid_long'].to_list()

    # Prepare to store results for each altitude
    for alt in altitudes:
        # Convert 'mid_lat' and 'mid_long' to geomagnetic coordinates for each altitude
        geomagnetic_lat, geomagnetic_lon = convert_geocentric_to_geomagnetic(geocen_lats, geocen_lons, alt, date_str)
        
        # Add new columns for geomagnetic latitude and longitude
        df = df.with_columns([
            pl.Series(f'geomag_lat_{alt}', geomagnetic_lat).cast(pl.Float32),
            pl.Series(f'geomag_lon_{alt}', geomagnetic_lon).cast(pl.Float32)
        ])

    return df  # Add return statement

def convert_geocentric_to_geomagnetic(geocen_lats: list, geocen_lons: list, alt: float, date_str: str) -> tuple[list, list]:
    # Convert date string to decimal year
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    decimal_year = date_obj.year + (date_obj.timetuple().tm_yday - 1) / 365.25
    
    # Initialize the Apex object with the decimal year
    apex_out = apexpy.Apex(date=decimal_year)

    # Convert geographic latitudes and longitudes to geomagnetic coordinates
    geomagnetic_lat, geomagnetic_lon = apex_out.convert(geocen_lats, geocen_lons, 'geo', 'apex', height=alt)
    
    return geomagnetic_lat, geomagnetic_lon  # Return as tuple

def convert_geomagnetic_to_geocentric(geomag_lats: list, geomag_lons: list, alt: float, date_str: str) -> tuple[list, list]:
    """Convert geomagnetic (apex) coordinates to geocentric (geographic) coordinates."""

    # Convert date string to decimal year
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    decimal_year = date_obj.year + (date_obj.timetuple().tm_yday - 1) / 365.25

    # Initialize the Apex object
    apex_out = apexpy.Apex(date=decimal_year)

    # Convert geomagnetic (apex) to geographic (geocentric) coordinates
    geocen_lat, geocen_lon = apex_out.convert(geomag_lats, geomag_lons, 'apex', 'geo', height=alt)

    return geocen_lat, geocen_lon

def calculate_source_percent(df: pl.DataFrame) -> dict:
    """Calculate the percentage of each data source in the given Polars DataFrame."""
    if df.is_empty():
        return {}

    # Convert 'source' column to string and count occurrences
    source_counts = df["source"].cast(str).value_counts().to_pandas()

    # Convert to dictionary
    source_counts_dict = dict(zip(source_counts["source"], source_counts["count"]))

    total_count = sum(source_counts_dict.values())

    # Compute percentages
    return {
        key: (value / total_count) * 100 if total_count > 0 else 0
        for key, value in source_counts_dict.items()
    }