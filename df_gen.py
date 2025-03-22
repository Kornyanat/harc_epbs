#!/usr/bin/env python3

import shutil
import dask.dataframe as dd
import polars as pl
import pandas as pd
from datetime import datetime
import apexpy
import logging
from pathlib import Path
from dask.diagnostics import ProgressBar

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HDF5PolarsLoader:
    def __init__(self, 
                 data_dir: str, 
                 date_str: str, 
                 cache_dir: str = "cache/df_gen", 
                 use_cache: bool = True, 
                 region: tuple = None, 
                 freq_range: tuple = None, 
                 distance_range: tuple = None, 
                 chunk_size: int = 10000, 
                 altitudes: list = None):
        """
        :param data_dir: Directory where the HDF5 files are stored.
        :param date_str: The date in the format 'YYYY-MM-DD' to construct the file name.
        :param cache_dir: Directory to store cached files.
        :param use_cache: Whether to load from cache if available.
        :param region: Region filter for data, containing 'lat_lim' and 'lon_lim'.
        :param freq_range: Frequency range filter for data, containing 'min_freq' and 'max_freq'.
        :param chunk_size: Chunk size for reading the HDF5 file.
        :param altitudes: List of altitudes to calculate geomagnetic coordinates.
        """
        self.data_dir       = Path(data_dir)
        self.date_str       = date_str
        self.cache_dir      = Path(cache_dir)
        self.use_cache      = use_cache
        self.region         = region
        self.freq_range     = freq_range
        self.distance_range = distance_range
        self.chunk_size     = chunk_size
        self.altitudes      = altitudes or [300, 400, 500]  # Default altitudes if not provided

        # Construct dynamic cache path
        if self.region:
            lat_min, lat_max = self.region['lat_lim']
            lon_min, lon_max = self.region['lon_lim']
            region_str = f"lat{lat_min}_{lat_max}_lon{lon_min}_{lon_max}"
        else:
            region_str = "full_region"
        
        if self.freq_range:
            min_freq_mhz = self.freq_range['min_freq'] / 1_000_000
            max_freq_mhz = self.freq_range['max_freq'] / 1_000_000
            freq_str = f"{min_freq_mhz:.2f}MHz_{max_freq_mhz:.2f}MHz"
        else:
            freq_str = "full_freq_range"
    
        if self.distance_range:
            min_dist = self.distance_range['min_dist']
            max_dist = self.distance_range['max_dist']
            distance_str = f"dist{min_dist}_{max_dist}km"
        else:
            distance_str = "full_distance_range"
    
        altitudes_str = '_'.join([str(alt) for alt in self.altitudes])
    
        # cache path
        self.cache_path = self.cache_dir / f"{self.date_str}_{region_str}_{freq_str}_{distance_str}_altitudes_{altitudes_str}.parquet"
        self.df = None
        self.log = logging.getLogger(__name__)
    
        self.cache_dir.mkdir(parents=True, exist_ok=True)


    def get_file_path(self):
        """Construct the file path based on the date string."""
        file_name = f"rsd{self.date_str}.01.hdf5"
        return self.data_dir / file_name

    def load_data(self):
        """Load the dataset from cache or process the HDF5 file using Dask and convert to polars df."""
        if self.use_cache and self.cache_path.exists():
            self.log.info(f"Loading data from cache for {self.date_str}...")
            self.df = pl.read_parquet(self.cache_path)
            return self.df

        file_path = self.get_file_path()
        self.log.info(f"Loading data from HDF5 file {file_path}...")        
        
        # Load only the required columns from the HDF5 file
        dask_df = dd.read_hdf(file_path, key="Data/Table Layout", chunksize=self.chunk_size)

        required_columns = ['year', 'month', 'day', 'hour', 'min', 'sec', 'pthlen', 'rxlat', 'rxlon', 'txlat', 'txlon', 'tfreq', 'latcen', 'loncen', 'ssrc']
        dask_df = dask_df[required_columns]
        
        # Downcast columns to lower precision where appropriate
        dask_df = dask_df.astype({
            'rxlat': 'float32',  # Downcast latitude to float32
            'rxlon': 'float32',  # Downcast longitude to float32
            'txlat': 'float32',  # Downcast latitude to float32
            'txlon': 'float32',  # Downcast longitude to float32
            'latcen': 'float32',
            'loncen': 'float32',
            'ssrc': 'category',
            'tfreq': 'float32',  # Downcast frequency to float32
            'pthlen': 'int32'    # Downcast distance to int32 (as you're only interested in distances under 8000 km)
        })
        
        # Apply filters to the data (if any) before proceeding with the rest of the steps
        if self.region:
            dask_df = dask_df.map_partitions(self.apply_region_filter)
        if self.freq_range:
            dask_df = dask_df.map_partitions(self.apply_freq_filter)
        if self.distance_range:
            dask_df = dask_df.map_partitions(self.apply_distance_filter)
        
        # Continue processing and converting to pandas (or directly to Polars)
        with ProgressBar():
            df = dask_df.compute()
        
        # Renaming and processing the dataframe as before
        df['occurred'] = pd.to_datetime(df['year'] + '-' + df['month'] + '-' + df['day'] + ' ' + df['hour'] + ':' + df['min'] + ':' + df['sec'])
        df.drop(['year', 'month', 'day', 'hour', 'min', 'sec'], axis=1, inplace=True)
        
        df = df.rename(columns={"occurred": "date",
                                "pthlen": "dist_Km", 
                                "rxlat": "rx_lat", 
                                "rxlon": "rx_long", 
                                "txlat": "tx_lat", 
                                "txlon": "tx_long",
                                "tfreq": "freq",
                                "ssrc": "source",
                                "latcen": "mid_lat",
                                "loncen": "mid_long"})

        df['band'] = df['freq'].apply(self.get_band).astype('int8')
        
        df = df[['date', 'freq', 'band', 'dist_Km', 'source', 'mid_lat', 'mid_long', 'rx_lat', 'tx_lat', 'rx_long', 'tx_long']]
        
        # Convert to Polars
        df_polars = pl.from_pandas(df)
        
        # Apply geomagnetic conversion
        df_polars = convert_lat_lon_to_geomagnetic(df_polars, date_str=self.date_str, altitudes=self.altitudes)
        
        # Save to cache
        df_polars.write_parquet(self.cache_path, compression='snappy')
        self.df = df_polars
        return self.df

    def apply_region_filter(self, df):
        """Apply the region filter to the Dask DataFrame using the midpoint lat/lon columns."""
        if self.region:
            lat_lim, lon_lim = self.region['lat_lim'], self.region['lon_lim']
            df = df[(df['latcen'] >= lat_lim[0]) & (df['latcen'] < lat_lim[1])]
            df = df[(df['loncen'] >= lon_lim[0]) & (df['loncen'] < lon_lim[1])]
        return df

    def apply_freq_filter(self, df):
        """Apply the frequency filter to the Dask DataFrame using the 'tfreq' column."""
        if self.freq_range:
            min_freq, max_freq = self.freq_range['min_freq'], self.freq_range['max_freq']
            df = df[(df['tfreq'] >= min_freq) & (df['tfreq'] <= max_freq)]
        return df

    def apply_distance_filter(self, df):
        """Apply the distance filter to the Dask DataFrame using the 'pthlen' column."""
        if self.distance_range:
            min_dist, max_dist = self.distance_range['min_dist'], self.distance_range['max_dist']
            df = df[(df['pthlen'] >= min_dist) & (df['pthlen'] <= max_dist)]
        return df

    def get_band(self, frequency):
        """Assign a frequency band based on the value."""
        if 137 <= frequency < 2000000:          # 160 meters band (0.137 - 2 MHz)
            return 160
        elif 2000 <= frequency < 4000000:       # 80 meters band (2 - 4 MHz)
            return 80
        elif 4000 <= frequency < 7000000:       # 40 meters band (4 - 7 MHz)
            return 40
        elif 7000 <= frequency < 14000000:      # 20 meters band (7 - 14 MHz)
            return 20
        elif 14000 <= frequency < 21000000:     # 15 meters band (14 - 21 MHz)
            return 15
        elif 21000 <= frequency < 30000000:     # 10 meters band (21 - 30 MHz)
            return 10
        else:
            return 0  

    def get_dataframe(self):
        """Return the loaded Polars DataFrame."""
        if self.df is None:
            self.load_data()
        return self.df

    def clear_cache(self):
        """Delete all files in the cache directory."""
        if self.cache_dir.exists() and self.cache_dir.is_dir():
            for cache_file in self.cache_dir.iterdir():
                if cache_file.is_file() and cache_file.name.startswith(f"{self.date_str}_"):
                    cache_file.unlink()
                    self.log.info(f"Cache file removed: {cache_file}")
        else:
            self.log.info(f"Cache directory not found: {self.cache_dir}")


def convert_lat_lon_to_geomagnetic(df: pl.DataFrame, date_str: str, altitudes: list) -> pl.DataFrame:
    # Convert date string to decimal year
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    decimal_year = date_obj.year + (date_obj.timetuple().tm_yday - 1) / 365.25
    
    # Initialize the Apex object with the decimal year
    apex_out = apexpy.Apex(date=decimal_year)
    
    # Prepare to store results for each altitude
    for alt in altitudes:
        # Convert mid_lat and mid_long to geomagnetic coordinates for each altitude
        geomagnetic_lat, geomagnetic_lon = apex_out.convert(df['mid_lat'].to_list(), df['mid_long'].to_list(), 'geo', 'apex', height=alt)
        
        # Add new columns for geomagnetic latitude and longitude
        df = df.with_columns([
            pl.Series(f'geomag_lat_{alt}', geomagnetic_lat).cast(pl.Float32),
            pl.Series(f'geomag_lon_{alt}', geomagnetic_lon).cast(pl.Float32)
        ])
    
    return df

if __name__ == "__main__":
    
    # Example region and frequency range definitions
    region = {
        'lat_lim': [-30, 30],  # Latitude range: 30ºN to 30ºS
        'lon_lim': [-100, -30]  # Longitude range: 30ºW to 100ºW
    }

#    region = {
#    'lat_lim': [-90, 90],  # Latitude range: -90º (South Pole) to 90º (North Pole)
#    'lon_lim': [-180, 180]  # Longitude range: 180ºW to 180ºE (entire globe)
#    }
    
    freq_range = {
        'min_freq': 6000000,  # Example minimum frequency (6 MHz)
        'max_freq': 15000000  # Example maximum frequency (15 MHz)
    }

    distance_range = {
        'min_dist': 0,    # Example minimum distance in kilometers
        'max_dist': 3000  # Example maximum distance in kilometers
    }

    # Define the altitudes for conversion
    altitudes = [100,300]  # Altitudes in km

    # Define the date range you want to process (e.g., a list of dates)
    date_range = ["2017-07-01"]

    # Process each date
    for date_str in date_range:
        loader = HDF5PolarsLoader(
            data_dir="data/madrigal", 
            date_str=date_str, 
            region=region, 
            freq_range=freq_range,
            distance_range=distance_range,
            altitudes=altitudes,  # Altitudes passed here
            use_cache=True
        )

        # Clear cache and load the dataframe (it will use the cache if available and `use_cache=True`)
        loader.clear_cache()  # Clear cache first
        df = loader.get_dataframe()  # Load the data

        # Print the processed data
        print(f"Processed data for {date_str}:")
        print(df)
