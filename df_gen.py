#!/usr/bin/env python3

import shutil
import dask.dataframe as dd
import polars as pl
import pandas as pd
import logging
from pathlib import Path
from dask.diagnostics import ProgressBar

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HDF5PolarsLoader:
    def __init__(self, 
                 data_dir, 
                 date_str, 
                 cache_dir="cache/df_gen", 
                 use_cache=True, 
                 region=None, 
                 freq_range=None, 
                 chunk_size=10000):
        """
        :param data_dir: Directory where the HDF5 files are stored.
        :param date_str: The date in the format 'YYYY-MM-DD' to construct the file name.
        :param cache_dir: Directory to store cached files.
        :param use_cache: Whether to load from cache if available.
        :param region: Region filter for data, containing 'lat_lim' and 'lon_lim'.
        :param freq_range: Frequency range filter for data, containing 'min_freq' and 'max_freq'.
        :param chunk_size: Chunk size for reading the HDF5 file.
        :param log: Custom logger to handle prints.
        """
        self.data_dir = Path(data_dir)
        self.date_str = date_str
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.region = region
        self.freq_range = freq_range
        self.chunk_size = chunk_size
        
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
        
        self.cache_path = self.cache_dir / f"{self.date_str}_{region_str}_{freq_str}.parquet"
        self.df = None
        self.log = logging.getLogger(__name__)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_file_path(self):
        """Construct the file path based on the date string."""
        file_name = f"rsd{self.date_str}.01.hdf5"
        return self.data_dir / file_name

    def load_data(self):
        """Load the dataset from cache or process the HDF5 file using Dask."""
        if self.use_cache and self.cache_path.exists():
            self.log.info(f"Loading data from cache for {self.date_str}...")
            self.df = pl.read_parquet(self.cache_path)
            return self.df

        file_path = self.get_file_path()

        self.log.info(f"Loading data from HDF5 file {file_path}...")
        dask_df = dd.read_hdf(file_path, key="Data/Table Layout", chunksize=self.chunk_size)
        
        if self.region:
            dask_df = dask_df.map_partitions(self.apply_region_filter)

        if self.freq_range:
            dask_df = dask_df.map_partitions(self.apply_freq_filter)

        with ProgressBar():
            df = dask_df.compute()

        df['occurred'] = pd.to_datetime(df['year'] + '-' + df['month'] + '-' + df['day'] + ' ' + df['hour'] + ':' + df['min'] + ':' + df['sec'])
        df.drop(['year', 'month', 'day', 'hour', 'min', 'sec'], axis=1, inplace=True)
        df["source"] = 1
        df = df.rename(columns={"pthlen": "dist_Km", 
                                "rxlat": "rx_lat", 
                                "rxlon": "rx_long", 
                                "txlat": "tx_lat", 
                                "txlon": "tx_long",
                                "tfreq": "freq",
                                })
        df['band'] = df['freq'].apply(self.get_band)

        df = df[['freq', 'band', 'occurred', 'rx_lat', 'rx_long', 'dist_Km']]
        df.rename(columns={"occurred": "date", "rx_lat": "mid_lat", "rx_long": "mid_long"}, inplace=True)

        df_polars = pl.from_pandas(df)

        df_polars.write_parquet(self.cache_path, compression='snappy')
        self.df = df_polars
        return self.df

    def apply_region_filter(self, df):
        """Apply the region filter to the Dask DataFrame using the original HDF5 columns."""
        if self.region:
            lat_lim, lon_lim = self.region['lat_lim'], self.region['lon_lim']
            df = df[(df['rxlat'] >= lat_lim[0]) & (df['rxlat'] < lat_lim[1])]
            df = df[(df['rxlon'] >= lon_lim[0]) & (df['rxlon'] < lon_lim[1])]
        return df

    def apply_freq_filter(self, df):
        """Apply the frequency filter to the Dask DataFrame using the 'tfreq' column."""
        if self.freq_range:
            min_freq, max_freq = self.freq_range['min_freq'], self.freq_range['max_freq']
            df = df[(df['tfreq'] >= min_freq) & (df['tfreq'] <= max_freq)]
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

if __name__ == "__main__":
    # Example region and frequency range definitions

        
    
    region = {
        'lat_lim': [-90, 90],  # From South Pole to North Pole
        'lon_lim': [-180, 180]  # From West to East
    }
    freq_range = {
        'min_freq': 1000000,  # Example minimum frequency (1 MHz)
        'max_freq': 30000000  # Example maximum frequency (30 MHz)
    }

    # Define the date range you want to process (e.g., a list of dates)
    date_range = ["2017-07-01"]

    # Process each date
    for date_str in date_range:
        loader = HDF5PolarsLoader(data_dir="data/madrigal", 
                                  date_str=date_str, 
                                  region=region, 
                                  freq_range=freq_range, 
                                  use_cache=True)
        
        # Get the dataframe (it will use the cache if available and `use_cache=True`)
        df = loader.clear_cache()
        df = loader.get_dataframe()
        print(f"Processed data for {date_str}:")
        print(df)