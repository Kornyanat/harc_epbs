#!/usr/bin/env python3

import polars as pl
from datetime import datetime, timedelta

def split_datetime_range_by_day(start_dt: datetime, end_dt: datetime) -> list[tuple[datetime, datetime, str]]:
    """Split a datetime range into daily chunks with start and end datetimes."""
    result = []

    current_start = start_dt
    while current_start.date() < end_dt.date():
        # End of the current day
        current_end = datetime.combine(current_start.date(), datetime.max.time()).replace(microsecond=0)
        date_str = current_start.strftime('%Y-%m-%d')
        result.append((current_start, current_end, date_str))

        # Move to next day
        current_start = datetime.combine(current_start.date() + timedelta(days=1), datetime.min.time())

    # Add final day segment
    date_str = current_start.strftime('%Y-%m-%d')
    result.append((current_start, end_dt, date_str))
    return result

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