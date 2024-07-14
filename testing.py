import polars as pl
import datetime

start_time = pl.datetime(2024, 5, 1, 0, 0, 0)
end_time   = pl.datetime(2024, 5, 1, 23, 30, 0)

print(start_time)
print(end_time)

while start_time <= end_time:
    start_time += pl.duration(minutes=minu)
