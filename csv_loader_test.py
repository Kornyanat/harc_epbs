import polars as pl

# Read CSV file
df = pl.read_csv("2024-05-01_PSK_fix.csv", has_header=False, ignore_errors=True)

# Print the first few rows
print(df.head())
