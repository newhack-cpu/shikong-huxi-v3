"""Task A: Load and process UCI Beijing Multi-Site Air Quality Data.

Reads 12 CSV files from 12 monitoring stations, unifies column names,
handles missing values, and outputs a merged CSV + summary JSON.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json, os, glob, sys

DATA_DIR = Path("/root/autodl-tmp/data/uci_multisite")
OUTPUT_DIR = Path("/root/autodl-tmp/data")

# Column mapping: source column name → unified name
COLUMN_MAP = {
    "No": "row_id",
    "year": "year",
    "month": "month",
    "day": "day",
    "hour": "hour",
    "PM2.5": "pm25",
    "PM10": "pm10",
    "SO2": "so2",
    "NO2": "no2",
    "CO": "co",
    "O3": "o3",
    "TEMP": "temp",
    "PRES": "pres",
    "DEWP": "dewp",
    "RAIN": "rain",
    "wd": "wnd_dir",
    "WSPM": "wnd_spd",
}

KEEP_COLS = [
    "timestamp", "station", "pm25", "pm10", "so2", "no2", "co", "o3",
    "temp", "pres", "dewp", "rain", "wnd_dir", "wnd_spd"
]

STATION_LATLON = {
    "Aotizhongxin": (39.982, 116.397),
    "Wanliu": (39.987, 116.287),
    "Dongsi": (39.929, 116.417),
    "Tiantan": (39.886, 116.407),
    "Guanyuan": (39.929, 116.339),
    "Gucheng": (39.914, 116.184),
    "Huairou": (40.328, 116.628),
    "Nongzhanguan": (39.937, 116.461),
    "Shunyi": (40.127, 116.655),
    "Wanshouxigong": (39.878, 116.352),
    "Changping": (40.218, 116.220),
    "Dingling": (40.292, 116.220),
}


def extract_station_name(filename: str) -> str:
    """Extract station name from filename like PRSA_Data_Aotizhongxin_20130301-20170228.csv"""
    basename = os.path.basename(filename)
    parts = basename.replace(".csv", "").split("_")
    # PRSA_Data_Aotizhongxin_20130301-20170228 → station name is parts[2]
    if len(parts) >= 3:
        return parts[2]
    return parts[1] if len(parts) >= 2 else "unknown"


def load_single_csv(filepath: str) -> pd.DataFrame | None:
    """Load and clean a single station CSV."""
    station = extract_station_name(filepath)
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"  [ERROR] Failed to read {filepath}: {e}")
        return None

    print(f"  Loaded {station}: {len(df)} rows, {len(df.columns)} columns")

    # Rename columns
    df = df.rename(columns=COLUMN_MAP)

    # Build timestamp from year/month/day/hour
    df["timestamp"] = pd.to_datetime(
        df[["year", "month", "day", "hour"]], errors="coerce"
    )

    # Drop rows with invalid timestamps
    before = len(df)
    df = df.dropna(subset=["timestamp"])
    after = len(df)
    if before != after:
        print(f"    Dropped {before - after} rows with invalid timestamps")

    # Add station column
    df["station"] = station

    # Keep only relevant columns (those that exist in the dataframe)
    available_cols = ["timestamp", "station"] + [c for c in KEEP_COLS[2:] if c in df.columns]
    df = df[available_cols]

    # Convert numeric columns
    for col in available_cols:
        if col in ("timestamp", "station", "wnd_dir"):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def main():
    print("=" * 60)
    print("Task A: Loading UCI Beijing Multi-Site Air Quality Data")
    print("=" * 60)

    # Find all CSV files
    csv_files = sorted(glob.glob(str(DATA_DIR / "*.csv")))
    if not csv_files:
        # Try looking for the zip extraction
        print("No CSV files found directly. Checking for extracted directories...")
        for root, dirs, files in os.walk(str(DATA_DIR)):
            for f in files:
                if f.endswith(".csv"):
                    csv_files.append(os.path.join(root, f))
        csv_files = sorted(csv_files)

    if not csv_files:
        print("ERROR: No CSV files found in", DATA_DIR)
        print("Attempting to extract zip file...")
        import zipfile
        zip_files = list(Path(DATA_DIR).glob("*.zip"))
        if zip_files:
            print(f"Found zip: {zip_files[0]}")
            with zipfile.ZipFile(zip_files[0], 'r') as zf:
                zf.extractall(DATA_DIR)
            print("Extraction complete. Scanning again...")
            for root, dirs, files in os.walk(str(DATA_DIR)):
                for f in files:
                    if f.endswith(".csv"):
                        csv_files.append(os.path.join(root, f))
            csv_files = sorted(csv_files)
        else:
            print("No zip file found either. Exiting.")
            sys.exit(1)

    print(f"\nFound {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")

    # Load all CSVs
    all_dfs = []
    station_record_counts = {}

    for f in csv_files:
        station_name = extract_station_name(f)
        df = load_single_csv(f)
        if df is not None:
            all_dfs.append(df)
            station_record_counts[station_name] = len(df)

    if not all_dfs:
        print("ERROR: No valid data loaded.")
        sys.exit(1)

    # Merge all data
    merged = pd.concat(all_dfs, ignore_index=True)
    merged = merged.sort_values(["timestamp", "station"]).reset_index(drop=True)

    print(f"\nMerged dataset: {len(merged)} records from {len(all_dfs)} stations")

    # Handle missing values: station-level forward fill
    print("\nHandling missing values (station-level forward fill)...")
    missing_cols = [c for c in KEEP_COLS if c not in ("timestamp", "station", "wnd_dir") and c in merged.columns]

    for col in missing_cols:
        # Count missing before
        missing_before = merged[col].isna().sum()
        # Forward fill within each station
        merged[f"{col}_missing"] = merged[col].isna().astype(int)
        merged[col] = merged.groupby("station")[col].ffill()
        # Catch remaining NaN after ffill (in case all values in a group are NaN)
        if merged[col].isna().any():
            merged[col] = merged.groupby("station")[col].bfill()
        # Global fill for any remaining
        merged[col] = merged[col].fillna(merged[col].median())
        missing_after = merged[col].isna().sum()
        if missing_before > 0:
            print(f"  {col}: {missing_before} missing → {missing_after} remaining after fill")

    # After filling, drop any rows where all pollutant columns are still NaN
    pollutant_cols = ["pm25", "pm10", "so2", "no2", "co", "o3"]
    available_pollutants = [c for c in pollutant_cols if c in merged.columns]
    merged = merged.dropna(subset=available_pollutants, how="all")

    total_after_drop = len(merged)
    print(f"\nTotal records after dropping all-NaN rows: {total_after_drop}")

    # Build summary
    time_range = (merged["timestamp"].min(), merged["timestamp"].max())
    summary = {
        "n_stations": len(all_dfs),
        "station_names": sorted(station_record_counts.keys()),
        "total_records": int(total_after_drop),
        "time_range_start": str(time_range[0]),
        "time_range_end": str(time_range[1]),
        "time_range_days": int((time_range[1] - time_range[0]).total_seconds() / 86400),
        "per_station_records": station_record_counts,
        "per_station_latlon": STATION_LATLON,
    }

    # Save merged CSV
    output_csv = OUTPUT_DIR / "multisite_real.csv"
    print(f"\nSaving merged dataset to {output_csv}...")
    merged.to_csv(output_csv, index=False)
    print(f"  Saved {len(merged)} rows × {len(merged.columns)} columns")

    # Save summary JSON
    summary_path = OUTPUT_DIR / "multisite_summary.json"
    print(f"Saving summary to {summary_path}...")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    # Print report
    print("\n" + "=" * 60)
    print("TASK A COMPLETE — UCI Multi-Site Data Summary")
    print("=" * 60)
    print(f"  Stations:         {len(summary['station_names'])}")
    for s in summary["station_names"]:
        lat, lon = STATION_LATLON.get(s, (None, None))
        n_records = station_record_counts.get(s, 0)
        print(f"    {s:20s} ({lat:7.3f}, {lon:7.3f}) — {n_records:>8,} records")
    print(f"  Total records:    {summary['total_records']:,}")
    print(f"  Time range:       {summary['time_range_start']} → {summary['time_range_end']}")
    print(f"  Time span:        {summary['time_range_days']} days (~{summary['time_range_days']/365:.1f} years)")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    main()
