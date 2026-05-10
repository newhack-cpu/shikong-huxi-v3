# Spatial Analysis Reproducibility Guide

## Data Source

UCI Beijing Multi-Site Air Quality Data (501)
- Primary: https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip
- Mirror: https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip

12 monitoring stations across Beijing, hourly records from 2013-03-01 to 2017-02-28.
Station metadata (lat/lon) is embedded in `scripts/spatial_analysis.py`.

## One-Line Reproduction

```bash
python scripts/load_multisite.py && python scripts/spatial_analysis.py
```

**Requirements**: Python 3.10+, pandas, numpy, scipy, matplotlib, seaborn

## What Each Script Does

### `load_multisite.py`
1. Reads all 12 station CSV files from `data/uci_multisite/`
2. Unifies column names (PM2.5->pm25, TEMP->temp, etc.)
3. Builds datetime from year/month/day/hour columns
4. Station-level forward fill + backward fill for missing values
5. Outputs `data/multisite_real.csv` (420,768 records x 25 columns) and `data/multisite_summary.json`

### `spatial_analysis.py`
1. Loads `data/multisite_real.csv`
2. Pivots PM2.5 to wide format (timestamp x 12 stations)
3. Computes 12x12 Pearson correlation matrix
4. Computes pairwise Haversine distances using station lat/lon
5. Fits exponential decay: corr = a * exp(-d / d0)
6. Generates `paper_figures/fig8_spatial_correlation.png` and `paper_figures/fig9_distance_decay.png`
7. Outputs `results/spatial_analysis.json` and `results/spatial_pairs.csv`

## Key Deliverables

| File | Description |
|------|-------------|
| `data/multisite_real.csv` | Merged 420,768 records from 12 stations |
| `data/multisite_summary.json` | Per-station record counts, time range, lat/lon |
| `results/spatial_analysis.json` | d0=256 km, max/min/mean correlation, station pair stats |
| `results/spatial_pairs.csv` | All 66 station pair distances and correlations |
| `paper_figures/fig8_spatial_correlation.png` | 12x12 Pearson correlation heatmap |
| `paper_figures/fig9_distance_decay.png` | Distance vs correlation scatter + exponential fit |
| `scripts/load_multisite.py` | Data loading and preprocessing script |
| `scripts/spatial_analysis.py` | Spatial correlation analysis script |

## Appendix

`appendix/preliminary_experiments/` contains early zero-shot transfer results (not part of the main claim).

## Notes

- The primary zip URL may serve a nested archive; if `PRSA2017_Data_20130301-20170228.zip` appears inside, extract that to get the 12 CSV files.
- Missing PM2.5 rate is ~2.1% across the full dataset.
- Exponential decay fit: corr = 0.965 * exp(-d / 256.0 km), where d is inter-station distance.
