"""Task B: Spatial Correlation Analysis of 12 Beijing Monitoring Stations.

Computes pairwise PM2.5 correlations, Haversine distances,
fits exponential decay model, and generates publication figures.
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit
from pathlib import Path
import json, warnings

warnings.filterwarnings("ignore")

BASE = Path("/root/autodl-tmp")
DATA_PATH = BASE / "data" / "multisite_real.csv"
RESULTS_DIR = BASE / "results"
FIG_DIR = BASE / "paper_figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

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


def haversine_km(lat1, lon1, lat2, lon2):
    """Compute Haversine distance between two lat/lon points in km."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
         * np.sin(dlon / 2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def main():
    print("=" * 60)
    print("Task B: Spatial Correlation Analysis")
    print("=" * 60)

    # Load data
    print(f"\nLoading {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    print(f"  Loaded {len(df):,} records, {df['station'].nunique()} stations")

    # Pivot PM2.5 to wide format: rows=timestamp, columns=station
    pm25_pivot = df.pivot_table(
        index="timestamp", columns="station", values="pm25", aggfunc="mean"
    )
    print(f"  Pivot shape: {pm25_pivot.shape[0]} timesteps × {pm25_pivot.shape[1]} stations")

    station_names = sorted(STATION_LATLON.keys())

    # ================================================================
    # 1. Pairwise Pearson correlation matrix
    # ================================================================
    print("\nComputing 12×12 Pearson correlation matrix...")
    corr_matrix = pm25_pivot[station_names].corr()
    print(f"  Correlation matrix shape: {corr_matrix.shape}")

    # ================================================================
    # 2. Pairwise Haversine distances
    # ================================================================
    print("\nComputing pairwise Haversine distances...")
    n_stations = len(station_names)
    dist_matrix = np.zeros((n_stations, n_stations))

    for i, s1 in enumerate(station_names):
        lat1, lon1 = STATION_LATLON[s1]
        for j, s2 in enumerate(station_names):
            lat2, lon2 = STATION_LATLON[s2]
            dist_matrix[i, j] = haversine_km(lat1, lon1, lat2, lon2)

    # ================================================================
    # 3. Extract upper triangle (excluding diagonal) for fitting
    # ================================================================
    print("\nExtracting distance-correlation pairs...")
    pairs = []
    for i in range(n_stations):
        for j in range(i + 1, n_stations):
            s1 = station_names[i]
            s2 = station_names[j]
            d = dist_matrix[i, j]
            r = corr_matrix.loc[s1, s2]
            pairs.append({
                "station_pair": f"{s1}-{s2}",
                "distance_km": round(d, 2),
                "correlation": round(r, 6),
            })

    pairs_df = pd.DataFrame(pairs)
    distances = pairs_df["distance_km"].values
    correlations = pairs_df["correlation"].values

    print(f"  {len(pairs)} station pairs (upper triangle)")
    print(f"  Distances: {distances.min():.1f} – {distances.max():.1f} km")
    print(f"  Correlations: {correlations.min():.4f} – {correlations.max():.4f}")

    # ================================================================
    # 4. Fit exponential decay: corr = a * exp(-d / d0)
    # ================================================================
    print("\nFitting exponential decay model: corr = a * exp(-d / d0)...")

    def exp_decay(d, a, d0):
        return a * np.exp(-d / d0)

    try:
        popt, pcov = curve_fit(
            exp_decay, distances, correlations,
            p0=[1.0, 50.0],
            bounds=([0.5, 1.0], [1.5, 500.0]),
            maxfev=10000,
        )
        a_fit, d0_fit = popt
        a_err, d0_err = np.sqrt(np.diag(pcov))
        print(f"  Fit result: a = {a_fit:.4f} ± {a_err:.4f}, d0 = {d0_fit:.1f} ± {d0_err:.1f} km")
    except Exception as e:
        print(f"  [WARNING] curve_fit failed: {e}")
        print("  Using simple log-linear regression fallback...")
        # Fallback: linear regression on log(corr) vs distance
        valid = correlations > 0
        log_corr = np.log(correlations[valid])
        d_valid = distances[valid]
        # log(corr) = log(a) - d/d0 → slope = -1/d0
        from numpy.polynomial.polynomial import polyfit
        coefs = polyfit(d_valid, log_corr, 1)
        d0_fit = -1.0 / coefs[1]
        a_fit = np.exp(coefs[0])
        print(f"  Fallback result: a = {a_fit:.4f}, d0 = {d0_fit:.1f} km")

    # ================================================================
    # 5. Find closest and farthest pairs
    # ================================================================
    pairs_df["abs_corr"] = pairs_df["correlation"].abs()
    closest_pair = pairs_df.loc[pairs_df["distance_km"].idxmin()]
    farthest_pair = pairs_df.loc[pairs_df["distance_km"].idxmax()]
    max_corr_pair = pairs_df.loc[pairs_df["correlation"].idxmax()]
    min_corr_pair = pairs_df.loc[pairs_df["correlation"].idxmin()]

    max_corr_val = correlations.max()
    min_corr_val = correlations.min()
    mean_corr_val = correlations.mean()

    print(f"\nCorrelation statistics:")
    print(f"  Max:  {max_corr_val:.4f} ({max_corr_pair['station_pair']}, {max_corr_pair['distance_km']:.1f} km)")
    print(f"  Min:  {min_corr_val:.4f} ({min_corr_pair['station_pair']}, {min_corr_pair['distance_km']:.1f} km)")
    print(f"  Mean: {mean_corr_val:.4f}")
    print(f"  Closest pair:  {closest_pair['station_pair']} ({closest_pair['distance_km']:.1f} km)")
    print(f"  Farthest pair: {farthest_pair['station_pair']} ({farthest_pair['distance_km']:.1f} km)")

    # ================================================================
    # 6. Generate figures
    # ================================================================
    print("\nGenerating figures...")

    # Use non-interactive backend
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # --- Fig 8: Spatial Correlation Heatmap ---
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.eye(n_stations, dtype=bool)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        vmin=0.3,
        vmax=1.0,
        mask=mask,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
        ax=ax,
    )
    ax.set_title(
        "Spatial Correlation Matrix of 12 Beijing Monitoring Stations\n"
        "(UCI Multi-Site, 2013-2017, n=420K hourly records)",
        fontsize=14, fontweight="bold", pad=20,
    )
    ax.set_xlabel("Station", fontsize=12)
    ax.set_ylabel("Station", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    fig8_path = FIG_DIR / "fig8_spatial_correlation.png"
    fig.savefig(fig8_path, dpi=200, bbox_inches="tight")
    print(f"  Saved {fig8_path}")
    plt.close()

    # --- Fig 9: Distance Decay ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(distances, correlations, c="#2c3e50", s=60, alpha=0.7, zorder=5,
               edgecolors="white", linewidth=0.5)

    # Fit curve
    d_smooth = np.linspace(0, distances.max() * 1.05, 200)
    corr_smooth = exp_decay(d_smooth, a_fit, d0_fit)
    ax.plot(d_smooth, corr_smooth, "-", color="#e74c3c", linewidth=2.5,
            label=f"exponential decay: corr = {a_fit:.2f} · exp(−d/{d0_fit:.0f} km)",
            zorder=4)

    # Annotation: d0 line
    ax.axvline(d0_fit, color="#e74c3c", linestyle="--", alpha=0.4)
    ax.annotate(
        f"d₀ = {d0_fit:.0f} km\n(e-folding distance)",
        xy=(d0_fit, exp_decay(d0_fit, a_fit, d0_fit)),
        xytext=(d0_fit + 15, exp_decay(d0_fit, a_fit, d0_fit) + 0.05),
        fontsize=10, color="#e74c3c",
        arrowprops=dict(arrowstyle="->", color="#e74c3c", alpha=0.6),
    )

    ax.set_xlabel("Inter-Station Distance (km)", fontsize=13)
    ax.set_ylabel("PM2.5 Pearson Correlation", fontsize=13)
    ax.set_title(
        f"Spatial Decay of PM2.5 Correlation\n"
        f"d₀ = {d0_fit:.0f} km (e-folding distance)",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fig9_path = FIG_DIR / "fig9_distance_decay.png"
    fig.savefig(fig9_path, dpi=200, bbox_inches="tight")
    print(f"  Saved {fig9_path}")
    plt.close()

    # ================================================================
    # 7. Save results
    # ================================================================
    results = {
        "decay_length_km": round(d0_fit, 1),
        "decay_amplitude": round(a_fit, 4),
        "max_corr": round(max_corr_val, 4),
        "min_corr": round(min_corr_val, 4),
        "mean_corr": round(mean_corr_val, 4),
        "closest_pair": f"{closest_pair['station_pair']} ({closest_pair['distance_km']:.1f} km, r={closest_pair['correlation']:.4f})",
        "farthest_pair": f"{farthest_pair['station_pair']} ({farthest_pair['distance_km']:.1f} km, r={farthest_pair['correlation']:.4f})",
        "max_corr_pair": f"{max_corr_pair['station_pair']} ({max_corr_pair['distance_km']:.1f} km, r={max_corr_pair['correlation']:.4f})",
        "min_corr_pair": f"{min_corr_pair['station_pair']} ({min_corr_pair['distance_km']:.1f} km, r={min_corr_pair['correlation']:.4f})",
        "n_stations": n_stations,
        "total_records": len(df),
    }

    results_path = RESULTS_DIR / "spatial_analysis.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {results_path}")

    # Also save pair details as CSV
    pairs_df.to_csv(RESULTS_DIR / "spatial_pairs.csv", index=False)
    print(f"Saved pair details to {RESULTS_DIR / 'spatial_pairs.csv'}")

    print("\n" + "=" * 60)
    print("TASK B COMPLETE")
    print(f"  d0 = {d0_fit:.0f} km (e-folding distance)")
    print(f"  Max correlation: {max_corr_val:.4f}")
    print(f"  Min correlation: {min_corr_val:.4f}")
    print(f"  Mean correlation: {mean_corr_val:.4f}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
