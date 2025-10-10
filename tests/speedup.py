# speedup_calculator.py

import pandas as pd
from scipy.stats import gmean
from pathlib import Path

# ================================================================
# Configuration - ✏️ EDIT THIS VALUE
# ================================================================

# 1. Path to your CSV file
CSV_FILE_PATH = Path("/home/anirudh/wd/Iridium/tests/results/09-10-2025-20-54-21/benchmark_results.csv")

# 2. Define the baseline and the configuration to compare against it
BASELINE_CONFIG = "QJS+SOURCE"
TARGET_CONFIG = "IRIDIUM"

# ================================================================
# Calculation Logic
# ================================================================

def calculate_non_outlier_mean(series: pd.Series) -> float:
    """
    Calculates the mean of a series after removing outliers based on the
    1.5 * IQR rule (the same logic used for box plot whiskers).
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Filter out outliers and calculate the mean
    filtered_series = series[(series >= lower_bound) & (series <= upper_bound)]
    return filtered_series.mean()


def main():
    """Main function to calculate and display speedup."""
    if not CSV_FILE_PATH.exists():
        print(f"Error: The file was not found at {CSV_FILE_PATH}")
        return

    print(f"Reading data from {CSV_FILE_PATH}\n")
    df = pd.read_csv(CSV_FILE_PATH)

    # We only care about execution time for speedup calculation
    exec_df = df[df['config'].isin([BASELINE_CONFIG, TARGET_CONFIG])].copy()
    
    # Group by benchmark and config, then apply the outlier removal and mean calculation
    mean_times = exec_df.groupby(['benchmark', 'config'])['exec_time_ms'].apply(calculate_non_outlier_mean)

    # Pivot the table to have configs as columns for easy comparison
    comparison_df = mean_times.unstack()
    
    # Calculate speedup for each benchmark
    comparison_df['speedup'] = comparison_df[BASELINE_CONFIG] / comparison_df[TARGET_CONFIG]
    
    # Drop rows where speedup could not be calculated (e.g., missing data for one config)
    comparison_df.dropna(inplace=True)
    
    print("--- Mean Execution Times (Outliers Removed) and Speedup ---")
    print(comparison_df.to_string(float_format="%.2f"))
    print("-" * 60)
    
    # Calculate the geometric mean of the speedup column
    if not comparison_df.empty:
        overall_speedup = gmean(comparison_df['speedup'])
        print(f"\n📈 Geometric Mean Speedup ({TARGET_CONFIG} vs. {BASELINE_CONFIG}): {overall_speedup:.3f}x")
    else:
        print("\nCould not calculate overall speedup. No valid benchmark data found.")


if __name__ == "__main__":
    main()