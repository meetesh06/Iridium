#!/usr/bin/env python3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# ======================
# Configuration
# ======================

# --- Directory Configuration ---
# Assumes this script is in the 'tests' directory
RESULTS_BC_DIR = Path("results_bc").resolve()
HEATMAP_OUTPUT_DIR = Path("bc_heatmaps").resolve()

# --- Opcode Classification (from PDF) ---
# NOTE: "OP_" prefix is removed to match the CSV column headers
OPCODE_CLASSIFICATION = {
    "Call Site Related": {
        "Least Optimised": ["call", "call_method", "return"],
        "Moderately Optimized": [],
        "Highly Optimized": ["return_undef"],
        "Most Optimized": ["tail_call", "tail_call_method", "call0", "call1", "call2", "call3"]
    },
    "Control Flow Related": {
        "Least Optimised": ["if_false", "if_true", "goto"],
        "Moderately Optimized": [],
        "Highly Optimized": ["goto16"],
        "Most Optimized": ["if_false8", "if_true8", "goto8"]
    },
    "Stack Frame References": {
        "Least Optimised": ["get_loc_check", "put_loc_check", "put_loc_check_init", "get_var_ref_check", "put_var_ref_check", "put_var_ref_check_init"], # may remove init
        "Moderately Optimized": ["get_loc", "put_loc", "set_loc", "get_arg", "put_arg", "set_arg", "get_var_ref", "put_var_ref", "set_var_ref", "get_loc8", "put_loc8", "set_loc8"],
        "Highly Optimized": [
            "get_loc0", "get_loc1", "get_loc2", "get_loc3",
            "put_loc0", "put_loc1", "put_loc2", "put_loc3",
            "get_arg0", "get_arg1", "get_arg2", "get_arg3",
            "put_arg0", "put_arg1", "put_arg2", "put_arg3",
            "get_var_ref0", "get_var_ref1", "get_var_ref2", "get_var_ref3",
            "put_var_ref0", "put_var_ref1", "put_var_ref2", "put_var_ref3"
            ],
        "Most Optimized": [
            "get_loc0_loc1", 
            "set_loc0", "set_loc1", "set_loc2", "set_loc3",
            "set_arg0", "set_arg1", "set_arg2", "set_arg3",
            "set_var_ref0", "set_var_ref1", "set_var_ref2", "set_var_ref3"
        ]
    },
    "Field Operations": {
        "Least Optimised": ["get_field", "to_propkey", "to_propkey2", "put_field", "get_array_el"],
        "Moderately Optimized": [],
        "Highly Optimized": [],
        "Most Optimized": ["get_array_el2", "get_length", "get_field2"]
    }
}

# For ordering the plot axes
ROW_HEADERS = ["Call Site Related", "Control Flow Related", "Stack Frame References", "Field Operations"]
COLUMN_HEADERS = ["Least Optimised", "Moderately Optimized", "Highly Optimized", "Most Optimized"]

# ======================
# Helper Functions
# ======================

def format_annotation(value):
    """Formats the annotation text for the heatmap cells."""
    if value == 0:
        return "NA"
    if value < 1000:
        return f"{int(value)}"
    return f"{value/1000:.0f}k"

# ======================
# Main Script
# ======================

def main():
    """Main function to find CSVs, process data, and generate heatmaps."""
    
    print("--- Starting Categorized Heatmap Generation ---")
    
    # Find the latest results directory
    try:
        latest_run_dir = sorted(RESULTS_BC_DIR.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True)[0]
    except IndexError:
        print(f"❌ Error: No results found in '{RESULTS_BC_DIR}'. Please run 'bytecode_analysis.py' first.")
        return
        
    print(f"🔍 Using latest results from: {latest_run_dir.name}")
    
    # Create the main output directory
    HEATMAP_OUTPUT_DIR.mkdir(exist_ok=True)
    
    csv_files_to_process = [
        "iridium_bytecode_counts.csv",
        "js3_bytecode_counts.csv",
        "qjs_source_bytecode_counts.csv"
    ]

    for csv_filename in csv_files_to_process:
        csv_path = latest_run_dir / csv_filename
        if not csv_path.exists():
            print(f"⚠️ Warning: Could not find '{csv_filename}', skipping.")
            continue
            
        config_name = csv_path.stem.split('_')[0]
        print(f"\n--- Processing {config_name} data ---")
        df = pd.read_csv(csv_path, index_col=0)
        
        for benchmark_name, opcode_counts in df.iterrows():
            # Create a dedicated folder for the benchmark
            benchmark_output_dir = HEATMAP_OUTPUT_DIR / benchmark_name
            benchmark_output_dir.mkdir(exist_ok=True)
            
            # --- Data Calculation ---
            raw_sums = np.zeros((4, 4))
            
            for i, category in enumerate(ROW_HEADERS):
                for j, opt_level in enumerate(COLUMN_HEADERS):
                    opcodes_in_cell = OPCODE_CLASSIFICATION[category][opt_level]
                    
                    # Sum counts for all opcodes in this category/level
                    cell_sum = 0
                    for opcode in opcodes_in_cell:
                        if opcode in opcode_counts:
                            cell_sum += opcode_counts[opcode]
                    
                    raw_sums[i, j] = cell_sum
            
            # --- Normalization and Annotation ---
            normalized_values = np.zeros((4, 4))
            annotations = np.empty((4, 4), dtype=object)
            
            for i in range(4): # For each row
                row_sum = np.sum(raw_sums[i, :])
                if row_sum > 0:
                    normalized_values[i, :] = raw_sums[i, :] / row_sum
                
                for j in range(4): # For each column
                    annotations[i, j] = format_annotation(raw_sums[i, j])

            # --- Plotting ---
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(
                normalized_values,
                annot=annotations,
                fmt='',  # Use our custom string annotations
                cmap='YlGnBu',
                linewidths=.5,
                ax=ax,
                cbar_kws={'label': 'Row-wise Normalized Frequency'}
            )
            
            ax.set_xticklabels(COLUMN_HEADERS, rotation=45, ha="right")
            ax.set_yticklabels(ROW_HEADERS, rotation=0)
            
            title = f"Opcode Distribution for '{benchmark_name}'\n({config_name.upper()} Configuration)"
            ax.set_title(title, fontsize=14, fontweight='bold')
            
            output_path = benchmark_output_dir / f"{config_name}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
    print(f"\n✅ All heatmaps generated in: {HEATMAP_OUTPUT_DIR}")

if __name__ == "__main__":
    main()