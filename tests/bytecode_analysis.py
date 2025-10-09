#!/usr/bin/env python3
import subprocess
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import shutil
import re
from collections import Counter
import seaborn as sns
import os
import numpy as np
# ======================
# Configuration
# ======================
IRIDIUM_DIR = Path("..").resolve()
QJS_DIR = Path("/home/anirudh/wd/quickjs")

# --- Directory Configuration ---
RESULTS_BC_DIR = Path("results_bc").resolve()
JSON_BENCHMARK_DIR = Path("json_benchmark").resolve()
JS3_BENCHMARK_DIR = Path("js3_benchmark").resolve()


SUNSPIDER = [
    # "benchmarks/sunspider/3d-cube.cjs",
    # "benchmarks/sunspider/3d-morph.cjs",
    # "benchmarks/sunspider/3d-raytrace.cjs",
    # "benchmarks/sunspider/access-binary-trees.cjs",
    # "benchmarks/sunspider/access-fannkuch.cjs",
    # "benchmarks/sunspider/access-nbody.cjs",
    # "benchmarks/sunspider/access-nsieve.cjs",
    # "benchmarks/sunspider/bitops-3bit-bits-in-byte.cjs",
    # "benchmarks/sunspider/bitops-bits-in-byte.cjs",
    # "benchmarks/sunspider/bitops-bitwise-and.cjs",
    # "benchmarks/sunspider/bitops-nsieve-bits.cjs",
    # "benchmarks/sunspider/controlflow-recursive.cjs",
    # "benchmarks/sunspider/crypto-aes.cjs",
    # "benchmarks/sunspider/crypto-md5.cjs",
    # "benchmarks/sunspider/crypto-sha1.cjs",
    # "benchmarks/sunspider/date-format-xparb.cjs",
    # "benchmarks/sunspider/math-cordic.cjs",
    # "benchmarks/sunspider/math-partial-sums.cjs",
    # "benchmarks/sunspider/math-spectral-norm.cjs",
    # "benchmarks/sunspider/regexp-dna.cjs",
    # "benchmarks/sunspider/string-base64.cjs",
    # "benchmarks/sunspider/string-fasta.cjs",
    # "benchmarks/sunspider/string-tagcloud.cjs",
    # "benchmarks/sunspider/string-unpack-code.cjs",
    # "benchmarks/sunspider/string-validate-input.cjs",
    # "benchmarks/kraken-1.0/kraken-1.0-ai-astar.cjs",
    # "benchmarks/kraken-1.0/kraken-1.0-audio-beat-detection.cjs",
    "benchmarks/kraken-1.0/kraken-1.0-audio-dft.cjs",
    # "benchmarks/kraken-1.0/kraken-1.0-audio-fft.cjs",
    # "benchmarks/kraken-1.0/kraken-1.0-audio-oscillator.cjs",
    # "benchmarks/kraken-1.0/kraken-1.0-imaging-darkroom.cjs",
    # "benchmarks/kraken-1.0/kraken-1.0-imaging-desaturate.cjs",
    # "benchmarks/kraken-1.0/kraken-1.0-imaging-gaussian-blur.cjs",
    # "benchmarks/kraken-1.0/kraken-1.0-json-parse-financial.cjs",
    # "benchmarks/kraken-1.0/kraken-1.0-json-stringify-tinderbox.cjs",
    # "benchmarks/kraken-1.0/kraken-1.0-stanford-crypto-aes.cjs",
    # "benchmarks/kraken-1.0/kraken-1.0-stanford-crypto-ccm.cjs",
    # "benchmarks/kraken-1.0/kraken-1.0-stanford-crypto-pbkdf2.cjs",
    # "benchmarks/kraken-1.0/kraken-1.0-stanford-crypto-sha256-iterative.cjs",
    # "benchmarks/ML/ML_load.cjs",
    # "benchmarks/Octane2/Regexpbenchmark.cjs",
    # "benchmarks/SeaMonster/gaussian-blur.cjs",
    # "benchmarks/UniPoker/benchmark.cjs",
]

# ======================
# Utilities
# ======================

def get_bytecode(cmd, cwd=None):
    """Runs a command and returns its combined stdout and stderr for parsing."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, cwd=cwd,
        )
        return result.stdout + result.stderr
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Command failed: {' '.join(map(str, cmd))}")
        print(f"   Output: {(e.stdout + e.stderr).strip()}")
        return ""

def parse_and_count_opcodes(bytecode_text):
    """Parses text to find and count all bytecode opcodes, respecting function boundaries."""
    opcode_pattern = re.compile(r"^\s*\d+:\s+([a-zA-Z_]+)")
    total_counts = Counter()
    
    function_blocks = bytecode_text.split("function:")
    
    for block in function_blocks[1:]:
        opcode_section_start = block.find(";;")
        if opcode_section_start == -1:
            continue
            
        opcode_section = block[opcode_section_start:]
        
        opcodes_in_block = []
        for line in opcode_section.splitlines():
            match = opcode_pattern.match(line)
            if match:
                opcodes_in_block.append(match.group(1))
        
        total_counts.update(opcodes_in_block)
        
    return total_counts

def save_opcode_results(data, ordered_benchmarks, output_dir, file_prefix):
    """Converts collected opcode data to a DataFrame, saves CSV, and generates a heatmap."""
    if not data:
        print(f"No opcode data to save for {file_prefix}")
        return

    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.reindex(ordered_benchmarks).dropna(how='all')
    df = df.fillna(0).astype(int)
    df = df.reindex(sorted(df.columns), axis=1)

    csv_path = output_dir / f"{file_prefix}_bytecode_counts.csv"
    df.to_csv(csv_path)
    print(f"✅ Saved Opcode CSV to {csv_path}")

    if df.empty:
        print(f"Skipping opcode heatmap for empty data in {file_prefix}")
        return
        
    heatmap_path = output_dir / f"{file_prefix}_bytecode_heatmap.png"
    plt.figure(figsize=(max(15, len(df.columns) * 0.4), max(8, len(df.index) * 0.5)))
    sns.heatmap(df, annot=True, fmt="d", cmap="viridis", linewidths=.5)
    plt.title(f"Bytecode Frequency Heatmap: {file_prefix.replace('_', ' ').title()}", fontsize=16)
    plt.xlabel("Opcodes", fontsize=12)
    plt.ylabel("Benchmarks", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"📊 Saved Opcode Heatmap to {heatmap_path}")
    
# Auxillary metrics
def parse_opcode_definitions(opcode_header_path=QJS_DIR/"quickjs-opcode.h"):
    opcode_sizes = {}
    try:
        with open(opcode_header_path, 'r') as f: content = f.read()
        pattern = r'DEF\s*\(\s*(\w+)\s*,\s*(\d+)\s*,'
        for match in re.finditer(pattern, content):
            opcode_sizes[match.group(1)] = int(match.group(2))
        print(f"✅ Parsed {len(opcode_sizes)} opcode definitions")
        return opcode_sizes
    except FileNotFoundError:
        print(f"⚠️  Warning: Could not find {opcode_header_path}. Assuming size 1 for all opcodes.")
        return {}

def aux_analysis(bytecode_str, opcode_sizes={}):
    """Analyze bytecode to extract various metrics."""
    metrics = { "sum_closure_vars": 0, "sum_local_vars": 0, "sum_stack_size": 0,
                "max_closure_vars": 0, "max_local_vars": 0, "max_stack_size": 0,
                "bytecode_count": 0, "bytecode_size": 0 }
    
    function_blocks = bytecode_str.split("function:")
    
    for block in function_blocks[1:]:
        closure_match = re.search(r'closure vars?:\s*\n((?:\s+\d+:.*\n)*)', block)
        closure_count = len(re.findall(r'^\s+\d+:', closure_match.group(1), re.MULTILINE)) if closure_match else 0
        
        local_match = re.search(r'locals:\s*\n((?:\s+\d+:.*\n)*)', block)
        local_count = len(re.findall(r'^\s+\d+:', local_match.group(1), re.MULTILINE)) if local_match else 0
        
        stack_match = re.search(r'stack_size:\s*(\d+)', block)
        stack_size = int(stack_match.group(1)) if stack_match else 0

        instruction_pattern = r'^\s+\d+:\s+(\w+)'
        instructions = re.findall(instruction_pattern, block, re.MULTILINE)
        
        metrics["bytecode_count"] += len(instructions)
        metrics["bytecode_size"] += sum(opcode_sizes.get(op, 1) for op in instructions)
        metrics["sum_closure_vars"] += closure_count
        metrics["sum_local_vars"] += local_count
        metrics["sum_stack_size"] += stack_size
        metrics["max_closure_vars"] = max(metrics["max_closure_vars"], closure_count)
        metrics["max_local_vars"] = max(metrics["max_local_vars"], local_count)
        metrics["max_stack_size"] = max(metrics["max_stack_size"], stack_size)
    
    return metrics

def save_aux_results(data, ordered_benchmarks, output_dir, file_prefix):
    """Write benchmark auxiliary metrics to CSV."""
    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.reindex(ordered_benchmarks).dropna(how='all')
    df = df.fillna(0)

    csv_path = output_dir / f"{file_prefix}_aux_metrics.csv"
    df.to_csv(csv_path)
    print(f"✅ Saved Aux CSV for {file_prefix} → {csv_path}")

def plot_aux_results(data, ordered_benchmarks, output_dir, file_prefix):
    """Plot auxiliary bytecode metrics."""
    if not data:
        print(f"No aux data to plot for {file_prefix}")
        return
    
    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.reindex(ordered_benchmarks).dropna(how='all')
    df = df.fillna(0)
    benchmarks = df.index.tolist()

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)
    fig.suptitle(f"Auxiliary Metrics: {file_prefix.replace('_', ' ').title()}", fontsize=16, fontweight='bold')
    x = np.arange(len(benchmarks))
    width = 0.25
    
    # Plot 1: SUM metrics
    axes[0].bar(x - width, df['sum_closure_vars'], width, label='Closure Vars', color='#FF6B6B')
    axes[0].bar(x, df['sum_local_vars'], width, label='Local Vars', color='#4ECDC4')
    axes[0].bar(x + width, df['sum_stack_size'], width, label='Stack Size', color='#45B7D1')
    axes[0].set_title('Sum Metrics', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Sum Values')
    
    # Plot 2: MAX metrics
    axes[1].bar(x - width, df['max_closure_vars'], width, label='Closure Vars', color='#FF6B6B')
    axes[1].bar(x, df['max_local_vars'], width, label='Local Vars', color='#4ECDC4')
    axes[1].bar(x + width, df['max_stack_size'], width, label='Stack Size', color='#45B7D1')
    axes[1].set_title('Max Metrics', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Max Values')

    # Plot 3: Bytecode metrics
    width2 = 0.35
    axes[2].bar(x - width2/2, df['bytecode_count'], width2, label='Bytecode Count', color='#A29BFE')
    axes[2].bar(x + width2/2, df['bytecode_size'], width2, label='Bytecode Size (bytes)', color='#FD79A8')
    axes[2].set_title('Bytecode Metrics', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Count / Size')

    for ax in axes:
        ax.set_xlabel('Benchmarks', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(benchmarks, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    output_path = output_dir / f'{file_prefix}_aux_metrics_plot.png'
    plt.savefig(output_path, dpi=150)
    print(f"📊 Saved Aux metrics plot to {output_path}")
    plt.close()

# ======================
# Main Analysis Loop
# ======================
def main():
    start_time = datetime.now()
    print(f"--- Script started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ---", flush=True)

    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    run_output_dir = RESULTS_BC_DIR / timestamp
    run_output_dir.mkdir(parents=True, exist_ok=True)
    JSON_BENCHMARK_DIR.mkdir(exist_ok=True)
    JS3_BENCHMARK_DIR.mkdir(exist_ok=True)

    all_counts_opcodes = {"QJS+Source": {}, "JS3": {}, "Iridium": {}}
    all_counts_aux = {"QJS+Source": {}, "JS3": {}, "Iridium": {}}
    
    opcode_sizes = parse_opcode_definitions()

    for test in SUNSPIDER:
        print(f"\n=== Analyzing {test} ===", flush=True)
        test_path = IRIDIUM_DIR / "tests" / test
        benchmark_name = Path(test).name

        target_json_path = JSON_BENCHMARK_DIR / f"{benchmark_name}.json"
        target_js3_path = JS3_BENCHMARK_DIR / f"{benchmark_name}.js3"
        
        if not (target_json_path.exists() and target_js3_path.exists()):
            print(f"  ▷ Cache miss. Compiling {benchmark_name} with Iridium...", flush=True)
            compile_result = subprocess.run(
                ["./iridium", "iri", "--ljson", ".", str(test_path)],
                cwd=IRIDIUM_DIR, capture_output=True, text=True
            )
            if compile_result.returncode != 0:
                print(f"  ❌ Iridium compilation FAILED. Skipping.", flush=True); continue

            outputs_dir = IRIDIUM_DIR / "outputs"
            generated_json = next(outputs_dir.glob("*.json"), None)
            if generated_json: shutil.move(str(generated_json), str(target_json_path))
            generated_js3 = next(outputs_dir.glob("*.js3"), None)
            if generated_js3: shutil.move(str(generated_js3), str(target_js3_path))

        json_file = target_json_path if target_json_path.exists() else None
        js3_file = target_js3_path if target_js3_path.exists() else None

        configs = {
            "QJS+Source": ["./script_bc.sh", str(test_path)] if test_path.exists() else None,
            "JS3": ["./script_bc.sh", str(js3_file)] if js3_file else None,
            "Iridium": ["./tiri_bc.sh", str(json_file)] if json_file else None,
        }

        for name, cmd in configs.items():
            if not cmd: continue
            
            print(f"→ Generating bytecode for {name}...", flush=True)
            bytecode_text = get_bytecode(cmd, cwd=QJS_DIR)
            if bytecode_text:
                all_counts_opcodes[name][benchmark_name] = parse_and_count_opcodes(bytecode_text)
                all_counts_aux[name][benchmark_name] = aux_analysis(bytecode_text, opcode_sizes)
                
    print("\n--- Aggregating and saving results ---")

    processed_benchmark_names = {k for d in all_counts_opcodes.values() for k in d}
    ordered_names = [Path(p).name for p in SUNSPIDER if Path(p).name in processed_benchmark_names]

    # --- FIX: Refactored saving/plotting loop for consistent naming ---
    for config_name in ["QJS+Source", "JS3", "Iridium"]:
        sanitized_name = config_name.lower().replace('+', '-')

        if all_counts_opcodes.get(config_name):
            save_opcode_results(all_counts_opcodes[config_name], ordered_names, run_output_dir, sanitized_name)

        if all_counts_aux.get(config_name):
            save_aux_results(all_counts_aux[config_name], ordered_names, run_output_dir, sanitized_name)
            plot_aux_results(all_counts_aux[config_name], ordered_names, run_output_dir, sanitized_name)

    end_time = datetime.now()
    print(f"\n--- Script finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')} ---", flush=True)
    print(f"--- Total runtime: {end_time - start_time} ---", flush=True)

if __name__ == "__main__":
    main()