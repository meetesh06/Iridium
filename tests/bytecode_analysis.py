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
QJS_DIR = Path("/home/maverick/js_project/quickjs")

# --- Directory Configuration ---
RESULTS_BC_DIR = Path("results_bc").resolve()
JSON_BENCHMARK_DIR = Path("json_benchmark").resolve()
JS3_BENCHMARK_DIR = Path("js3_benchmark").resolve()


SUNSPIDER = [
    "benchmarks/sunspider/3d-cube.cjs",
    "benchmarks/sunspider/3d-morph.cjs",
    "benchmarks/sunspider/3d-raytrace.cjs",
    "benchmarks/sunspider/access-binary-trees.cjs",
    "benchmarks/sunspider/access-fannkuch.cjs",
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
    # "benchmarks/kraken-1.0/kraken-1.0-audio-dft.cjs",
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
    # "benchmarks/ML/ML_load.js",
]

# ======================
# Utilities
# ======================

# Bytecode histogram
def get_bytecode(cmd, cwd=None):
    """Runs a command and returns its stdout for parsing."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, cwd=cwd,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Command failed: {' '.join(map(str, cmd))}")
        print(f"   Stderr: {e.stderr.strip()}")
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
    """Converts collected data to a DataFrame, saves CSV, and generates a heatmap."""
    if not data:
        print(f"No data to save for {file_prefix}")
        return

    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.reindex(ordered_benchmarks).dropna(how='all')
    df = df.fillna(0).astype(int)
    df = df.reindex(sorted(df.columns), axis=1)

    csv_path = output_dir / f"{file_prefix}_bytecode_counts.csv"
    df.to_csv(csv_path)
    print(f"✅ Saved CSV to {csv_path}")

    if df.empty:
        print(f"Skipping heatmap for empty data in {file_prefix}")
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
    print(f"📊 Saved Heatmap to {heatmap_path}")
    
# Auxillary metrics
def parse_opcode_definitions(opcode_header_path="quickjs-opcode.h"):
    """
    Parse quickjs-opcode.h to extract opcode sizes.
    
    Returns:
        Dictionary mapping opcode names to their sizes
    """
    opcode_sizes = {}
    
    if not os.path.exists(opcode_header_path):
        raise FileNotFoundError(f"❌ ERROR: Could not find {opcode_header_path}")
    
    try:
        with open(opcode_header_path, 'r') as f:
            content = f.read()
        
        # Pattern to match DEF(opcode_name, size, ...)
        pattern = r'DEF\s*\(\s*(\w+)\s*,\s*(\d+)\s*,'
        
        matches_found = 0
        for match in re.finditer(pattern, content):
            opcode_name = match.group(1)
            size = int(match.group(2))
            opcode_sizes[opcode_name] = size
            matches_found += 1
            
        if matches_found == 0:
            raise ValueError(f"❌ ERROR: No opcode definitions found in {opcode_header_path}")
            
        print(f"✅ Parsed {len(opcode_sizes)} opcode definitions from {opcode_header_path}")
        
        # Print some examples for verification
        print(f"   Examples: push_i32={opcode_sizes.get('push_i32', 'N/A')}, "
              f"get_var={opcode_sizes.get('get_var', 'N/A')}, "
              f"add={opcode_sizes.get('add', 'N/A')}")
        
        return opcode_sizes
        
    except Exception as e:
        raise RuntimeError(f"❌ ERROR parsing {opcode_header_path}: {str(e)}")


def extract_base_opcode(opcode_with_suffix):
    """
    Extract base opcode name from variants like:
    - get_var_ref0 -> get_var_ref
    - put_loc8 -> put_loc
    - if_false8 -> if_false
    - push_i8 -> push_i32 (special case)
    - fclosure8 -> fclosure
    """
    # Handle special push_i variants
    if opcode_with_suffix.startswith('push_i') and opcode_with_suffix[6:].isdigit():
        return 'push_i32'
    
    # Remove numeric suffix at the end
    base = re.sub(r'\d+$', '', opcode_with_suffix)
    
    # Handle 8/16 suffixes (e.g., if_false8 -> if_false)
    if base.endswith('8') or base.endswith('16'):
        # But check if it's part of the name (like push_i8)
        if not base.startswith('push_i'):
            base = base[:-1] if base.endswith('8') else base[:-2]
    
    return base


def get_opcode_size(opcode_name, opcode_sizes):
    """
    Get size for an opcode, handling variants and reporting errors.
    """
    # Try exact match first
    if opcode_name in opcode_sizes:
        return opcode_sizes[opcode_name]
    
    # Try base opcode
    base_opcode = extract_base_opcode(opcode_name)
    if base_opcode in opcode_sizes:
        return opcode_sizes[base_opcode]
    
    # If still not found, report error
    raise ValueError(f"❌ ERROR: Unknown opcode '{opcode_name}' (tried base: '{base_opcode}')")


def extract_opcodes_from_section(block):
    """
    Extract opcodes from the opcodes section of a function block.
    
    The opcodes section starts after "opcodes: <number>" and continues until
    the next section (marked by a line starting with a letter at the beginning,
    or end of block).
    
    Returns:
        List of opcode names (strings after line number and colon, before semicolon)
    """
    opcodes = []
    
    # Find the "opcodes:" line
    opcodes_match = re.search(r'opcodes:\s*\d+', block)
    if not opcodes_match:
        return opcodes
    
    # Get text after "opcodes: N" line
    start_pos = opcodes_match.end()
    remaining_text = block[start_pos:]
    
    # Process line by line
    for line in remaining_text.split('\n'):
        # Stop if we hit a new section (line starting with non-whitespace letter)
        # Or if line starts with "function:", "closure vars:", "locals:", etc.
        if re.match(r'^[a-zA-Z]', line):
            break
            
        # Check if this is an opcode line: "    0:  opcode_name  ..."
        # Pattern: optional whitespace, number, colon, whitespace, opcode_name
        opcode_match = re.match(r'^\s+\d+:\s+(\w+)', line)
        if opcode_match:
            # Remove anything after semicolon
            line_before_comment = line.split(';')[0]
            
            # Extract opcode (first word after the line number)
            opcode_match = re.search(r'^\s+\d+:\s+(\w+)', line_before_comment)
            if opcode_match:
                opcode_name = opcode_match.group(1)
                opcodes.append(opcode_name)
    
    return opcodes

def analyze_qjs_bytecode(bytecode_str, opcode_sizes):
    """
    Analyze QuickJS bytecode (QJS+Source format).
    """
    total_closure = 0
    total_local = 0
    total_stack = 0
    max_closure = 0
    max_local = 0
    max_stack = 0
    total_bytecode_count = 0
    total_bytecode_size = 0
    
    unknown_opcodes = set()
    all_opcodes_found = []
    
    # Split by function headers: /path:line:col: function:
    function_blocks = re.split(r'/[^\n]+:\d+:\d+: function:', bytecode_str)
    
    if len(function_blocks) <= 1:
        print("⚠️  WARNING: No functions found in QJS bytecode")
        return create_empty_result()
    
    functions_analyzed = 0
    
    for block in function_blocks[1:]:
        functions_analyzed += 1
        
        # Extract closure vars
        closure_section = re.search(r'closure vars:\s*\n((?:\s+\d+:.*\n)*)', block)
        if closure_section:
            closure_lines = re.findall(r'^\s+\d+:', closure_section.group(1), re.MULTILINE)
            closure_count = len(closure_lines)
        else:
            closure_count = 0
        
        # Extract local vars
        local_section = re.search(r'locals:\s*\n((?:\s+\d+:.*\n)*)', block)
        if local_section:
            local_lines = re.findall(r'^\s+\d+:', local_section.group(1), re.MULTILINE)
            local_count = len(local_lines)
        else:
            local_count = 0
        
        # Extract stack size
        stack_match = re.search(r'stack_size:\s*(\d+)', block)
        stack_size = int(stack_match.group(1)) if stack_match else 0
        
        # Extract bytecode count from "opcodes: N" line
        opcode_match = re.search(r'opcodes:\s*(\d+)', block)
        bytecode_count = int(opcode_match.group(1)) if opcode_match else 0
        total_bytecode_count += bytecode_count
        
        # Extract opcodes from the opcodes section ONLY
        opcodes = extract_opcodes_from_section(block)
        all_opcodes_found.extend(opcodes)
        
        # Calculate bytecode size
        function_bytecode_size = 0
        for opcode in opcodes:
            try:
                opcode_size = get_opcode_size(opcode, opcode_sizes)
                function_bytecode_size += opcode_size
            except ValueError as e:
                unknown_opcodes.add(opcode)
                # Use size 1 as fallback
                function_bytecode_size += 1
        
        total_bytecode_size += function_bytecode_size
        
        # Verify: number of opcodes extracted should match bytecode_count
        if len(opcodes) != bytecode_count:
            print(f"⚠️  WARNING: Function has opcodes:{bytecode_count} but found {len(opcodes)} opcodes")
        
        # Update totals
        total_closure += closure_count
        total_local += local_count
        total_stack += stack_size
        
        # Update maxes
        max_closure = max(max_closure, closure_count)
        max_local = max(max_local, local_count)
        max_stack = max(max_stack, stack_size)
    
    print(f"   Analyzed {functions_analyzed} functions")
    print(f"   Found {len(all_opcodes_found)} total opcode instances")
    
    if unknown_opcodes:
        print(f"❌ ERROR: Found {len(unknown_opcodes)} unknown opcodes: {sorted(unknown_opcodes)}")
        raise ValueError(f"Unknown opcodes found: {sorted(unknown_opcodes)}")
    
    # Verify bytecode size >= bytecode count
    if total_bytecode_size < total_bytecode_count:
        print(f"❌ ERROR: bytecode_size ({total_bytecode_size}) < bytecode_count ({total_bytecode_count})")
        raise ValueError("Bytecode size calculation is incorrect - size cannot be less than count!")
    
    return {
        "sum_closure_vars": total_closure,
        "sum_local_vars": total_local,
        "sum_stack_size": total_stack,
        "max_closure_vars": max_closure,
        "max_local_vars": max_local,
        "max_stack_size": max_stack,
        "bytecode_count": total_bytecode_count,
        "bytecode_size": total_bytecode_size,
    }

def analyze_js3_bytecode(bytecode_str, opcode_sizes):
    """
    Analyze JS3 bytecode format (similar to QJS but with different local var format).
    """
    # JS3 format is very similar to QJS, use same logic
    return analyze_qjs_bytecode(bytecode_str, opcode_sizes)

def analyze_iridium_bytecode(bytecode_str, opcode_sizes):
    """
    Analyze Iridium bytecode format.
    Different header: "function: <null>" instead of path-based
    """
    total_closure = 0
    total_local = 0
    total_stack = 0
    max_closure = 0
    max_local = 0
    max_stack = 0
    total_bytecode_count = 0
    total_bytecode_size = 0
    
    unknown_opcodes = set()
    all_opcodes_found = []
    
    # Split by Iridium function headers: "function: <null>" or "function: name"
    function_blocks = re.split(r'^function:', bytecode_str, flags=re.MULTILINE)
    
    if len(function_blocks) <= 1:
        print("⚠️  WARNING: No functions found in Iridium bytecode")
        return create_empty_result()
    
    functions_analyzed = 0
    
    for block in function_blocks[1:]:
        functions_analyzed += 1
        
        # Extract closure vars
        closure_section = re.search(r'closure vars:\s*\n((?:\s+\d+:.*\n)*)', block)
        if closure_section:
            closure_lines = re.findall(r'^\s+\d+:', closure_section.group(1), re.MULTILINE)
            closure_count = len(closure_lines)
        else:
            closure_count = 0
        
        # Extract local vars
        local_section = re.search(r'locals:\s*\n((?:\s+\d+:.*\n)*)', block)
        if local_section:
            local_lines = re.findall(r'^\s+\d+:', local_section.group(1), re.MULTILINE)
            local_count = len(local_lines)
        else:
            local_count = 0
        
        # Extract stack size
        stack_match = re.search(r'stack_size:\s*(\d+)', block)
        stack_size = int(stack_match.group(1)) if stack_match else 0
        
        # Extract bytecode count
        opcode_match = re.search(r'opcodes:\s*(\d+)', block)
        bytecode_count = int(opcode_match.group(1)) if opcode_match else 0
        total_bytecode_count += bytecode_count
        
        # Extract opcodes from the opcodes section ONLY
        opcodes = extract_opcodes_from_section(block)
        all_opcodes_found.extend(opcodes)
        
        # Calculate bytecode size
        function_bytecode_size = 0
        for opcode in opcodes:
            try:
                opcode_size = get_opcode_size(opcode, opcode_sizes)
                function_bytecode_size += opcode_size
            except ValueError as e:
                unknown_opcodes.add(opcode)
                # Use size 1 as fallback
                function_bytecode_size += 1
        
        total_bytecode_size += function_bytecode_size
        
        # Verify: number of opcodes extracted should match bytecode_count
        if len(opcodes) != bytecode_count:
            print(f"⚠️  WARNING: Function has opcodes:{bytecode_count} but found {len(opcodes)} opcodes")
        
        # Update totals
        total_closure += closure_count
        total_local += local_count
        total_stack += stack_size
        
        # Update maxes
        max_closure = max(max_closure, closure_count)
        max_local = max(max_local, local_count)
        max_stack = max(max_stack, stack_size)
    
    print(f"   Analyzed {functions_analyzed} functions")
    print(f"   Found {len(all_opcodes_found)} total opcode instances")
    
    if unknown_opcodes:
        print(f"❌ ERROR: Found {len(unknown_opcodes)} unknown opcodes: {sorted(unknown_opcodes)}")
        raise ValueError(f"Unknown opcodes found: {sorted(unknown_opcodes)}")
    
    # Verify bytecode size >= bytecode count
    if total_bytecode_size < total_bytecode_count:
        print(f"❌ ERROR: bytecode_size ({total_bytecode_size}) < bytecode_count ({total_bytecode_count})")
        raise ValueError("Bytecode size calculation is incorrect!")
    
    return {
        "sum_closure_vars": total_closure,
        "sum_local_vars": total_local,
        "sum_stack_size": total_stack,
        "max_closure_vars": max_closure,
        "max_local_vars": max_local,
        "max_stack_size": max_stack,
        "bytecode_count": total_bytecode_count,
        "bytecode_size": total_bytecode_size,
    }

def analyze_bytecode_auto(bytecode_str, opcode_sizes, config_name=None):
    """
    Auto-detect format and analyze bytecode.
    """
    # Auto-detect format
    if config_name:
        config_lower = config_name.lower()
        if 'iridium' in config_lower or 'iri' in config_lower:
            return analyze_iridium_bytecode(bytecode_str, opcode_sizes)
        elif 'js3' in config_lower:
            return analyze_js3_bytecode(bytecode_str, opcode_sizes)
    
    # Check content for Iridium markers
    if '[Iridium]' in bytecode_str or re.search(r'^function: <', bytecode_str, re.MULTILINE):
        return analyze_iridium_bytecode(bytecode_str, opcode_sizes)
    else:
        return analyze_qjs_bytecode(bytecode_str, opcode_sizes)

def create_empty_result():
    """Return empty result dictionary."""
    return {
        "sum_closure_vars": 0,
        "sum_local_vars": 0,
        "sum_stack_size": 0,
        "max_closure_vars": 0,
        "max_local_vars": 0,
        "max_stack_size": 0,
        "bytecode_count": 0,
        "bytecode_size": 0,
    }
    
def save_aux_results(run_output_dir, config_name, all_counts):
    """Write benchmark results (per config) to CSV with bytecode metrics."""
    csv_path = run_output_dir / f"{config_name}_aux.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Benchmark",
            "Max_Closure_Vars", "Max_Local_Vars", "Max_Stack_Size",
            "Sum_Closure_Vars", "Sum_Local_Vars", "Sum_Stack_Size",
            "Bytecode_Count", "Bytecode_Size"
        ])
        for benchmark, data in all_counts[config_name].items():
            writer.writerow([
                benchmark,
                data["max_closure_vars"], data["max_local_vars"], data["max_stack_size"],
                data["sum_closure_vars"], data["sum_local_vars"], data["sum_stack_size"],
                data["bytecode_count"], data["bytecode_size"],
            ])
    print(f"✅ Saved CSV for {config_name} → {csv_path}")

def plot_aux_results(run_output_dir, config_name, all_counts):
    """
    Plot bytecode metrics including bytecode count and size.
    
    Args:
        run_output_dir: Directory to save the plots
        config_name: Name of the configuration
        all_counts: Dictionary with structure all_counts[config_name][benchmark][metric]
    """
    if config_name not in all_counts or not all_counts[config_name]:
        print(f"No data to plot for {config_name}")
        return
    
    all_counts_aux = all_counts[config_name]
    benchmarks = list(all_counts_aux.keys())
    
    # Prepare data arrays
    sum_closure = [all_counts_aux[b]["sum_closure_vars"] for b in benchmarks]
    sum_local = [all_counts_aux[b]["sum_local_vars"] for b in benchmarks]
    sum_stack = [all_counts_aux[b]["sum_stack_size"] for b in benchmarks]
    max_closure = [all_counts_aux[b]["max_closure_vars"] for b in benchmarks]
    max_local = [all_counts_aux[b]["max_local_vars"] for b in benchmarks]
    max_stack = [all_counts_aux[b]["max_stack_size"] for b in benchmarks]
    bytecode_count = [all_counts_aux[b]["bytecode_count"] for b in benchmarks]
    bytecode_size = [all_counts_aux[b]["bytecode_size"] for b in benchmarks]
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(20, 6))
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    x = np.arange(len(benchmarks))
    width = 0.25
    
    # Plot 1: SUM metrics
    bars1 = ax1.bar(x - width, sum_closure, width, label='Closure Vars', color='#FF6B6B', alpha=0.8)
    bars2 = ax1.bar(x, sum_local, width, label='Local Vars', color='#4ECDC4', alpha=0.8)
    bars3 = ax1.bar(x + width, sum_stack, width, label='Stack Size', color='#45B7D1', alpha=0.8)
    
    ax1.set_xlabel('Benchmarks', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Sum Values', fontsize=12, fontweight='bold')
    ax1.set_title(f'{config_name} - Sum Metrics', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(benchmarks, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: MAX metrics
    bars4 = ax2.bar(x - width, max_closure, width, label='Closure Vars', color='#FF6B6B', alpha=0.8)
    bars5 = ax2.bar(x, max_local, width, label='Local Vars', color='#4ECDC4', alpha=0.8)
    bars6 = ax2.bar(x + width, max_stack, width, label='Stack Size', color='#45B7D1', alpha=0.8)
    
    ax2.set_xlabel('Benchmarks', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Max Values', fontsize=12, fontweight='bold')
    ax2.set_title(f'{config_name} - Max Metrics', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(benchmarks, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bars in [bars4, bars5, bars6]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Bytecode metrics
    width2 = 0.35
    bars7 = ax3.bar(x - width2/2, bytecode_count, width2, label='Bytecode Count', color='#A29BFE', alpha=0.8)
    bars8 = ax3.bar(x + width2/2, bytecode_size, width2, label='Bytecode Size', color='#FD79A8', alpha=0.8)
    
    ax3.set_xlabel('Benchmarks', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Count / Size', fontsize=12, fontweight='bold')
    ax3.set_title(f'{config_name} - Bytecode Metrics', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(benchmarks, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bars in [bars7, bars8]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    os.makedirs(run_output_dir, exist_ok=True)
    output_path = os.path.join(run_output_dir, f'{config_name}_bytecode_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved bytecode metrics plot to {output_path}")
    plt.close()

# ======================
# Main Analysis Loop
# ======================
def main():
    # --- ADDED: Record start time ---
    start_time = datetime.now()
    print(f"--- Script started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ---", flush=True)

    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    run_output_dir = RESULTS_BC_DIR / timestamp
    run_output_dir.mkdir(parents=True, exist_ok=True)
    JSON_BENCHMARK_DIR.mkdir(exist_ok=True)
    JS3_BENCHMARK_DIR.mkdir(exist_ok=True)

    all_counts_opcodes = {"QJS+Source": {}, "JS3": {}, "Iridium": {}}
    all_counts_aux = {"QJS+Source": {}, "JS3": {}, "Iridium": {}}

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
                print(f"  ❌ Iridium compilation FAILED for {benchmark_name}. Skipping.", flush=True)
                print(f"  Stderr: {compile_result.stderr.strip()}", flush=True)
                continue

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
            if not cmd:
                print(f"→ Skipping {name} (no input file)")
                continue
            
            print(f"→ Generating bytecode for {name}...", flush=True)
            bytecode_text = get_bytecode(cmd, cwd=QJS_DIR)
            opcode_sizes = parse_opcode_definitions("/home/maverick/js_project/quickjs/quickjs-opcode.h")
            # print(bytecode_text)
            if bytecode_text:
                opcode_counts = parse_and_count_opcodes(bytecode_text)
                all_counts_opcodes[name][benchmark_name] = opcode_counts

                if(name=="QJS+Source"):
                    aux_results = analyze_qjs_bytecode(bytecode_text, opcode_sizes)
                elif(name=="JS3"):
                    aux_results = analyze_js3_bytecode(bytecode_text, opcode_sizes)
                else:
                    aux_results = analyze_iridium_bytecode(bytecode_text, opcode_sizes)
                all_counts_aux[name][benchmark_name] = aux_results
                
    print("\n--- Aggregating and saving results ---")

    processed_benchmark_names = set()
    for config_data in all_counts_opcodes.values():
        processed_benchmark_names.update(config_data.keys())

    ordered_names = [Path(p).name for p in SUNSPIDER if Path(p).name in processed_benchmark_names]

    save_opcode_results(all_counts_opcodes["Iridium"], ordered_names, run_output_dir, "iridium")
    save_opcode_results(all_counts_opcodes["JS3"], ordered_names, run_output_dir, "js3")
    save_opcode_results(all_counts_opcodes["QJS+Source"], ordered_names, run_output_dir, "qjs_source")
    
    for config_name in all_counts_aux:
        if all_counts_aux[config_name]:
            save_aux_results(run_output_dir, config_name, all_counts_aux)
            plot_aux_results(run_output_dir, config_name, all_counts_aux)

    # --- ADDED: Record end time and calculate duration ---
    end_time = datetime.now()
    total_runtime = end_time - start_time
    print(f"\n--- Script finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')} ---", flush=True)
    print(f"--- Total runtime: {total_runtime} ---", flush=True)


if __name__ == "__main__":
    main()