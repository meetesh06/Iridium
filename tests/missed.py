#!/usr/bin/env python3
import subprocess
from pathlib import Path
from datetime import datetime

# ======================
# Configuration
# ======================
IRIDIUM_DIR = Path("..").resolve()
OUTPUT_FILE = Path("missed_op.txt")

# The 16 benchmark files you specified
BENCHMARKS = [
    "benchmarks/sunspider/3d-cube.cjs",
    "benchmarks/sunspider/3d-morph.cjs",
    "benchmarks/sunspider/access-fannkuch.cjs",
    "benchmarks/sunspider/bitops-bitwise-and.cjs",
    "benchmarks/sunspider/math-spectral-norm.cjs",
    "benchmarks/sunspider/string-fasta.cjs",
    "benchmarks/kraken-1.0/kraken-1.0-ai-astar.cjs",
    "benchmarks/kraken-1.0/kraken-1.0-audio-beat-detection.cjs",
    "benchmarks/kraken-1.0/kraken-1.0-imaging-darkroom.cjs",
    "benchmarks/kraken-1.0/kraken-1.0-imaging-desaturate.cjs",
    "benchmarks/kraken-1.0/kraken-1.0-imaging-gaussian-blur.cjs",
    "benchmarks/kraken-1.0/kraken-1.0-stanford-crypto-pbkdf2.cjs",
    "benchmarks/ML/ML_load.cjs",
    "benchmarks/Octane2/Regexpbenchmark.cjs",
    "benchmarks/SeaMonster/gaussian-blur.cjs",
    "benchmarks/UniPoker/benchmark.cjs",
]

# The exact strings to count from your C++ code
SEARCH_STRING_READ = "JSComputedFieldReadSEXP - removed field cast"
SEARCH_STRING_WRITE = "JSComputedFieldWriteSEXP - removed field cast"

# ======================
# Main Script
# ======================
def main():
    """
    Compiles benchmarks with Iridium and counts specific compiler output messages.
    """
    start_time = datetime.now()
    print(f"--- Script started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ---", flush=True)

    all_results = []

    for test_file in BENCHMARKS:
        print(f"Compiling: {test_file} ...", flush=True)
        test_path = IRIDIUM_DIR / "tests" / test_file

        # This is the command to compile the file with Iridium
        cmd = ["./iridium", "iri", "--ljson", ".", str(test_path)]

        # Run the compiler and capture its output
        result = subprocess.run(
            cmd,
            cwd=IRIDIUM_DIR,
            capture_output=True,
            text=True
        )

        # Combine stdout and stderr to ensure we search all output
        compiler_output = result.stdout + result.stderr

        # Count the occurrences of the two specific strings
        read_count = compiler_output.count(SEARCH_STRING_READ)
        write_count = compiler_output.count(SEARCH_STRING_WRITE)

        all_results.append((test_file, read_count, write_count))

    # --- Write results to the output file ---
    with open(OUTPUT_FILE, "w") as f:
        f.write("Iridium Compiler Optimization Counts\n")
        f.write("=" * 40 + "\n\n")

        for benchmark, reads, writes in all_results:
            f.write(f"{benchmark}:\n")
            f.write(f"  Removed Read Casts:  {reads}\n")
            f.write(f"  Removed Write Casts: {writes}\n\n")

    print(f"\n✅ Analysis complete. Results written to '{OUTPUT_FILE}'")
    
    end_time = datetime.now()
    print(f"\n--- Script finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')} ---", flush=True)
    print(f"--- Total runtime: {end_time - start_time} ---", flush=True)


if __name__ == "__main__":
    main()