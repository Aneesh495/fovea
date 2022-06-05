"""
Runner script to execute all Fovea benchmarks and output structured results.
"""

import sys
import subprocess
import os

BENCHMARKS = [
    "bench_throughput.py",
    "bench_stencil_fusion.py",
    "bench_vector_scaling.py",
]

def main():
    bench_dir = os.path.dirname(os.path.abspath(__file__))
    print("=" * 80)
    print("EXECUTING FOVEA COMPILER AND ISP BENCHMARK SUITE")
    print("=" * 80)

    for bench in BENCHMARKS:
        path = os.path.join(bench_dir, bench)
        print(f"\n>>> Running {bench}...")
        res = subprocess.run([sys.executable, path], capture_output=True, text=True)
        if res.returncode != 0:
            print(f"FAILED: {bench}")
            print(res.stderr)
            sys.exit(res.returncode)
        else:
            print(res.stdout.strip())

    print("\n" + "=" * 80)
    print("ALL BENCHMARKS EXECUTED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()
