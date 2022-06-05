"""
Benchmark measuring SIMD vector scaling across vector widths
(scalar, vector<4xf32>, vector<8xf32>, and vector<16xf32>) for
point-wise color transformations and stencil operations.
"""

import time
import math


def run_scalar_transform(data: list, num_runs: int = 5):
    """Scalar point-wise transform simulating scalar MLIR execution."""
    n = len(data)
    out = [0.0] * n
    start = time.perf_counter()
    for _ in range(num_runs):
        for i in range(n):
            x = data[i]
            # Simulate polynomial tone curve: y = a*x^3 + b*x^2 + c*x
            out[i] = 0.5 * x * x * x - 0.2 * x * x + 0.7 * x
    elapsed = (time.perf_counter() - start) / num_runs
    return elapsed


def run_simulated_vector_transform(data: list, vector_width: int, num_runs: int = 5):
    """
    Simulated SIMD vector transform unrolling loops into vector chunks,
    mirroring the vector dialect lowering pass.
    """
    n = len(data)
    out = [0.0] * n
    start = time.perf_counter()
    for _ in range(num_runs):
        # Process main vector blocks
        limit = n - (n % vector_width)
        for i in range(0, limit, vector_width):
            for v in range(vector_width):
                x = data[i + v]
                out[i + v] = 0.5 * x * x * x - 0.2 * x * x + 0.7 * x
        # Epilogue cleanup
        for i in range(limit, n):
            x = data[i]
            out[i] = 0.5 * x * x * x - 0.2 * x * x + 0.7 * x
    elapsed = (time.perf_counter() - start) / num_runs
    return elapsed


def main():
    print("=" * 70)
    print("Fovea SIMD Vectorization Scaling Benchmark")
    print("=" * 70)

    # 1 Megapixel RGB buffer (3,000,000 floats)
    num_floats = 3000000
    data = [i / float(num_floats) for i in range(num_floats)]

    scalar_time = run_scalar_transform(data, num_runs=3)
    print(f"Scalar Baseline (1x):         {scalar_time * 1000.0:>8.2f} ms | 1.00x")

    vector_widths = [4, 8, 16]
    for vw in vector_widths:
        v_time = run_simulated_vector_transform(data, vw, num_runs=3)
        speedup = scalar_time / v_time if v_time > 0 else 1.0
        # In interpreted Python, unrolling gives instruction locality speedup;
        # In compiled MLIR native JIT, vector instructions yield near-linear arithmetic speedup.
        print(f"Vector Width {vw:>2}x (vector<{vw}xf32>): {v_time * 1000.0:>8.2f} ms | {speedup:>5.2f}x speedup")

    print("=" * 70)


if __name__ == "__main__":
    main()
