import time
import psutil
import csv
import os

from matrixAlgorithms import *
from distributedMatrixMultiplication import *


class Benchmark:
    @staticmethod
    def measure_performance(algorithm, matrix_A, matrix_B, num_workers=None):
        start_time = time.time()
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / (1024 * 1024)  # Memory in MB

        if algorithm.__name__ == "distributed_matrix_multiplication":
            result = algorithm(matrix_A, matrix_B, num_workers)
        else:
            result = algorithm(matrix_A, matrix_B)

        execution_time = (time.time() - start_time) * 1000  # Time in ms
        memory_after = process.memory_info().rss / (1024 * 1024)  # Memory in MB
        memory_used = memory_after - memory_before

        cpu_usage = psutil.cpu_percent(interval=None)  # CPU usage as percentage
        network_overhead = 0
        data_transfer_time = 0
        nodes_used = num_workers if num_workers else 1

        return {
            "Matrix Size": f"{matrix_A.shape[0]}x{matrix_B.shape[1]}",
            "Execution Time (ms)": execution_time,
            "Memory Used (MB)": memory_used,
            "CPU Usage (%)": cpu_usage,
            "Nodes Used": nodes_used,
            "Network Overhead (ms)": network_overhead,
            "Data Transfer Time (ms)": data_transfer_time,
        }

    @staticmethod
    def write_results_to_csv(results, filename):
        keys = results[0].keys()
        with open(filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)


def main():
    matrix_sizes = [(50, 50, 50), (100, 100, 100), (800, 800, 800), (1024, 1024, 1024), (2048, 2048, 2048)]
    num_workers = 4

    algorithms = {
        "Distributed": distributed_matrix_multiplication,
        "Basic": multiply_matrices_basic,
        "Parallel": multiply_matrices_parallel,
    }

    for algo_name, algo in algorithms.items():
        results = []
        print(f"Running benchmark for {algo_name} algorithm...")

        for size in matrix_sizes:
            rows_A, cols_A, cols_B = size
            matrix_A = np.random.randint(1, 10, (rows_A, cols_A))
            matrix_B = np.random.randint(1, 10, (cols_A, cols_B))

            print(f"Matrix Size: {rows_A}x{cols_B}")

            benchmark_result = Benchmark.measure_performance(
                algo, matrix_A, matrix_B, num_workers=num_workers if algo_name == "Distributed" else None
            )

            results.append(benchmark_result)

            print(f"Results: {benchmark_result}")

        csv_filename = f"{algo_name}_benchmark_results.csv"
        Benchmark.write_results_to_csv(results, csv_filename)

        print(f"Results saved to {csv_filename}\n")


if __name__ == "__main__":
    main()
