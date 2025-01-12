import time
import psutil
import csv
from memory_profiler import memory_usage

def measure_time_and_memory(func, *args):
    """Measures execution time and memory usage for a given function."""
    start_time = time.time()
    mem_usage = memory_usage((func, args), interval=0.1, timeout=None)
    end_time = time.time()
    result = func(*args)  # To include memory usage calculation after the function runs
    cpu_usage = psutil.cpu_percent(interval=1)
    return result, end_time - start_time, max(mem_usage), cpu_usage

def log_to_csv(file_name, headers, rows):
    """Logs benchmark data to a CSV file."""
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)