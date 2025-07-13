# OCaTSPlus Benchmarking Suite

This README describes how to benchmark the KNN + cache system using various cache eviction policies (`simple`, `lfu`, `lru`) and lambda thresholds.

## Overview

The benchmarking script measures the performance of different cache strategies in the OCaTSPlus framework. It computes key metrics such as:

- Latency (mean, p95, p99)
- Cache hit rate
- LLM call frequency
- Query throughput (queries per second)
- Total evaluation time

The results are saved in JSON format for analysis and comparison.

---

## How to Run

### Prerequisites

Ensure you have the following installed:
- Python 3.9+
- PyTorch
- Required dependencies listed in your `requirements.txt`
### Run the Benchmark

From the project root directory:

```bash
PYTHONPATH=. python src/benchmark.py
