import time
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset

from src.models.knn import KNNClassifier
from src.caches import CACHE_REGISTRY
from utils.seeding import set_seed

def benchmark_cache_model(cache_type: str, lambda_: float, output_dir="results/benchmark", use_tuned_thresholds=True):
    set_seed(42)

    # Config
    BATCH_SIZE = 1
    DATA_PATH = "data/raw/banking77/test.csv"
    OUTPUT_FILE = f"{output_dir}/benchmark_knn_{cache_type}_lambda{lambda_}.json"
    
    # Load tuned thresholds if available
    if use_tuned_thresholds:
        try:
            with open("results/Tuning/lambda-tuning-cache_best_results.json", "r") as f:
                tuning_results = json.load(f)
            entropy_threshold = tuning_results["lambda-tuning-cache"][str(lambda_)]["Entropy Threshold"]
        except (FileNotFoundError, KeyError):
            print(f"Warning: No tuned threshold found for lambda={lambda_}, using default 0.7")
            entropy_threshold = 0.7
    else:
        entropy_threshold = 0.7

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data and encoder
    df = pd.read_csv(DATA_PATH)
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    encoder = SentenceTransformer("all-mpnet-base-v2").to(device)
    embeddings = encoder.encode(texts, convert_to_tensor=True, device=device, show_progress_bar=True)
    dataset = TensorDataset(embeddings, torch.tensor(labels))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model + cache
    cache_cls = CACHE_REGISTRY[cache_type]
    cache = cache_cls().to(device)
    cache.fit(embeddings, labels)
    model = KNNClassifier(cache)

    latencies = []
    cache_hits = 0
    llm_calls = 0
    total_queries = len(dataset)

    for query_tensor, label_tensor in tqdm(loader, desc=f"Benchmarking {cache_type} λ={lambda_}"):
        query_tensor = query_tensor.to(device)
        label_tensor = label_tensor.to(device)

        start = time.perf_counter()
        probs = model(query_tensor)
        end = time.perf_counter()

        latency = end - start
        latencies.append(latency)

        # Cache hit logic using tuned thresholds
        entropy = -torch.sum(probs * torch.log(probs), dim=1)
        is_hit = torch.lt(entropy, entropy_threshold) and cache.is_near(query_tensor)

        if is_hit:
            cache_hits += 1
        else:
            llm_calls += 1
            cache.add(query_tensor, label_tensor)

    latencies_np = np.array(latencies)
    summary = {
        "cache_type": cache_type,
        "lambda": lambda_,
        "entropy_threshold": entropy_threshold,
        "mean_latency_sec": float(np.mean(latencies_np)),
        "p95_latency_sec": float(np.percentile(latencies_np, 95)),
        "p99_latency_sec": float(np.percentile(latencies_np, 99)),
        "cache_hit_rate": cache_hits / total_queries,
        "throughput_qps": total_queries / np.sum(latencies_np),
        "total_queries": total_queries,
        "total_llm_calls": llm_calls,
        "total_cache_hits": cache_hits,
        "total_duration_sec": float(np.sum(latencies_np))
    }

    # Save JSON
    os.makedirs(output_dir, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    # Benchmark different cache types and configurations:
    # - First parameter: cache type ("simple", "lfu", "lru")
    #   * "simple": No eviction policy (original model)
    #   * "lfu": Least Frequently Used eviction policy
    #   * "lru": Least Recently Used eviction policy
    # - Second parameter: lambda value (distance threshold)[0.05 0.1 0.2 0.3 0.4 0.5 0.6]
    # - Optional: use_tuned_thresholds=False to use original 0.7 entropy threshold
    
    benchmark_cache_model("lfu", 0.1)
    benchmark_cache_model("simple", 0.1)