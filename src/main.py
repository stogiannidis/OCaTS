# External library imports
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer  # For sentence embeddings

# Local module imports
from utils.seeding import set_seed  # For setting the seed
from models.knn import KNNClassifier  # KNN classifier
from models.mpnet import MPNetClassifier  # MPNet classifier
from src.caches.base import CACHE_REGISTRY  # For caching functionality

# Third-party library imports
from tqdm.auto import tqdm  # For progress bar
import torch  # PyTorch deep learning library
from torch.utils.data import DataLoader, TensorDataset # For creating data loaders
import numpy as np

# Standard library imports
import pandas as pd  # For data manipulation and analysis
from evaluate import load # For loading the evaluation metrics
import argparse  # For command-line argument parsing
import json  # For working with JSON data

import os

# TPU-specific imports
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    TPU_FLAG = True
except ImportError:
    TPU_FLAG = False

#Parse the command-line arguments
parser = argparse.ArgumentParser(description='Process command line arguments')

# Path to the configuation file
parser.add_argument('-c', '--config', type=str, help='Config file path', default="src/utils/config.json")
parser.add_argument('-m', '--model', type=str, help='Model to evaluate', default="knn")
parser.add_argument('-ct', '--cache_type', type=str, help='Type of cache to use', default="simple")
parser.add_argument('-l', '--lambdas', type=float, help='List of lambda values', default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5], nargs="*")
parser.add_argument('-s', '--seed', type=int, help='Seed for reproducibility', default=54321)
parser.add_argument('-t', '--train_path', type=str, help='Path to the training data', default="data/processed/banking77/best3_train.csv")
parser.add_argument('-d', '--test_path', type=str, help='Path to the test data', default="data/raw/banking77/test.csv")


args = parser.parse_args()


# CONSTANTS
CONFIG_PATH = args.config
MODEL_NAME = args.model
CACHE_TYPE = args.cache_type
LAMBDAS = args.lambdas
SEED = args.seed
TRAIN_PATH = args.train_path
TEST_PATH = args.test_path

# Set the seed
set_seed(seed=SEED)

# Set the device
device = xm.xla_device() if TPU_FLAG else torch.device("cuda" if torch.cuda.is_available() else "mps")

# Initialize the encoder
encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").to(device)
encoder.eval()

# Load the data
train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)

# Create embeddings for the data
with torch.no_grad():
    train_embeddings = encoder.encode(train_data["text"].tolist(), show_progress_bar=True, convert_to_tensor=True, device=device)
    test_embeddings = encoder.encode(test_data["text"].tolist(), show_progress_bar=True, convert_to_tensor=True, device=device)
train_labels = train_data["label"].tolist()
test_labels = test_data["label"].tolist()
test_gpt_labels = test_data["gpt-label"].tolist()

# Define the evaluation metrics
eval_fn = load("accuracy")
def train(model: torch.nn.Module, data: DataLoader, config: dict) -> None:
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["mpnet_nn"]["lr"])
    # Define the loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    # Define the number of epochs
    epochs = 4
    model.train()
    # Iterate over the epochs
    for epoch in range(epochs):
        # Iterate over the batches
        for step, (v, l) in enumerate(tqdm(data, desc=f"Epoch {epoch+1}", leave=False)):
            # Move the data to the device
            v = v.to(device)
            l = l.to(device)

            # Get the logits
            logit = model(v)
            # Calculate the loss
            loss = loss_fn(logit, l)
            # Backpropagate the loss
            loss.backward()
            # Update the parameters
            optimizer.step()
            # Reset the gradients
            optimizer.zero_grad()
            # Mark the step if the model is on TPU
            xm.mark_step() if TPU_FLAG else None


def main(lamda: float, config: dict, model_name: str, cache_type: str) -> tuple:
    RETRAIN_NUM = 100
    test_set = TensorDataset(test_embeddings, torch.tensor(test_labels), torch.tensor(test_gpt_labels))
    num_runs = 5

    e_thresh = config[model_name]["e_thresh"][str(lamda)]
    d_thresh = config[model_name]["d_thresh"][str(lamda)]

    avg_accs, avg_disc_accs, avg_calls = [], [], []

    for run in range(num_runs):
        accs, disc_accs, calls_ = [], [], []

        cache_class = CACHE_REGISTRY[cache_type]
        cache = cache_class(train_embeddings, train_labels, d_thresh)

        if model_name == "knn":
            model = KNNClassifier(cache=cache)
        elif model_name == "mpnet":
            hidden_size = config[model_name]["hidden_size"]
            activation = config[model_name]["activation"]
            dropout = config[model_name]["dropout"]
            checkpoint = config[model_name]["checkpoint"]
            model = MPNetClassifier(hidden_size, dropout, activation).to(device)
            model.load_state_dict(torch.load(checkpoint))
            model.eval()
        else:
            print("Invalid model. Exiting...")
            exit()

        online_stream = DataLoader(test_set, batch_size=1, shuffle=True)

        predictions, labels = [], []
        calls = 0

        for step, (v, gt, l) in enumerate(tqdm(online_stream, desc=f"Online Eval - Lambda: {lamda} - Run: {run+1}", leave=False)):
            v, l, gt = v.to(device), l.to(device), gt.to(device)
            cls_probs = model(v)
            if model_name == "mpnet":
                cls_probs = torch.softmax(cls_probs, dim=1)
            pred = torch.argmax(cls_probs)

            entropy = -torch.sum(cls_probs * torch.log(cls_probs))

            if torch.gt(entropy, e_thresh) or not cache.is_near(v):
                cache.add(v, l)
                pred = l
                calls += 1

            predictions.append(pred)
            labels.append(gt)

            if model_name == "mpnet" and calls % RETRAIN_NUM == 0 and calls != 0:
                last_100 = cache.get_last_p_added(p=100)
                loader = DataLoader(last_100, batch_size=32, shuffle=True)
                train(model, loader, config)
                RETRAIN_NUM += 100

            accuracy = eval_fn.compute(predictions=predictions, references=labels)["accuracy"]
            disc_acc = accuracy - lamda * calls / len(predictions)
            accs.append(accuracy)
            disc_accs.append(disc_acc)
            calls_.append(calls)

        avg_accs.append(accs)
        avg_disc_accs.append(disc_accs)
        avg_calls.append(calls_)

    avg_accs = np.mean(avg_accs, axis=0)
    avg_disc_accs = np.mean(avg_disc_accs, axis=0)
    avg_calls = np.mean(avg_calls, axis=0)

    # === Save results ===
    result_dir = os.path.join("results", "Evaluate")
    os.makedirs(result_dir, exist_ok=True)

    npz_path = os.path.join(result_dir, f"results_{model_name}_{cache_type}_lambda{lamda}.npz")
    np.savez(npz_path, accs=avg_accs, disc_accs=avg_disc_accs, calls=avg_calls)
    print(f"✅ Saved .npz results to {npz_path}")

    # Optional JSON summary
    json_summary = {
        "final_accuracy": float(avg_accs[-1]),
        "final_discounted_accuracy": float(avg_disc_accs[-1]),
        "total_calls": int(avg_calls[-1])
    }
    json_path = os.path.join(result_dir, f"summary_{model_name}_{cache_type}_lambda{lamda}.json")
    with open(json_path, "w") as f:
        json.dump(json_summary, f, indent=2)
    print(f"✅ Saved JSON summary to {json_path}")

    return avg_accs, avg_disc_accs, avg_calls

def plot_results(accs: list, disc_accs: list, calls: list, lambdas: list, model_name: str) -> None:
    # Plot the results
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i, (accs, disc_accs, calls) in enumerate(zip(accs, disc_accs, calls)):
        ax[0].plot(accs, label=f"Lambda: {lambdas[i]}")
        ax[1].plot(disc_accs, label=f"Lambda: {lambdas[i]}")
        ax[2].plot(calls, label=f"Lambda: {lambdas[i]}")

    ax[0].set_title(f"Accuracy vs. Incoming instances - {model_name}")
    ax[0].set_xlabel("Incoming instances")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()

    ax[1].set_title(f"Discrepancy Accuracy vs. Incoming instances - {model_name}")
    ax[1].set_xlabel("Incoming instances")
    ax[1].set_ylabel("Discrepancy Accuracy")

    ax[2].set_title(f"Number of calls vs. Incoming instances - {model_name}")
    ax[2].set_xlabel("Incoming instances")
    ax[2].set_ylabel("Number of calls")
    ax[2].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Iterate over the lambda values
    config = json.load(open(CONFIG_PATH, "r"))
    total_accs, total_disc_accs, total_calls = [], [], []
    for lamda in LAMBDAS:
        # Run the main function
        accs, disc_accs, calls = main(lamda, config, MODEL_NAME, CACHE_TYPE)
        # Append the results
        total_accs.append(accs)
        total_disc_accs.append(disc_accs)
        total_calls.append(calls)

    # Plot the results
    plot_results(total_accs, total_disc_accs, total_calls, LAMBDAS, MODEL_NAME)