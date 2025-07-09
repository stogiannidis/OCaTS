import os
import json
import argparse
import pandas as pd
import optuna
from sentence_transformers import SentenceTransformer
from src.caches import CACHE_REGISTRY
from evaluate import load
from utils.seeding import set_seed
from models.knn import KNNClassifier
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    TPU_FLAG = True
except ImportError:
    print("Not a TPU environment.")
    TPU_FLAG = False

set_seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description='Process command line arguments')

    # Path to the configuation file
    parser.add_argument('-c', '--config', type=str, help='Config file path', required=False)
    parser.add_argument('-ct', '--cache_type', type=str, help='Type of cache to use', default="simple")
    parser.add_argument('-l', '--lambdas', type=float, help='List of lambda values', default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5], nargs="*")
    parser.add_argument('-t', '--train_path', type=str, help='Path to the train data', default="data/processed/banking77/best3_train.csv")
    parser.add_argument('-d', '--dev_path', type=str, help='Path to the dev data', default="data/processed/banking77/dev.csv")
    parser.add_argument('-s', '--study_name', type=str, help='Name of the study', default="kNN-HyperParam-Tuning-Banking77")

    args = parser.parse_args()

    return args.config, args.cache_type, args.lambdas, args.train_path, args.dev_path, args.study_name

CONFIG, CACHE_TYPE, LAMBDAS, TRAIN_PATH, DEV_PATH, STUDY_NAME = parse_args()
device = xm.xla_device() if TPU_FLAG else torch.device("cuda" if torch.cuda.is_available() else "mps")
encoder = SentenceTransformer("all-mpnet-base-v2").to(device)

train_data = pd.read_csv(TRAIN_PATH)
data = pd.read_csv(DEV_PATH)

#shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

# Create datasets and dataloaders
train_vec = encoder.encode(train_data["text"].tolist(), show_progress_bar=True, convert_to_tensor=True, device=device)
dev_embeddings = encoder.encode(data["text"].tolist(), show_progress_bar=True, convert_to_tensor=True, device=device)
dev_set = TensorDataset(dev_embeddings, torch.tensor(data["gpt-label"].tolist()))
gold_labels = data["label"].tolist()
loader = DataLoader(dev_set, batch_size=1, shuffle=False)

# Define the objective function
def objective(trial) -> float:

    # Define the hyperparameters
    e_thresh = trial.suggest_float("Entropy Threshold", 0.1, 0.72)
    d_thresh = trial.suggest_float("Distance Threshold", 0.05, 2)

    # Initialize the cache
    cache_class = CACHE_REGISTRY[CACHE_TYPE]
    cache = cache_class().to(device)
    cache.set_threshold(d_thresh)
    cache.fit(train_vec, train_data["label"].tolist())

    # Initialize the model
    model = KNNClassifier(cache)


    acc_fn = load("accuracy")
    
    preds = []
    calls = 0 

    for i, (query, label) in enumerate(tqdm(loader)):
        query = query.to(device)
        label = label.to(device)
        
        probs = model(query)

        entropy = -torch.sum(probs * torch.log(probs), dim=1)

        if torch.lt(entropy, e_thresh) and cache.is_near(query):
            pred = probs.argmax()

        else:
            pred = label
            cache.add(query, label)
            calls += 1
        
        preds.append(pred)

    # Calculate the accuracy
    preds = torch.tensor(preds)
    labels = torch.tensor(gold_labels)
    acc = acc_fn.compute(predictions=preds, references=labels)["accuracy"]
    # Calculate the discounted accuracy
    discounted_acc = acc - lambda_ * calls / len(data)
    
    kwargs ={
        "Accuracy": acc,
        "Discounted Accuracy": discounted_acc,
        "Calls": calls,
        "Entropy Threshold": e_thresh,
        "Distance Threshold": d_thresh
    }
    trial.set_user_attr("Accuracy", acc)
    trial.set_user_attr("Calls", calls)

    return discounted_acc
            
if __name__ == "__main__":

    # Initialize the best thresholds dictionary
    best_thresholds = {}
    best_thresholds[STUDY_NAME] = {}
    
    # Initialize dictionaries to store all trials and best results
    all_trials = {}
    all_trials[STUDY_NAME] = {}
    best_results = {}
    best_results[STUDY_NAME] = {}

    # For each lambda value create a study and optimize the objective function
    for lambda_ in LAMBDAS:
        # Define the Sampler
        best_thresholds[STUDY_NAME][lambda_] = {}
        all_trials[STUDY_NAME][lambda_] = {}
        sampler = optuna.samplers.TPESampler(seed=42)

        # dist_grid = np.linspace(0.05, 2, 5)
        # ent_grid = np.linspace(0.05, 0.72, 5)

        study = optuna.create_study(direction="maximize", study_name=f"{STUDY_NAME}-{lambda_}", sampler=sampler)

        # for d in dist_grid:
        #     for e in ent_grid:
        #         study.enqueue_trial({"Entropy Threshold": e, "Distance Threshold": d})

        study.optimize(objective, n_trials=5)

        # Log the best parameters to Neptune and WandB 
        best_args = {
            "Trial": study.best_trial.number,
            "Entropy Threshold": study.best_params["Entropy Threshold"],
            "Distance Threshold": study.best_params["Distance Threshold"],
            "Discounted Accuracy": study.best_value,
            "Accuracy": study.best_trial.user_attrs["Accuracy"],
            "Calls": study.best_trial.user_attrs["Calls"]
        }

        print(f"Best arguments for Lambda: {lambda_}")
        print(best_args)

        best_thresholds[STUDY_NAME][lambda_]["e_thresh"] = best_args["Entropy Threshold"]
        best_thresholds[STUDY_NAME][lambda_]["d_thresh"] = best_args["Distance Threshold"]
        
        # Store best results for this lambda
        best_results[STUDY_NAME][lambda_] = best_args
        
        # Store all trials for this lambda
        trials_data = []
        for trial in study.trials:
            trial_data = {
                "trial_number": trial.number,
                "params": trial.params,
                "value": trial.value,
                "state": trial.state.name,
                "user_attrs": trial.user_attrs
            }
            trials_data.append(trial_data)
        all_trials[STUDY_NAME][lambda_] = trials_data

    # Create results directory if it doesn't exist
    results_dir = "results/Tuning"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save all trials to a JSON file
    trials_file = os.path.join(results_dir, f"{STUDY_NAME}_all_trials.json")
    with open(trials_file, "w") as f:
        json.dump(all_trials, f, indent=4)
    print(f"All trials saved to: {trials_file}")
    
    # Save best results to a JSON file
    best_results_file = os.path.join(results_dir, f"{STUDY_NAME}_best_results.json")
    with open(best_results_file, "w") as f:
        json.dump(best_results, f, indent=4)
    print(f"Best results saved to: {best_results_file}")

    # Save the best thresholds to a json file (config file)
    if CONFIG:
        if not os.path.exists(CONFIG):
            with open(CONFIG, "w") as f:
                json.dump(best_thresholds, f, indent=4)
        else:
            with open(CONFIG, "r") as f:
                config = json.load(f)
            config.update(best_thresholds)
            with open(CONFIG, "w") as f:
                json.dump(config, f, indent=4)
        print(f"Best thresholds saved to config: {CONFIG}")
    else:
        # If no config file specified, save to results directory
        thresholds_file = os.path.join(results_dir, f"{STUDY_NAME}_best_thresholds.json")
        with open(thresholds_file, "w") as f:
            json.dump(best_thresholds, f, indent=4)
        print(f"Best thresholds saved to: {thresholds_file}")
