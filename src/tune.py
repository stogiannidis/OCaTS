import os
import json
import argparse

import numpy as np
import pandas as pd
import optuna
from sentence_transformers import SentenceTransformer

from models.cache import Cache
from evaluate import load
from utils.seeding import set_seed
from models.mpnet import MPNetClassifier
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    TPU_FLAG = True
except ImportError:
    TPU_FLAG = False

set_seed(42)
    
def parse_args():
    parser = argparse.ArgumentParser(description='Process command line arguments')

    # Path to the configuation file
    parser.add_argument('-c', '--config', type=str, help='Config file path', required=True, default="src/utils/config.json")
    parser.add_argument('-l', '--lambdas', type=float, help='List of lambda values', default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5], nargs="*")
    parser.add_argument('-t', '--train_path', type=str, help='Path to the train data', default="data/processed/banking77/best3_train.csv")
    parser.add_argument('-d', '--dev_path', type=str, help='Path to the dev data', default="data/processed/banking77/dev.csv")
    parser.add_argument('-s', '--study_name', type=str, help='Name of the study', default="kNN-HyperParam-Tuning-IMDB")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    return args.config, config, args.lambdas, args.train_path, args.dev_path, args.study_name

CONFIG_PATH, CONFIG, LAMBDAS, TRAIN_PATH, DEV_PATH, STUDY_NAME = parse_args()

device = xm.xla_device() if TPU_FLAG else torch.device("cuda" if torch.cuda.is_available() else "mps")
encoder = SentenceTransformer("all-mpnet-base-v2").to(device)

train_data = pd.read_csv(TRAIN_PATH)
data = pd.read_csv(DEV_PATH)

#shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

test_embeddings = encoder.encode(train_data["text"].tolist(), show_progress_bar=True, convert_to_tensor=True, device=device)
dev_embeddings = encoder.encode(data["text"].tolist(), show_progress_bar=True, convert_to_tensor=True, device=device)
dev_set = TensorDataset(torch.tensor(dev_embeddings), torch.tensor(data["gpt-label"].tolist()))
gold_labels = data["label"].tolist()
loader = DataLoader(dev_set, batch_size=1, shuffle=True)



def train(model, data, config):

    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["mpnet"]["lr"])
    loss_fn = torch.nn.CrossEntropyLoss()
    loader = DataLoader(data, batch_size=64, shuffle=True)

    for epoch in range(2):
        for v, l in tqdm(loader, desc="Training", leave=False):
            v = v.to(device)
            l = l.to(device)

            optimizer.zero_grad()
            logits = model(v)
            loss = loss_fn(logits, l)
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step() 
            xm.mark_step() if TPU_FLAG else None

def objective(trial, config) -> float:
    
    model_checkpoint = config["mpnet"]["checkpoint"]
    hidden_size = config["mpnet"]["hidden_size"]
    activation = config["mpnet"]["activation"]
    dropout = config["mpnet"]["dropout"]

    model = MPNetClassifier(hidden_size, dropout, activation)    
    model.load_state_dict(torch.load(model_checkpoint))
    model.requires_grad_(True)
    model.to(device)
    model.eval()
    
    cache = Cache()
    cache.fit(test_embeddings, train_data["label"].tolist())
    cache.to(device)

    e_thresh = trial.suggest_float("Entropy Threshold", 0.0, 4.9)
    d_thresh = trial.suggest_float("Distance Threshold", 0.0, 2.0)    
    cache.set_threshold(d_thresh)

    acc_fn = load("accuracy")

    preds = []
    calls = 0
    calls_to_retrain = 100

    for v, l in tqdm(loader, desc="Inference", leave=False):
        v = v.to(device)
        l = l.to(device)

        logits = model(v)
        probs = torch.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs), dim=1)

        #Check if the entropy is less than the threshold and if the vector is within the distance threshold
        if torch.lt(entropy, e_thresh) and cache.is_near_wcentroid(v):
            preds.append(logits.argmax(dim=1))
        else:
            cache.add(v, l)
            preds.append(l)
            calls += 1
        
        #retrain the model after every 100 calls
        if calls == calls_to_retrain:
            #Get last 100 predictions and labels
            data100 = cache.get_last_100_added()
            # Train the model on the last 100 data points
            train(model, data100, config)
            calls_to_retrain += 100 #Increase the number of calls to retrain by 100            

    acc = acc_fn.compute(predictions=preds, references=gold_labels)["accuracy"]
    trial.set_user_attr("Accuracy", acc)
    trial.set_user_attr("Calls", calls)
    disc_acc = acc - lamda * calls / len(data)
    

    # Log the results to Neptune for tracking and analysis
    kwargs = {
        "Accuracy": acc,
        "Calls": calls,
    }

    trial.set_user_attr("Accuracy", acc)
    trial.set_user_attr("Calls", calls)

    return disc_acc



if __name__ == "__main__":

    best_thresholds = {}
    best_thresholds[STUDY_NAME] = {}

    for lamda in LAMBDAS:
        best_thresholds[STUDY_NAME][lambda_] = {}
        
        # Define the Sampler
        sampler = optuna.samplers.CmaEsSampler()

        dist_grid = np.linspace(0.05, 2, 10)
        ent_grid = np.linspace(0.1, 4.9, 10)

        study = optuna.create_study(direction="maximize", study_name=f"{STUDY_NAME}-{lamda}", sampler=sampler)

        for d in dist_grid:
            for e in ent_grid:
                study.enqueue_trial({"Entropy Threshold": e, "Distance Threshold": d})

        study.optimize(lambda trial: objective(trial, CONFIG), n_trials=150, callbacks=[neptune_callback])

        best_args = {
            "Trial": study.best_trial.number,
            "Entropy Threshold": study.best_params["Entropy Threshold"],
            "Distance Threshold": study.best_params["Distance Threshold"],
            "Discounted Accuracy": study.best_value,
            "Accuracy": study.best_trial.user_attrs["Accuracy"],
            "Calls": study.best_trial.user_attrs["Calls"]
        }
        
        best_thresholds[STUDY_NAME][lambda_]["e_thresh"] = best_args["Entropy Threshold"]
        best_thresholds[STUDY_NAME][lambda_]["d_thresh"] = best_args["Distance Threshold"]

    # Save the best thresholds to a json file
    if not os.path.exists(CONFIG):
        with open(CONFIG, "w") as f:
            json.dump(best_thresholds, f, indent=4)
    with open(CONFIG, "r") as f:
        config = json.load(f)
        config.update(best_thresholds)
