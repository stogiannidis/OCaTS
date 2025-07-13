import os
import pandas as pd
import transformers
import torch

import json

import optuna
import neptune
import neptune.integrations.optuna as optuna_utils

from evaluate import load  
from utils.seeding import set_seed
from utils.early_stopping import EarlyStopping

from models.mpnet import MPNetClassifier
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset  
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer

import argparse

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp

    TPU_FLAG = True
except ImportError: 
    print("Not a TPU environment.")
    TPU_FLAG = False
# loading metric
metric = load("accuracy")
BEST_LOSS = float("inf")

parser = argparse.ArgumentParser(description='Process command line arguments')
parser.add_argument('-c', '--config', type=str, help='Config file path', required=True, default="src/utils/config.json")
parser.add_argument('-t', '--train_path', type=str, help='Path to the train data', default="data/processed/banking77/best3_train.csv")
parser.add_argument('-d', '--dev_path', type=str, help='Path to the dev data', default="data/processed/banking77/dev.csv")
parser.add_argument('-m' ,'--model_dir', type=str, help='Path to the model directory', default="bin/mpnet-nn.pt")

args = parser.parse_args()
#Data directories
TRAIN_DATA_DIR = args.train_path
DEV_DATA_DIR = args.dev_path
MODEL_DIR = args.model_dir

def train_fn(model, train_loader, val_loader, optimizer, epochs, save_path):
    best_loss = float("inf")
    best_acc = 0.0
    device = xm.xla_device() if TPU_FLAG else torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using device: {device}")
    # model = model.to(device)
    patience = 10
    early_stopping = EarlyStopping(tolerance=patience, min_delta=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []
        for batch in tqdm(train_loader, desc="Training", leave=False):
            optimizer.zero_grad()

            inputs = batch[0].to(device)
            targets = batch[1].to(device)


            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            xm.mark_step() if TPU_FLAG else None

            total_loss += loss.item()
            predictions = logits.argmax(1)
            all_preds.append(predictions)
            all_labels.append(targets)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        total_acc = metric.compute(predictions=all_preds, references=all_labels)
        print(f"Epoch {epoch + 1}/{epochs} - Training loss: {total_loss / len(train_loader)} - Training accuracy: {total_acc['accuracy']}")

        model.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0
        for step, batch in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            
            with torch.no_grad():
                logits = model(inputs)
            loss = criterion(logits, targets)
            total_loss += loss.item()
            predictions = logits.argmax(1)

            all_preds.append(predictions)
            all_labels.append(targets)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        total_acc = metric.compute(predictions=all_preds, references=all_labels)
        avg_val_loss = total_loss / len(val_loader)
        print(f"Validation loss: {avg_val_loss} - Validation accuracy: {total_acc['accuracy']}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_acc = total_acc
        global BEST_LOSS
        if best_loss < BEST_LOSS:
            BEST_LOSS = best_loss
            # Handle case where save_path is a directory
            if os.path.isdir(save_path) or save_path.endswith('/'):
                save_file = os.path.join(save_path, "best_model.pt")
            else:
                save_file = save_path
            torch.save(model.state_dict(), save_file)
            print(f"Best loss over all Trials: {BEST_LOSS}, Model saved to {save_file}")
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    return best_loss, best_acc["accuracy"]


def objective(trial, train_dataset, dev_dataset):
    # Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    epochs = trial.suggest_int("epochs", 1, 100)
    # Model specific hyperparameters
    hidden_size = trial.suggest_categorical("hidden_size", [256, 512, 1024])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    activation = trial.suggest_categorical("activation", ["relu", "tanh", "silu", "hardswish", "gelu"])
    # Creating dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    # Model
    device = xm.xla_device() if TPU_FLAG else torch.device("cuda" if torch.cuda.is_available() else "mps")
    model = MPNetClassifier(hidden_size=hidden_size, dropout=dropout, activation=activation).to(device)
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr)
    # Training
    best_loss, best_acc = train_fn(model, train_loader, val_loader, optimizer, epochs, MODEL_DIR)
    return best_loss, best_acc

def preprocess_data():
    # Loading data  
    train = pd.read_csv(TRAIN_DATA_DIR)
    dev = pd.read_csv(DEV_DATA_DIR)
    # Load mpnet 
    device = xm.xla_device() if TPU_FLAG else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer('all-mpnet-base-v2').to(device)
    #Create vector embeddings
    with torch.no_grad():
        train_embeddings = model.encode(train['text'].tolist(), convert_to_tensor=True, show_progress_bar=True, batch_size=128)
        dev_embeddings = model.encode(dev['text'].tolist(), convert_to_tensor=True, show_progress_bar=True)
    #Creating tensor datasets
    train_dataset = TensorDataset(train_embeddings, torch.tensor(train['label'].tolist()))
    dev_dataset = TensorDataset(dev_embeddings, torch.tensor(dev['label'].tolist()))
    return train_dataset, dev_dataset

if __name__ == "__main__":

    set_seed(42)

    # Load data
    train_dataset, dev_dataset = preprocess_data()
    print(f"Data loaded with trainig size: {len(train_dataset)} and dev size: {len(dev_dataset)}")

    # Create study
    study = optuna.create_study(study_name="npnet-banking77", directions=["minimize", "maximize"])
    study.optimize(lambda trial: objective(trial, train_dataset, dev_dataset), n_trials=100)

    # Create results folder if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # --- Save all trials ---
    all_trials_data = []
    for trial in study.trials:
        all_trials_data.append({
            "number": trial.number,
            "state": str(trial.state),
            "values": trial.values,
            "params": trial.params
        })

    with open("results/all_trials.json", "w") as f:
        json.dump(all_trials_data, f, indent=2)
    print("✅ Saved all trials to results/all_trials.json")

    # --- Save best trials (Pareto front) ---
    best_trials_data = []
    for trial in study.best_trials:
        best_trials_data.append({
            "number": trial.number,
            "values": trial.values,
            "params": trial.params
        })

    with open("results/best_trials.json", "w") as f:
        json.dump(best_trials_data, f, indent=2)
    print("✅ Saved best trials to results/best_trials.json")