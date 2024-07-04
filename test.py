import os
import csv
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import *
from model import *

type = 'H_sapiens'
ratio = 1
seq_length = 41
balance_weight = True
gpu = 3
seed = 42
set_seed(seed)
device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else torch.device("cpu")

# Define paths
search_result_path = f'../search_param/result_{type}_{ratio}_bw_{balance_weight}.csv'
test_set_path = f"../../m5C_dataset/{type}/dataset_1-{ratio}/test.csv"
result_csv_path = '../result_41.csv'

# Load the search result and find the best trial
search_results = pd.read_csv(search_result_path)
filtered_results = search_results[search_results['seq_length'] == seq_length]
best_trial = filtered_results.loc[filtered_results['MCC'].idxmax()].copy()
best_trial_number = best_trial['trial_number']

# Write the 5cv_average performance to result.csv
with open(result_csv_path, 'w', newline='') as csvfile:
    fieldnames = ['type', 'ACC', 'SEN', 'SPE', 'PRE', 'MCC', 'AUROC', 'F1']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    best_trial['type'] = '5cv_average'
    writer.writerow(best_trial[fieldnames].to_dict())

# Load the test set
test_df = pd.read_csv(test_set_path)
test_df['seq'] = [extract_middle_sequence(seq, seq_length) for seq in test_df['seq']]
x_test = test_df['seq']
y_test = test_df['label']
test_dataset = One_Hot_Dataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize storage for predictions and probabilities
all_probs = []
all_preds = []
fold_metrics = []

# Load each model and evaluate on the test set
for fold_idx in range(1, 6):
    model_path = f'../search_model/{type}/dataset_1-{ratio}/seq_length_{seq_length}/bw_{balance_weight}/trial_{best_trial_number}/model_fold_{fold_idx}.pth'
    model = torch.load(model_path).to(device)
    model.eval()

    test_probs = []
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for feature, labels in test_loader:
            feature, labels = feature.to(device), labels.to(device)
            outputs = model(feature)
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            test_probs.extend(probs)
            test_preds.extend(preds)
            test_labels.extend(labels.cpu().numpy())

    all_probs.append(test_probs)
    all_preds.append(test_preds)
    # Compute metrics for this fold
    metrics = compute_metrics(test_labels, np.array(test_probs), test_preds)
    fold_metrics.append(metrics)

# Compute average metrics across all folds
avg_metrics = {key: round(np.mean([fold_metrics[fold][key] for fold in range(5)]), 4) for key in fold_metrics[0] if key != 'type'}
avg_metrics['type'] = 'test_average'

# Write the average metrics to result.csv
with open(result_csv_path, 'a', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(avg_metrics)

# Compute ensemble metrics
ensemble_probs = np.mean(all_probs, axis=0)
ensemble_preds = (ensemble_probs >= 0.5).astype(int)
ensemble_metrics = compute_metrics(test_labels, ensemble_probs, ensemble_preds)
ensemble_metrics = {key: round(value, 4) for key, value in ensemble_metrics.items()}
ensemble_metrics['type'] = 'test_ensemble'

with open(result_csv_path, 'a', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(ensemble_metrics)

# Compute voting metrics
voting_preds = np.mean(all_preds, axis=0).astype(int)
voting_metrics = compute_metrics(test_labels, None, voting_preds)
voting_metrics = {key: round(value, 4) for key, value in voting_metrics.items() if key != 'AUROC'}
voting_metrics['type'] = 'test_voting'

with open(result_csv_path, 'a', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(voting_metrics)