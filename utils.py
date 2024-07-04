import os
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, f1_score, confusion_matrix

# Set a random seed to ensure reproducibility
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Disabling certain features in PyTorch for deterministic behavior
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set random seed as {seed}")


# Convert a nucleotide sequence to one-hot encoding
def one_hot(seq):
    a_vec = [1 if nucleotide == 'A' else 0 for nucleotide in seq]
    c_vec = [1 if nucleotide == 'C' else 0 for nucleotide in seq]
    g_vec = [1 if nucleotide == 'G' else 0 for nucleotide in seq]
    u_vec = [1 if nucleotide == 'U' or nucleotide == 'T' else 0 for nucleotide in seq]

    return a_vec, c_vec, g_vec, u_vec


# Custom dataset class for One-Hot encoding of sequences
class One_Hot_Dataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        a, c, g, u = one_hot(seq)
        return torch.tensor([a, c, g, u], dtype=torch.float), self.labels[idx]
    

def extract_middle_sequence(sequence, seq_length):
    start_idx = (len(sequence) - seq_length) // 2
    return sequence[start_idx:start_idx + seq_length]


def compute_metrics(y_true, y_prob, y_pred):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else None  # Set AUROC to None if y_prob is None
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sen = tp / (tp + fn) if tp + fn > 0 else 0  # Sensitivity
    spe = tn / (tn + fp) if tn + fp > 0 else 0  # Specificity
    pre = tp / (tp + fp) if tp + fp > 0 else 0  # Precision

    # Organize the metrics into a dictionary
    metrics = {
        'ACC': acc,
        'SEN': sen,
        'SPE': spe,
        'PRE': pre,
        'MCC': mcc,
        'AUROC': auc,  # AUROC will be None if y_prob is None
        'F1': f1
    }

    return metrics


# Function to train the model
def train_model(model, train_loader, valid_loader, optimizer, scheduler, epochs, max_grad_norm, device, patience, model_dir, result_dir, fold_idx, trial_num, ratio, balance_weight):
    best_mcc = -float('inf')
    best_metric = [0, 0, 0, 0, 0, 0]
    no_improvement_count = 0
    zero_mcc_count = 0
    # Paths to save model and results
    model_dir = os.path.join(model_dir, f"trial_{trial_num}/")
    result_dir = os.path.join(result_dir, f"trial_{trial_num}/")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"model_fold_{fold_idx}.pth")
    prob_path = os.path.join(result_dir, f"probs_fold_{fold_idx}.csv")

    # Set the criterion based on balance_ratio
    if balance_weight:
        pos_weight = torch.tensor([1.0], device=device)  # no weighting
    else:
        pos_weight = torch.tensor([ratio], device=device)  # weighted by ratio

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        # Training loop
        for feature, labels in train_loader:
            feature, labels = feature.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(feature)
            # loss = criterion(outputs.squeeze(), labels.float())
            # loss.backward()
            # # Gradient clipping for stability
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            # optimizer.step()
            # total_loss += loss.item()

            # 调整输出的形状，使其与标签的形状匹配
            if outputs.dim() > 1 and outputs.size(1) == 1:
                outputs = outputs.view(-1)
            elif outputs.dim() == 0:
                outputs = outputs.view(1)
            labels = labels.float()

            try:
                loss = criterion(outputs, labels)
                loss.backward()

                # 为了稳定性进行梯度裁剪
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

                total_loss += loss.item()
            except ValueError as e:
                print(f"计算损失时发生错误: {e}")
                print(f"输出: {outputs}")
                print(f"标签: {labels}")

        scheduler.step()

        # Validation loop
        model.eval()
        valid_probs = []
        valid_preds = []
        valid_labels = []

        with torch.no_grad():
            for feature, labels in valid_loader:
                feature, labels = feature.to(device), labels.to(device)
                outputs = model(feature)

                preds = torch.round(torch.sigmoid(outputs)).detach()
                probs = torch.sigmoid(outputs).detach()

                valid_probs.extend(probs.cpu().numpy())
                valid_preds.extend(preds.cpu().numpy())
                valid_labels.extend(labels.cpu().numpy())

        valid_metric = compute_metrics(valid_labels, np.array(valid_probs), valid_preds)

        # Update best model if improvement is seen
        if valid_metric['MCC'] > best_mcc or epoch == 0:
            best_mcc = valid_metric['MCC']
            best_metric = valid_metric
            no_improvement_count = 0
            torch.save(model.cpu(), model_path)
            model.to(device)

            print(f"Epoch {epoch} | Fold {fold_idx}")
            print(best_metric)
            # Save best model's validation predictions, probabilities, and labels
            prob_results = pd.DataFrame({
                'prob': [prob[0] for prob in valid_probs],  
                'pred': [str(int(pred)) for pred in valid_preds],  
                'label': [str(int(lbl)) for lbl in valid_labels]  
            })
            prob_results.to_csv(prob_path, index=False)

        else:
            no_improvement_count += 1
        
        if valid_metric['MCC'] == 0:
            zero_mcc_count += 1

        # Early stopping condition
        if no_improvement_count >= patience or zero_mcc_count >= 5:
            print(f"Early stopping triggered!!")
            return best_metric
        
    return best_metric




