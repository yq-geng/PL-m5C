from utils import *
from model import *
import os
import warnings
import torch
import numpy as np
from datetime import datetime
import torch.nn as nn
import csv
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import CosineAnnealingLR
import optuna

# Define parameters
type = 'H_sapiens'
ratio = 10
seq_length = 81
balance_weight = False

gpu = 2
seed = 42
epochs = 500
patience = 10

# Set device and warnings
warnings.filterwarnings('ignore')
set_seed(seed)
device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else torch.device("cpu")

# Objective function for Optuna
def objective(trial, seq_length):
    # Time
    now = datetime.now()
    start_time = now.strftime("%m-%d %H:%M:%S")
    
    result_dir = f'../search_result/{type}/dataset_1-{ratio}/seq_length_{seq_length}/bw_{balance_weight}/'
    model_dir = f'../search_model/{type}/dataset_1-{ratio}/seq_length_{seq_length}/bw_{balance_weight}/'
    search_param_dir = f'../search_param/'
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(search_param_dir, exist_ok=True)
    
    search_result_path = f'{search_param_dir}result_{type}_{ratio}_bw_{balance_weight}.csv'
    train_set_path = f"../../m5C_dataset/{type}/dataset_1-{ratio}/train.csv"
    # 参数搜索
    params = {
        'max_grad_norm': trial.suggest_int('max_grad_norm', 1, 10),
        'weight_decay': trial.suggest_loguniform('weight_decay', 1e-8, 1e-4),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-6, 1e-2),
        'T_max': trial.suggest_int('T_max', 10, 100),
        'drop_out': trial.suggest_uniform('drop_out', 0.0, 0.5),
        'batch_size': 64,
        'num_layers': trial.suggest_int('num_layers', 1, 4),
        'num_heads': trial.suggest_categorical('num_heads', [2, 4, 8])
    }

    # 打印log
    print(f"Trial Number: {trial.number}")
    
    df = pd.read_csv(train_set_path)
    df['seq'] = [extract_middle_sequence(seq, seq_length) for seq in df['seq']]
    x = df['seq']
    y = df['label']
    print(f"Dataset read from {train_set_path} with total {len(df)} entries.")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    metrics = []
    fold_idx = 0

    for train_index, val_index in skf.split(x, y):
        fold_idx += 1
        x_train, x_val = x.iloc[train_index].reset_index(drop=True), x.iloc[val_index].reset_index(drop=True)
        y_train, y_val = y.iloc[train_index].reset_index(drop=True), y.iloc[val_index].reset_index(drop=True)
        train_dataset = One_Hot_Dataset(x_train, y_train)
        val_dataset = One_Hot_Dataset(x_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
        
        model = RNASeqClassifier(num_layers=params['num_layers'],
                                 num_heads=params['num_heads'], 
                                 drop_out=params['drop_out']).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        scheduler = CosineAnnealingLR(optimizer, T_max=params['T_max'])

        eval_metric = train_model(model, train_loader, val_loader, optimizer, scheduler, 
                                  max_grad_norm=params['max_grad_norm'], epochs=epochs, 
                                  device=device, patience=patience, result_dir=result_dir, model_dir=model_dir,
                                  fold_idx=fold_idx, trial_num=trial.number, ratio=ratio, balance_weight=balance_weight)
        
        metrics.append(eval_metric)
        if(eval_metric['MCC'] < 0.1):
            print(f"Fold_{fold_idx}'s MCC is less than 0.1, trial_{trial.number} early stopping!")
            break
    
    # 计算每个评估指标的平均值
    avg_metric = {key: np.mean([m[key] for m in metrics]) for key in metrics[0].keys()}

    # 保留4位小数
    avg_metric = {key: round(value, 4) for key, value in avg_metric.items()}
    # trial基本信息
    trial_info = {
        "type": type,
        "ratio": ratio,
        "seq_length": seq_length,
        "start_time": start_time,
        "trial_number": trial.number,  
    }
    result = {**trial_info, **avg_metric, **params}

    file_exists = os.path.isfile(search_result_path)
    with open(search_result_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

    return avg_metric['MCC']

# Loop over different sequence lengths
# for seq_length in range(21, 302, 20):

print(f"Starting optimization for sequence length {seq_length}")
sampler = optuna.samplers.TPESampler(seed=seed)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(lambda trial: objective(trial, seq_length), n_trials=100)  # 设置试验次数

