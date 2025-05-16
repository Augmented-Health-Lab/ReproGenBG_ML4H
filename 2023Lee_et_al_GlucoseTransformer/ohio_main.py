import sys
import os
import argparse
# Add the parent directory to the path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import mean_squared_error

# Import from local src package
from src.models import TransformerEncoder_version2, PositionalEncoding
from src.data import (TimeSeriesDataset, load_ohio_series_train, create_5fold_splits, create_5fold_splits_T1DEXI,
                     convert_to_datetime, load_train_data_by_fold, load_train_data_by_fold_T1DEXI,
                     split_into_continuous_series,
                     create_population_splits, create_train_val_datasets)
from src.train import train_model, evaluate_model
from src.utils import (save_model, load_model, load_model_population,
                      evaluate_and_save_metrics, evaluate_and_save_metrics_population,
                      evaluate_and_save_metrics_diatrend, evaluate_and_save_metrics_T1DEXI)


def setup_directories():
    """Create necessary directories for saving models and evaluations."""
    model_dir = './saved_models/'
    evaluation_dir = './evaluation/'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(evaluation_dir, exist_ok=True)
    return model_dir, evaluation_dir

def load_population_splits(data_dir):
    """Load population splits for training and testing."""
    folder_path_train_2018 = os.path.join(data_dir, "./OhioT1DM/2018/train")
    folder_path_train_2020 = os.path.join(data_dir, "./OhioT1DM/2020/train")
    train_files_2018 = [os.path.join(folder_path_train_2018, f) for f in os.listdir(folder_path_train_2018) if f.endswith('.xml')]
    train_files_2020 = [os.path.join(folder_path_train_2020, f) for f in os.listdir(folder_path_train_2020) if f.endswith('.xml')]

    folder_path_test_2018 = os.path.join(data_dir, "./OhioT1DM/2018/test")
    folder_path_test_2020 = os.path.join(data_dir, "./OhioT1DM/2020/test")
    test_files_2018 = [os.path.join(folder_path_test_2018, f) for f in os.listdir(folder_path_test_2018) if f.endswith('.xml')]
    test_files_2020 = [os.path.join(folder_path_test_2020, f) for f in os.listdir(folder_path_test_2020) if f.endswith('.xml')]

    return create_population_splits(
        folder_path_train_2018, folder_path_train_2020, train_files_2018, train_files_2020,
        folder_path_test_2018, folder_path_test_2020, test_files_2018, test_files_2020
    )

def train_population_model(population_splits, model_dir, past_sequence_length, future_offset, batch_size, max_interval_minutes):
    """Train the population model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model
    model = TransformerEncoder_version2(
        past_seq_len=past_sequence_length,
        num_layers=1,
        d_model=512,
        nhead=4,
        input_dim=1,
        dropout=0.2
    ).to(device)

    # Load and process training data
    train_dfs = [load_ohio_series_train(file, "glucose_level", "value") for file in population_splits['train']]
    for df in train_dfs:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Create datasets
    train_series_list = []
    for df in train_dfs:
        series_list = split_into_continuous_series(df, past_sequence_length, future_offset, max_interval_minutes)
        train_series_list.extend(series_list)

    train_dataset, val_dataset = create_train_val_datasets(
        train_series_list, train_ratio=0.8, past_seq_len=past_sequence_length, future_offset=future_offset
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Train the model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=200,
        learning_rate=1e-3
    )

    # Save the trained model
    save_model(model, f'population_sh{past_sequence_length}', save_dir=os.path.join(model_dir, 'saved_models_original_ohio/'))

def evaluate_population_model(population_splits, model_dir, evaluation_dir, past_sequence_length, future_offset, batch_size, max_interval_minutes):
    """Evaluate the population model."""
    model = load_model_population(
        f'population_sh{past_sequence_length}', past_sequence_length, save_dir=os.path.join(model_dir, 'saved_models_original_ohio/')
    )

    # Evaluate on the entire test set
    metrics = evaluate_and_save_metrics_population(
        model=model,
        test_file_paths=population_splits['test'],
        save_dir=os.path.join(evaluation_dir, 'evaluation_metrics_original_ohio'),
        past_sequence_length=past_sequence_length,
        future_offset=future_offset,
        batch_size=batch_size,
        max_interval_minutes=max_interval_minutes
    )

    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"MAPE: {metrics['mape']:.2f}%")

    # Evaluate on individual test files
    test_eval = []
    for test_file in population_splits['test']:
        metrics = evaluate_and_save_metrics(
            model=model,
            test_file_path=test_file,
            save_dir=os.path.join(evaluation_dir, f'evaluation_metrics_original_ohio/individual_sh{past_sequence_length}/'),
            past_sequence_length=past_sequence_length,
            future_offset=future_offset,
            batch_size=batch_size,
            max_interval_minutes=max_interval_minutes
        )
        test_id = os.path.basename(test_file).split('-')[0]
        test_eval.append([test_id, round(metrics['rmse'], 2), round(metrics['mae'], 2), round(metrics['mape'], 2)])

    # Save individual evaluation results
    df = pd.DataFrame(test_eval, columns=['test patient', 'RMSE', 'MAE', 'MAPE'])
    df.to_csv(os.path.join(evaluation_dir, f'evaluation_metrics_original_ohio/individual_test_eval_sh{past_sequence_length}.csv'), index=False)

def main(data_dir):
    """Main function to train and evaluate the population model."""
    # Setup directories
    model_dir, evaluation_dir = setup_directories()

    # Data directory
    # data_dir = '../../datasets/OhioT1DM/'

    # Load population splits
    population_splits = load_population_splits(data_dir)

    # Hyperparameters
    past_sequence_length = 12
    future_offset = 6
    batch_size = 64
    max_interval_minutes = 30

    # Train the model
    train_population_model(population_splits, model_dir, past_sequence_length, future_offset, batch_size, max_interval_minutes)

    # Evaluate the model
    evaluate_population_model(population_splits, model_dir, evaluation_dir, past_sequence_length, future_offset, batch_size, max_interval_minutes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate OhioT1DM models')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to the diatrend dataset directory')
    
    args = parser.parse_args()
    main(args.data_dir)