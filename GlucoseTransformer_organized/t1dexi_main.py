import sys
import os
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
    model_dir = '../saved_models/'
    evaluation_dir = '../evaluation/'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(evaluation_dir, exist_ok=True)
    return model_dir, evaluation_dir


def train_fold_models(fold_splits, data_dir, model_dir, past_sequence_length, future_offset, batch_size, max_interval_minutes):
    """Train models for each fold."""
    for fold in fold_splits.keys():
        train_df = load_train_data_by_fold_T1DEXI(fold, fold_splits, data_dir)
        print(fold, '\ntrain data shape:', train_df.shape)

        # Move model to GPU if available
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

        # Create datasets
        train_series_list = []
        for uid in train_df['USUBJID'].unique():
            cur_df = train_df[train_df['USUBJID'] == uid]
            cur_df.drop(columns=['USUBJID'], inplace=True)
            series_list = split_into_continuous_series(cur_df, past_sequence_length, future_offset, max_interval_minutes)
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
        save_dir = os.path.join(model_dir, f'saved_models_T1DEXI/5_fold_sh{past_sequence_length}/')
        os.makedirs(save_dir, exist_ok=True)
        save_model(model, f'sh{past_sequence_length}_{fold}', save_dir)


def evaluate_fold_models(fold_splits, data_dir, model_dir, evaluation_dir, past_sequence_length, future_offset, batch_size, max_interval_minutes):
    """Evaluate models for each fold."""
    test_eval = []
    for fold in fold_splits.keys():
        print(fold, fold_splits[fold]['test'])

        # Load the saved model
        model = load_model_population(
            f'sh{past_sequence_length}_{fold}', past_sequence_length, model_class=TransformerEncoder_version2,
            save_dir=os.path.join(model_dir, f'saved_models_T1DEXI/5_fold_sh{past_sequence_length}/')
        )

        for test in fold_splits[fold]['test']:
            uid = test.split('.')[0]
            test_df = pd.read_csv(os.path.join(data_dir, test))
            test_df = test_df.rename(columns={"LBORRES": "mg/dl", "LBDTC": "timestamp"})
            test_df['timestamp'] = test_df['timestamp'].apply(convert_to_datetime)
            test_df = test_df.loc[:, ['timestamp', 'mg/dl']]

            metrics = evaluate_and_save_metrics_T1DEXI(
                model=model,
                test_df=test_df,
                save_dir=os.path.join(evaluation_dir, f'evaluation_metrics_T1DEXI/5_fold_individual_sh{past_sequence_length}/'),
                past_sequence_length=past_sequence_length,
                future_offset=future_offset,
                batch_size=batch_size,
                max_interval_minutes=max_interval_minutes,
                uid=uid
            )

            test_eval.append([uid, round(metrics['rmse'], 2), round(metrics['mae'], 2), round(metrics['mape'], 2)])

            print(f"RMSE: {metrics['rmse']:.2f}")
            print(f"MAE: {metrics['mae']:.2f}")
            print(f"MAPE: {metrics['mape']:.2f}%")

    # Save evaluation results
    df = pd.DataFrame(test_eval, columns=['test patient', 'RMSE', 'MAE', 'MAPE'])
    df.to_csv(os.path.join(evaluation_dir, f'evaluation_metrics_T1DEXI/5_fold_test_eval_sh{past_sequence_length}.csv'), index=False)


def main():
    """Main function to train and evaluate models for the T1DEXI dataset."""
    # Setup directories
    model_dir, evaluation_dir = setup_directories()

    # Data directory
    data_dir = '../../../ReproGenBG_Dataset/T1DEXI_processed/'

    # Create 5-fold splits
    fold_splits = create_5fold_splits_T1DEXI(data_dir)
    print(fold_splits.keys())

    # Hyperparameters
    past_sequence_length = 24
    future_offset = 6
    batch_size = 64
    max_interval_minutes = 30

    # Train models for each fold
    train_fold_models(fold_splits, data_dir, model_dir, past_sequence_length, future_offset, batch_size, max_interval_minutes)

    # Evaluate models for each fold
    evaluate_fold_models(fold_splits, data_dir, model_dir, evaluation_dir, past_sequence_length, future_offset, batch_size, max_interval_minutes)


if __name__ == "__main__":
    main()