import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
from src.models import TransformerEncoder_version2

def save_model(model, test_file_path, save_dir='saved_models'):
    """
    Save a trained model to disk.
    
    Args:
        model: The model to save
        test_file_path: Path to test file (used for naming)
        save_dir: Directory to save the model
    """
    os.makedirs(save_dir, exist_ok=True)
    # Extract the base filename from the test file path
    test_file_name = os.path.splitext(os.path.basename(test_file_path))[0]
    model_path = os.path.join(save_dir, f'model_{test_file_name}.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def load_model(test_file_path, model_class, save_dir='saved_models', **model_kwargs):
    """
    Load a trained model from disk.
    
    Args:
        test_file_path: Path to test file (used for naming)
        model_class: Model class to instantiate
        save_dir: Directory where models are saved
        **model_kwargs: Arguments to pass to model constructor
        
    Returns:
        Loaded model
    """
    # Extract the base filename from the test file path
    test_file_name = os.path.splitext(os.path.basename(test_file_path))[0]
    model_path = os.path.join(save_dir, f'model_{test_file_name}.pth')
    
    # Initialize a new model
    model = model_class(**model_kwargs)
    
    # Load the saved weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    return model


def load_model_population(test_file_path, past_sequence_length, model_class=TransformerEncoder_version2, save_dir='../saved_models'):  
    """
    Load a trained population model from disk.
    
    Args:
        test_file_path: path to test files.
        past_sequence_length: Length of input sequence
        model_class: Transformer model which will be used.
        save_dir: Directory where models are saved
                
    Returns:
        Loaded model
    """
    test_file_name = os.path.splitext(os.path.basename(test_file_path))[0]
    model_path = os.path.join(save_dir, f'model_{test_file_name}.pth')

    model = model_class(
        past_seq_len=past_sequence_length,
        num_layers=1,
        d_model=512,
        nhead=4,
        input_dim=1,
        dropout=0.2
    )

    # Load the saved weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"Model loaded from {model_path}")
    return model


def evaluate_and_save_metrics(model, test_file_path, save_dir="metrics", 
                            past_sequence_length=7, future_offset=6, 
                            batch_size=32, max_interval_minutes=30):
    """
    Evaluate model performance on test data and save metrics to file.
    
    Args:
        model: The trained model
        test_file_path: Path to test file
        save_dir: Directory to save metrics
        past_sequence_length: Length of input sequence
        future_offset: Prediction horizon
        batch_size: Batch size for testing
        max_interval_minutes: Maximum interval between readings to consider continuous
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Importing at function level to avoid circular imports
    from src.data import load_ohio_series_train, split_into_continuous_series, create_train_val_datasets
    from torch.utils.data import DataLoader
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load and preprocess test data
    test_df = load_ohio_series_train(test_file_path, "glucose_level", "value")
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    test_df = test_df.sort_values('timestamp')
    
    # Split into continuous series
    test_series_list = split_into_continuous_series(test_df, past_sequence_length, future_offset, max_interval_minutes)
    
    # Create dataset and dataloader
    test_dataset, _ = create_train_val_datasets(
        test_series_list,
        train_ratio=0.99,
        past_seq_len=past_sequence_length,
        future_offset=future_offset
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluate model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            ground_truths.extend(targets.cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(predictions).flatten()
    ground_truths = np.array(ground_truths).flatten()
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(ground_truths, predictions))
    mae = np.mean(np.abs(predictions - ground_truths))
    mape = np.mean(np.abs((ground_truths - predictions) / ground_truths)) * 100
    
    # Print metrics
    print(f'Test file: {os.path.basename(test_file_path)}')
    print(f'Root Mean Square Error (RMSE): {rmse:.2f}')
    print(f'Mean Absolute Error (MAE): {mae:.2f}')
    print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')
    
    # Save metrics to file
    metrics_filename = f"metrics_{os.path.basename(test_file_path).split('.')[0]}.txt"
    metrics_path = os.path.join(save_dir, metrics_filename)
    
    with open(metrics_path, 'w') as f:
        f.write(f"Test File: {test_file_path}\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"MAE: {mae:.2f}\n")
        f.write(f"MAPE: {mape:.2f}%\n")
    
    # Create plots
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(predictions[:200], label='Predictions', color='r')
    plt.plot(ground_truths[:200], label='Ground Truth', color='b')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.title('Predictions vs Ground Truth')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(ground_truths, predictions, alpha=0.5)
    plt.plot([min(ground_truths), max(ground_truths)], 
             [min(ground_truths), max(ground_truths)], 
             'r--', label='Perfect Prediction')
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.title(f'Scatter Plot (RMSE: {rmse:.2f})')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'predictions': predictions,
        'ground_truths': ground_truths
    }


def evaluate_and_save_metrics_population(model, test_file_paths, save_dir="metrics", 
                                       past_sequence_length=7, future_offset=6, 
                                       batch_size=32, max_interval_minutes=30):
    """
    Evaluate model performance on multiple test files and save metrics.
    
    Args:
        model: The trained model
        test_file_paths: List of paths to test files
        save_dir: Directory to save metrics
        past_sequence_length: Length of input sequence
        future_offset: Prediction horizon
        batch_size: Batch size for testing
        max_interval_minutes: Maximum interval between readings to consider continuous
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Importing at function level to avoid circular imports
    from src.data import load_ohio_series_train, split_into_continuous_series, create_train_val_datasets
    from torch.utils.data import DataLoader
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load and preprocess test data
    test_df = pd.DataFrame([])
    for path in test_file_paths:
        cur_test_df = load_ohio_series_train(path, "glucose_level", "value")
        test_df = pd.concat([test_df, cur_test_df])
    print(f"Combined test data shape: {test_df.shape}")
    
    # Convert timestamps and sort
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    test_df = test_df.sort_values('timestamp')
    
    # Split into continuous series
    test_series_list = split_into_continuous_series(test_df, past_sequence_length, future_offset, max_interval_minutes)
    
    # Create dataset and dataloader
    test_dataset, _ = create_train_val_datasets(
        test_series_list,
        train_ratio=0.99,
        past_seq_len=past_sequence_length,
        future_offset=future_offset
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluate model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            ground_truths.extend(targets.cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(predictions).flatten()
    ground_truths = np.array(ground_truths).flatten()
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(ground_truths, predictions))
    mae = np.mean(np.abs(predictions - ground_truths))
    mape = np.mean(np.abs((ground_truths - predictions) / ground_truths)) * 100
    
    # Print metrics
    print(f'Test files: population evaluation')
    print(f'Root Mean Square Error (RMSE): {rmse:.2f}')
    print(f'Mean Absolute Error (MAE): {mae:.2f}')
    print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')
    
    # Save metrics to file
    metrics_filename = "metrics_population.txt"
    metrics_path = os.path.join(save_dir, metrics_filename)
    
    with open(metrics_path, 'w') as f:
        f.write("Test Files: Population evaluation\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"MAE: {mae:.2f}\n")
        f.write(f"MAPE: {mape:.2f}%\n")
    
    # Create plots
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(predictions[:200], label='Predictions', color='r')
    plt.plot(ground_truths[:200], label='Ground Truth', color='b')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.title('Predictions vs Ground Truth')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(ground_truths, predictions, alpha=0.5)
    plt.plot([min(ground_truths), max(ground_truths)], 
             [min(ground_truths), max(ground_truths)], 
             'r--', label='Perfect Prediction')
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.title(f'Scatter Plot (RMSE: {rmse:.2f})')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'predictions': predictions,
        'ground_truths': ground_truths
    }
    
def evaluate_and_save_metrics_diatrend(model, test_df, save_dir="metrics",
                            past_sequence_length=7, future_offset=6,
                            batch_size=32, max_interval_minutes=30, uid=None):
    """
    Evaluate model performance on test data and save metrics to file.

    Args:
        model: The trained model
        save_dir: Directory to save metrics
        past_sequence_length: Length of input sequence
        future_offset: Prediction horizon
        batch_size: Batch size for testing
        max_interval_minutes: Maximum interval between readings to consider continuous
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    from src.data import (split_into_continuous_series, create_train_val_datasets)
    from torch.utils.data import DataLoader

    # Split into continuous series
    test_series_list = split_into_continuous_series(test_df, past_sequence_length, future_offset,max_interval_minutes)

    # Create dataset and dataloader
    test_dataset, _ = create_train_val_datasets(
        test_series_list,
        train_ratio=0.9999,
        past_seq_len=past_sequence_length,
        future_offset=future_offset
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate model
    model.eval()
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to('cuda') if torch.cuda.is_available() else inputs
            targets = targets.to('cuda') if torch.cuda.is_available() else targets

            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            ground_truths.extend(targets.cpu().numpy())

    # Convert to numpy arrays
    predictions = np.array(predictions).flatten()
    ground_truths = np.array(ground_truths).flatten()

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(ground_truths, predictions))
    mae = np.mean(np.abs(predictions - ground_truths))
    mape = np.mean(np.abs((ground_truths - predictions) / ground_truths)) * 100

    # Print metrics
    print(f'Test file: {uid}')
    print(f'Root Mean Square Error (RMSE): {rmse:.2f}')
    print(f'Mean Absolute Error (MAE): {mae:.2f}')
    print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')

    # Save metrics to file
    metrics_filename = f"metrics_{uid}.txt"
    metrics_path = os.path.join(save_dir, metrics_filename)

    with open(metrics_path, 'w') as f:
        f.write(f"Test File: {uid}\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"MAE: {mae:.2f}\n")
        f.write(f"MAPE: {mape:.2f}%\n")

    # Create plots
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(predictions[:200], label='Predictions', color='r')
    plt.plot(ground_truths[:200], label='Ground Truth', color='b')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.title('Predictions vs Ground Truth')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(ground_truths, predictions, alpha=0.5)
    plt.plot([min(ground_truths), max(ground_truths)],
             [min(ground_truths), max(ground_truths)],
             'r--', label='Perfect Prediction')
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.title(f'Scatter Plot (RMSE: {rmse:.2f})')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'predictions': predictions,
        'ground_truths': ground_truths
    }
    
def evaluate_and_save_metrics_T1DEXI(model, test_df, save_dir="metrics",
                            past_sequence_length=7, future_offset=6,
                            batch_size=32, max_interval_minutes=30, uid=None):
    """
    Evaluate model performance on test data and save metrics to file.

    Args:
        model: The trained model
        save_dir: Directory to save metrics
        past_sequence_length: Length of input sequence
        future_offset: Prediction horizon
        batch_size: Batch size for testing
        max_interval_minutes: Maximum interval between readings to consider continuous
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    from src.data import (split_into_continuous_series, create_train_val_datasets)
    from torch.utils.data import DataLoader

    # Split into continuous series
    test_series_list = split_into_continuous_series(test_df, past_sequence_length, future_offset,max_interval_minutes)

    # Create dataset and dataloader
    test_dataset, _ = create_train_val_datasets(
        test_series_list,
        train_ratio=0.99,
        past_seq_len=past_sequence_length,
        future_offset=future_offset
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate model
    model.eval()
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to('cuda') if torch.cuda.is_available() else inputs
            targets = targets.to('cuda') if torch.cuda.is_available() else targets

            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            ground_truths.extend(targets.cpu().numpy())

    # Convert to numpy arrays
    predictions = np.array(predictions).flatten()
    ground_truths = np.array(ground_truths).flatten()

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(ground_truths, predictions))
    mae = np.mean(np.abs(predictions - ground_truths))
    mape = np.mean(np.abs((ground_truths - predictions) / ground_truths)) * 100

    # Print metrics
    print(f'Test file: {uid}')
    print(f'Root Mean Square Error (RMSE): {rmse:.2f}')
    print(f'Mean Absolute Error (MAE): {mae:.2f}')
    print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')

    # Save metrics to file
    metrics_filename = f"metrics_{uid}.txt"
    metrics_path = os.path.join(save_dir, metrics_filename)

    with open(metrics_path, 'w') as f:
        f.write(f"Test File: {uid}\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"MAE: {mae:.2f}\n")
        f.write(f"MAPE: {mape:.2f}%\n")

    # Create plots
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(predictions[:200], label='Predictions', color='r')
    plt.plot(ground_truths[:200], label='Ground Truth', color='b')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.title('Predictions vs Ground Truth')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(ground_truths, predictions, alpha=0.5)
    plt.plot([min(ground_truths), max(ground_truths)],
             [min(ground_truths), max(ground_truths)],
             'r--', label='Perfect Prediction')
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.title(f'Scatter Plot (RMSE: {rmse:.2f})')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'predictions': predictions,
        'ground_truths': ground_truths
    }