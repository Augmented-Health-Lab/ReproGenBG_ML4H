import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import mean_squared_error

# Modified TimeSeriesDataset to work with individual series
class TimeSeriesDataset(Dataset):
    def __init__(self, series, past_seq_len, future_offset):
        self.series = series
        self.past_seq_len = past_seq_len
        self.future_offset = future_offset
        self.values = series['mg/dl'].values
        
    def __len__(self):
        return len(self.values) - self.past_seq_len - self.future_offset + 1
    
    def __getitem__(self, idx):
        seq_x = self.values[idx:idx + self.past_seq_len]
        seq_y = self.values[idx + self.past_seq_len + self.future_offset - 1]
        seq_x = seq_x[:, np.newaxis]  # Add feature dimension
        return torch.FloatTensor(seq_x), torch.FloatTensor([seq_y])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # [1, max_len, d_model]
        
    def forward(self, x):
        # x needs to be shape [batch_size, seq_len, d_model]
        # print(x.shape)
        # print(self.encoding[:, :x.size(1), :].shape)
        return x + self.encoding[:, :x.size(1), :].to(x.device)
    
# class TransformerEncoderLayer(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
#         super(TransformerEncoderLayer, self).__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
        
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
        
#         self.activation = nn.ReLU()
    
#     def forward(self, src):
#         src2 = self.self_attn(src, src, src)[0]
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#         return src

class TransformerEncoder_version2(nn.Module):
    def __init__(self, past_seq_len, num_layers, d_model, nhead, input_dim=1, dropout=0.1):
        super(TransformerEncoder_version2, self).__init__()
        self.d_model = d_model
        self.past_seq_len = past_seq_len
        
        # Positional Encoding (green block in the figure)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Initial projection of input data
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Stack of Encoder Layers
        self.encoder_layers = nn.ModuleList([])
        for _ in range(num_layers):
            # Each encoder layer contains:
            layer = nn.ModuleDict({
                # 1. Multi-Head Attention block (yellow in figure)
                'attention': nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=nhead,
                    dropout=dropout,
                    batch_first=True
                ),
                # 2. Add & Norm after attention (yellow in figure)
                'norm1': nn.LayerNorm(d_model),
                
                # 3. Feed Forward block (blue in figure)
                'feed_forward': nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.ReLU(),
                    # nn.Dropout(dropout),
                    # nn.Linear(d_model, d_model)
                ),
                # 4. Add & Norm after feed forward (yellow in figure)
                'norm2': nn.LayerNorm(d_model)
            })
            self.encoder_layers.append(layer)

        # # Fully Connected output layer (orange in figure)
        # self.output_layer = nn.Sequential(
        #     nn.Linear(d_model, past_seq_len),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(past_seq_len, 1)
        # )
        
        # 5. two linear layers with dimension reduction, matching the layer names in figure 6
        self.linear2 = nn.Linear(d_model, 1)
        self.linear3 = nn.Linear(past_seq_len, 1)
        
    def forward(self, src):
        # src shape: [batch_size, seq_len, input_dim]
        
        # Initial projection and positional encoding
        x = self.input_projection(src)
        x = self.pos_encoder(x)
        
        # Process through encoder layers
        for layer in self.encoder_layers:
            # Multi-Head Attention
            attn_output, _ = layer['attention'](x, x, x)
            # Add & Norm (first residual connection)
            x = layer['norm1'](x + attn_output)
            # Feed Forward
            ff_output = layer['feed_forward'](x)
            # Add & Norm (second residual connection)
            x = layer['norm2'](x + ff_output)
        
        # print('x2: ', x.shape)
        # Global average pooling over sequence length
        # x = x.mean(dim=2)

        # print('x3: ', x.shape)
        x = self.linear2(x)
        x = x.squeeze(-1)
        output = self.linear3(x)
        # print('output2: ', output.shape)
        # print('output: ', output)   
        return output




class TransformerEncoder(nn.Module):
    # !!! This class is the final model !!!
    def __init__(self, num_layers, d_model, nhead, input_dim=1, dim_feedforward=256, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        
        # Positional Encoding (green block in the figure)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Initial projection of input data
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Stack of Encoder Layers
        self.encoder_layers = nn.ModuleList([])
        for _ in range(num_layers):
            # Each encoder layer contains:
            layer = nn.ModuleDict({
                # 1. Multi-Head Attention block (yellow in figure)
                'attention': nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=nhead,
                    dropout=dropout,
                    batch_first=True
                ),
                # 2. Add & Norm after attention (yellow in figure)
                'norm1': nn.LayerNorm(d_model),
                
                # 3. Feed Forward block (blue in figure)
                'feed_forward': nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, d_model)
                ),
                # 4. Add & Norm after feed forward (yellow in figure)
                'norm2': nn.LayerNorm(d_model)
            })
            self.encoder_layers.append(layer)
        
        # Fully Connected output layer (orange in figure)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), # 12
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1)
        )
        
    def forward(self, src):
        # src shape: [batch_size, seq_len, input_dim]
        
        # Initial projection and positional encoding
        x = self.input_projection(src)
        x = self.pos_encoder(x)
        
        # Process through encoder layers
        for layer in self.encoder_layers:
            # Multi-Head Attention
            attn_output, _ = layer['attention'](x, x, x)
            # Add & Norm (first residual connection)
            x = layer['norm1'](x + attn_output)
            
            # Feed Forward
            ff_output = layer['feed_forward'](x)
            # Add & Norm (second residual connection)
            x = layer['norm2'](x + ff_output)
        
        # Global average pooling over sequence length
        x = x.mean(dim=1)
        
        # Final output projection
        output = self.output_layer(x)
        
        return output

# Load the Ohio series train data
def load_ohio_series_train(path_filename, variate_name, attribute, time_attribue="ts"):
    tree = ET.parse(f"{path_filename}")
    root = tree.getroot()
    for child in root:
        if child.tag == variate_name:
            dates = []
            values = []
            for event in child:
                ts = event.attrib[time_attribue]
                date = pd.to_datetime(ts, format='%d-%m-%Y %H:%M:%S')
                date = date.replace(second=0)
                value = float(event.attrib[attribute])
                dates.append(date)
                values.append(value)
            index = pd.DatetimeIndex(dates)
            series = pd.Series(values, index=index)
            seriesdf = series.reset_index()

            # Rename the columns (optional)
            seriesdf.columns = ['timestamp', 'mg/dl']
            return seriesdf

# population data splits on OhioT1DM-1 dataset
def create_population_splits(folder_path_train_2018, folder_path_train_2020, train_files_2018, train_files_2020,
                             folder_path_test_2018, folder_path_test_2020, test_files_2018, test_files_2020):
    # Create full paths for all files
    train_2018 = [os.path.join(folder_path_train_2018, f) for f in train_files_2018]
    train_2020 = [os.path.join(folder_path_train_2020, f) for f in train_files_2020]
    
    test_2018 = [os.path.join(folder_path_test_2018, f) for f in test_files_2018]
    test_2020 = [os.path.join(folder_path_test_2020, f) for f in test_files_2020]
    
    # Combine all paths
    all_train = train_2018 + train_2020
    all_test = test_2018 + test_2020
    
    population_splits = {
        'train': all_train,
        'test': all_test
    }
        
    print(f"Test file: {population_splits['test']}")
    print(f"Number of training files: {len(population_splits['train'])}")
    print("Training files:")
    for train_path in population_splits['train'][:3]:  # Show first 3 training files
        print(f"  {os.path.basename(train_path)}")
    
    return population_splits
    

def create_loocv_splits(folder_path_train_2018, folder_path_train_2020, train_files_2018, train_files_2020):
    # Create full paths for all files
    paths_2018 = [os.path.join(folder_path_train_2018, f) for f in train_files_2018]
    paths_2020 = [os.path.join(folder_path_train_2020, f) for f in train_files_2020]
    
    # Combine all paths
    all_paths = paths_2018 + paths_2020
    
    # Create splits
    loocv_splits = {}
    
    for i, test_path in enumerate(all_paths):
        # Create a split where current file is test and others are train
        fold_name = f"fold{i+1}"
        loocv_splits[fold_name] = {
            'test': test_path,
            'train': [path for path in all_paths if path != test_path]
        }
    
    # Print summary
    print(f"Created {len(loocv_splits)} LOOCV splits")
    for fold_name, split in loocv_splits.items():
        print(f"\n{fold_name}:")
        print(f"Test file: {os.path.basename(split['test'])}")
        print(f"Number of training files: {len(split['train'])}")
        print("Training files:")
        for train_path in split['train'][:3]:  # Show first 3 training files
            print(f"  {os.path.basename(train_path)}")
        if len(split['train']) > 3:
            print("  ...")
    
    return loocv_splits

def create_4fold_splits(folder_path_train_2018, folder_path_train_2020, train_files_2018, train_files_2020):
    # Create full paths for all files
    paths_2018 = [os.path.join(folder_path_train_2018, f) for f in train_files_2018]
    paths_2020 = [os.path.join(folder_path_train_2020, f) for f in train_files_2020]
    
    # Combine all paths
    all_paths = paths_2018 + paths_2020
    total_files = len(all_paths)
    
    # Calculate files per fold (rounded up for the first folds)
    files_per_fold = total_files // 4
    remainder = total_files % 4
    
    # Create splits
    fold_splits = {}
    start_idx = 0
    
    for fold in range(4):
        fold_name = f"fold{fold+1}"
        
        # Calculate number of files for this fold's test set
        if fold < remainder:
            current_fold_size = files_per_fold + 1
        else:
            current_fold_size = files_per_fold
            
        # Get test indices for this fold
        end_idx = start_idx + current_fold_size
        test_indices = list(range(start_idx, end_idx))
        
        # Create test and train sets
        test_files = [all_paths[i] for i in test_indices]
        train_files = [path for i, path in enumerate(all_paths) if i not in test_indices]
        
        # Add to splits dictionary
        fold_splits[fold_name] = {
            'test': test_files,
            'train': train_files
        }
        
        # Update start index for next fold
        start_idx = end_idx
    
    return fold_splits

# First, let's separate the series based on large intervals
def split_into_continuous_series(df, past_sequence_length, future_offset, max_interval_minutes=30):
    # Ensure DataFrame is sorted by timestamp
    df = df.sort_values('timestamp')
    
    # Calculate time differences
    time_diff = df['timestamp'].diff()
    
    # Find break points where interval > max_interval_minutes
    break_points = time_diff > pd.Timedelta(minutes=max_interval_minutes)
    
    # Create a series ID for each continuous sequence
    series_ids = break_points.cumsum()
    
    # Split the dataframe into list of series
    series_list = []
    for series_id in range(series_ids.max() + 1):
        series = df[series_ids == series_id].copy()
        # Only keep series with enough data points
        if len(series) > past_sequence_length + future_offset:
            series_list.append(series)
    
    return series_list



# Split data and create datasets
def create_train_val_datasets(series_list, train_ratio=0.8, past_seq_len=7, future_offset=6):
    train_datasets = []
    val_datasets = []
    
    for series in series_list:
        n_samples = len(series)
        n_train = int(n_samples * train_ratio)
        
        if n_train > past_seq_len + future_offset:  # Ensure enough samples for training
            # Split into train and validation
            train_series = series.iloc[:n_train]
            val_series = series.iloc[n_train - past_seq_len - future_offset:]
            
            # Create datasets
            if len(train_series) > past_seq_len + future_offset:
                train_datasets.append(TimeSeriesDataset(train_series, past_seq_len, future_offset))
            if len(val_series) > past_seq_len + future_offset:
                val_datasets.append(TimeSeriesDataset(val_series, past_seq_len, future_offset))
    
    # Combine datasets
    combined_train_dataset = ConcatDataset(train_datasets)
    combined_val_dataset = ConcatDataset(val_datasets)
    
    return combined_train_dataset, combined_val_dataset

def train_model(model, train_loader, val_loader, num_epochs=200, learning_rate=1e-3):
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    best_val_loss = float('inf')
    patience = 25  # Increased patience
    patience_counter = 0
    best_model = None
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
            targets = targets.to('cuda' if torch.cuda.is_available() else 'cpu')
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        val_loss = evaluate_model(model, val_loader, criterion)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model = model.state_dict().copy()
        else:
            patience_counter += 1
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if epoch % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}")
        
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
    
    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)
    
    return train_losses, val_losses

def evaluate_model(model, data_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to('cuda') if torch.cuda.is_available() else inputs
            targets = targets.to('cuda') if torch.cuda.is_available() else targets
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    val_loss /= len(data_loader)
    return val_loss

def evaluate_and_save_metrics_population(model, test_file_path, save_dir="metrics", 
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
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load and preprocess test data
    test_df = pd.DataFrame([])
    for path in test_file_path:
        cur_test_df = load_ohio_series_train(path, "glucose_level", "value")
        test_df = pd.concat([test_df, cur_test_df])
    print(test_df.shape)
    
    # test_df = load_ohio_series_train(test_file_path, "glucose_level", "value")
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    test_df = test_df.sort_values('timestamp')
    
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
    print(f'Test file: population')
    print(f'Root Mean Square Error (RMSE): {rmse:.2f}')
    print(f'Mean Absolute Error (MAE): {mae:.2f}')
    print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')
    
    # Save metrics to file
    metrics_filename = f"metrics_population.txt"
    metrics_path = os.path.join(save_dir, metrics_filename)
    
    with open(metrics_path, 'w') as f:
        f.write(f"Test File: population'\n")
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
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load and preprocess test data
    test_df = load_ohio_series_train(test_file_path, "glucose_level", "value")
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    test_df = test_df.sort_values('timestamp')
    
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

def save_model(model, test_file_path, save_dir='saved_models'):
    os.makedirs(save_dir, exist_ok=True)
    # Extract the base filename from the test file path
    test_file_name = os.path.splitext(os.path.basename(test_file_path))[0]
    model_path = os.path.join(save_dir, f'model_{test_file_name}.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def load_model(test_file_path, model_class=TransformerEncoder, save_dir='saved_models'):
    # Extract the base filename from the test file path
    test_file_name = os.path.splitext(os.path.basename(test_file_path))[0]
    model_path = os.path.join(save_dir, f'model_{test_file_name}.pth')
    
    # Initialize a new model with the same architecture
    model = model_class(
        num_layers=1,
        d_model=512,
        nhead=4,
        input_dim=1,
        dim_feedforward=256,
        dropout=0.2
    )
    
    # Load the saved weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    return model

def load_model_population(test_file_path, past_sequence_length, model_class=TransformerEncoder_version2, save_dir='saved_models'):
    # Extract the base filename from the test file path
    test_file_name = os.path.splitext(os.path.basename(test_file_path))[0]
    model_path = os.path.join(save_dir, f'model_{test_file_name}.pth')

    model = TransformerEncoder_version2(
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