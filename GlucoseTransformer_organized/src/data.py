import os
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

class TimeSeriesDataset(Dataset):
    """
    Dataset class for glucose time series data.
    
    Prepares sequences of past values and corresponding future values
    for time series forecasting tasks.
    
    Args:
        series: Pandas DataFrame containing glucose time series data
        past_seq_len: Length of input sequence
        future_offset: Prediction horizon (how many steps ahead to predict)
    """
    def __init__(self, series, past_seq_len, future_offset):
        self.series = series
        self.past_seq_len = past_seq_len
        self.future_offset = future_offset
        self.values = series['mg/dl'].values
        
    def __len__(self):
        """Returns the number of available samples."""
        return len(self.values) - self.past_seq_len - self.future_offset + 1
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (input_sequence, target_value)
        """
        seq_x = self.values[idx:idx + self.past_seq_len]
        seq_y = self.values[idx + self.past_seq_len + self.future_offset - 1]
        seq_x = seq_x[:, np.newaxis]  # Add feature dimension
        return torch.FloatTensor(seq_x), torch.FloatTensor([seq_y])


def load_ohio_series_train(path_filename, variate_name, attribute, time_attribue="ts"):
    """
    Load glucose data from Ohio format XML files.
    
    Args:
        path_filename: Path to XML file
        variate_name: Name of the variable in XML
        attribute: Attribute name containing the value
        time_attribue: Attribute name containing timestamp
        
    Returns:
        Pandas DataFrame with timestamp and glucose values
    """
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

            # Rename the columns
            seriesdf.columns = ['timestamp', 'mg/dl']
            return seriesdf


def create_population_splits(folder_path_train_2018, folder_path_train_2020, train_files_2018, train_files_2020,
                           folder_path_test_2018, folder_path_test_2020, test_files_2018, test_files_2020):
    """
    Create data splits for population-level training and testing.
    
    Args:
        folder_path_train_2018: Folder path for 2018 training data
        folder_path_train_2020: Folder path for 2020 training data
        train_files_2018: List of training files from 2018
        train_files_2020: List of training files from 2020
        folder_path_test_2018: Folder path for 2018 test data
        folder_path_test_2020: Folder path for 2020 test data
        test_files_2018: List of test files from 2018
        test_files_2020: List of test files from 2020
        
    Returns:
        Dictionary containing train and test file paths
    """
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
    """
    Create leave-one-out cross validation splits.
    
    Args:
        folder_path_train_2018: Folder path for 2018 training data
        folder_path_train_2020: Folder path for 2020 training data
        train_files_2018: List of training files from 2018
        train_files_2020: List of training files from 2020
        
    Returns:
        Dictionary of LOOCV folds, where each fold has one test file and the rest as training
    """
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


# def create_4fold_splits(folder_path_train_2018, folder_path_train_2020, train_files_2018, train_files_2020):
    """
    Create 4-fold cross-validation splits.
    
    Args:
        folder_path_train_2018: Folder path for 2018 training data
        folder_path_train_2020: Folder path for 2020 training data
        train_files_2018: List of training files from 2018
        train_files_2020: List of training files from 2020
        
    Returns:
        Dictionary of 4 folds with train/test splits
    """
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

def create_5fold_splits(data_path):
    """
    Create 4-fold cross-validation splits.
    
    Args:
        data_path: Folder path
  
    Returns:
        Dictionary of 4 folds with train/test splits
    """
    uids = [int(file.split('.')[0].split('processed_cgm_data_Subject')[1]) for file in os.listdir(data_path)]
    # print(uids)
    splits = [0, 11, 22, 33, 44, float('inf')]
    fold_splits = {}
    for fold in range(5):
        fold_name = f"fold{fold+1}"

        # Create test and train sets
        test_files = [i for i in uids if splits[fold] < i <= splits[fold+1]]
        # print("test", test_files)
        train_files = [i for i in uids if i not in test_files]
        # print("train", train_files)
        # Add to splits dictionary
        fold_splits[fold_name] = {
            'test': ['processed_cgm_data_Subject'+str(i)+'.csv' for i in test_files],
            'train': ['processed_cgm_data_Subject'+str(i)+'.csv' for i in train_files]
        }

        # break

    return fold_splits

def create_5fold_splits_T1DEXI(data_path):
    uids = [int(file.split('.')[0]) for file in os.listdir(data_path)]
    # print(uids)
    splits = [0, 248, 1201, 1348, 1459, float('inf')]
    fold_splits = {}
    for fold in range(5):
        fold_name = f"fold{fold+1}"

        # Create test and train sets
        test_files = [i for i in uids if splits[fold] < i <= splits[fold+1]]
        # print(test_files)
        train_files = [i for i in uids if i not in test_files]

        # Add to splits dictionary
        fold_splits[fold_name] = {
            'test': [str(i)+'.csv' for i in test_files],
            'train': [str(i)+'.csv' for i in train_files]
        }

        # break

    return fold_splits

def convert_to_datetime(date_str):
    """
    Convert timedata into datetime format.
    
    Args:
        date_str: string format time.
  
    Returns:
        datetime format time. 
    """
    try:
        return pd.to_datetime(date_str)
    except ValueError:
        return pd.to_datetime(date_str + ' 00:00:00')

def load_train_data_by_fold(fold_name, fold_splits, data_dir):
  """
    Loading training data by each fold. 
    
    Args:
        fold_name: Current working fold.
        fold_splits: Dictionary of train/test splits for each fold.
        data_dir: Root to saved data files. 
        
    Returns:
        Tuple of (training_dataset, validation_dataset)
    """

  train_df = pd.DataFrame()

  for file in os.listdir(data_dir):
    if file in fold_splits[fold_name]['train']:
      df = pd.read_csv(os.path.join(data_dir, file))
      uid = file.split('.')[0].split('processed_cgm_data_Subject')[1]
      df = df.rename(columns={"date": "timestamp"})
      df['USUBJID'] = [uid] * len(df)
      df['timestamp'] = df['timestamp'].apply(convert_to_datetime)

      df = df.loc[:, ['USUBJID', 'timestamp', 'mg/dl']]
      train_df = pd.concat([train_df, df])

  return train_df

def load_train_data_by_fold_T1DEXI(fold_name, fold_splits, data_dir):
    """
        Loading training data by each fold. 
        
        Args:
            fold_name: Current working fold.
            fold_splits: Dictionary of train/test splits for each fold.
            data_dir: Root to saved data files. 
            
        Returns:
            Tuple of (training_dataset, validation_dataset)
    """
    train_df = pd.DataFrame()

    for file in os.listdir(data_dir):
        if file in fold_splits[fold_name]['train']:
            df = pd.read_csv(os.path.join(data_dir, file))
            # df.drop(columns=['USUBJID'], inplace=True)
            df = df.rename(columns={"LBORRES": "mg/dl", "LBDTC": "timestamp"})
            df['timestamp'] = df['timestamp'].apply(convert_to_datetime)
            df = df.loc[:, ['USUBJID', 'timestamp', 'mg/dl']] # reorder to keep the same format as Diatrend for future training

        train_df = pd.concat([train_df, df])
      # break
    return train_df

def split_into_continuous_series(df, past_sequence_length, future_offset, max_interval_minutes=30):
    """
    Split time series into continuous segments based on time gaps.
    
    Args:
        df: DataFrame containing glucose data
        past_sequence_length: Length of input sequence
        future_offset: Prediction horizon
        max_interval_minutes: Maximum gap (minutes) allowed within a continuous series
        
    Returns:
        List of DataFrames, each representing a continuous series
    """
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


def create_train_val_datasets(series_list, train_ratio=0.8, past_seq_len=7, future_offset=6):
    """
    Create training and validation datasets from series list.
    
    Args:
        series_list: List of continuous time series DataFrames
        train_ratio: Ratio of data to use for training
        past_seq_len: Length of input sequence
        future_offset: Prediction horizon
        
    Returns:
        Tuple of (training_dataset, validation_dataset)
    """
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