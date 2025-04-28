from __future__ import division, print_function

import collections
import csv
import datetime
import torch
import glob
import os
import sys
import pickle

import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device, flush = True)

# Set backends to true for faster training
torch.backends.cudnn.benchmark = True

##############################################################################
#                     
#
#                                 FUNCTIONS 
#
#
##############################################################################

def preprocess_t1dexi_cgm(path, round):
    """
    Preprocess the T1DEXI data from a CSV file.
    Args:
        path (str): Path to the CSV file containing DiaTrend data.
    
    Returns:
        list: A list of dictionaries containing the processed data with timestamps and glucose values.
    """
    
    # Reads path to the T1DEXI file and processes the data
    subject = pd.read_csv(path)
    selected_cgm = subject[['LBDTC', 'LBORRES']]
    new_df_cgm = pd.DataFrame(selected_cgm)

    new_df_cgm['LBDTC'] = pd.to_datetime(new_df_cgm['LBDTC'], errors='coerce')  
    new_df_cgm.sort_values('LBDTC', inplace=True)  

    if round == True:
        rounded_timestamp = []
        for ts in new_df_cgm["LBDTC"]:
            rounded_timestamp.append(round_up_to_nearest_five_minutes(ts))
        new_df_cgm["rounded_LBDTC"] = rounded_timestamp
        formatted_data = [[{'ts': row['rounded_LBDTC'], 'value': row['LBORRES']}] for _, row in new_df_cgm.iterrows()]

    else:
        # Convert each row to the desired format
        formatted_data = [[{'ts': row['LBDTC'].to_pydatetime(), 'value': row['LBORRES']}] for _, row in new_df_cgm.iterrows()]
    
    return formatted_data

def segement_data_as_6_min(data, user_id):
    """
    Segments the data into smaller chunks based on time gaps.
    Args:
        data (pd.DataFrame): DataFrame containing the glucose data with timestamps.
        user_id (int): User ID for naming segments.
    
    Returns:
        dict: A dictionary where keys are segment names and values are DataFrames for each segment.
    """
    
    df = pd.DataFrame(data)

    # Calculate time differences
    df['time_diff'] = df['timestamp'].diff()

    # Identify large gaps
    df['new_segment'] = df['time_diff'] > pd.Timedelta(hours=0.1)

    # Find indices where new segments start
    segment_starts = df[df['new_segment']].index

    # Initialize an empty dictionary to store segments
    segments = {}
    prev_index = 0

    # Loop through each segment start and slice the DataFrame accordingly
    for i, start in enumerate(segment_starts, 1):
        segments[f'segment_{user_id}_{i}'] = df.iloc[prev_index:start].reset_index(drop=True)
        prev_index = start

    # Add the last segment from the last gap to the end of the DataFrame
    segments[f'segment_{user_id}_{len(segment_starts) + 1}'] = df.iloc[prev_index:].reset_index(drop=True)

    # Optionally remove helper columns from each segment
    for segment in segments.values():
        segment.drop(columns=['time_diff', 'new_segment'], inplace=True)
    
    return segments


def prepare_dataset(segments, ph, history_len):
    '''
    Function to prepare the dataset for training the LSTM model.
    
    Args:
        segments (dict): Dictionary containing segmented DataFrames.
        ph (int): Prediction horizon (in minutes).
        history_len (int): Length of the history to consider for each prediction (in minutes).
            
    Returns:
        features_list (list): List of feature arrays for each segment.
        raw_glu_list (list): List of raw glucose values for each segment.
    
    ph = 6, means 30 minutes ahead
    ph = 12, means 60 minutes ahead
    '''
    
    features_list = []
    labels_list = []
    raw_glu_list = []
    
    
    # Iterate over each segment
    for segment_name, segment_df in segments.items():
       

        # Fill NaNs that might have been introduced by conversion errors
        segment_df.fillna(0, inplace=True)

        # Ensures that the loop does not go out of bounds
        max_index = len(segment_df) - (history_len + ph)  
        
        # Iterate through the data to create feature-label pairs
        for i in range(max_index):
            segment_df = segment_df.reset_index(drop = True)
            features = segment_df.loc[i:i+history_len, ['glucose_value']].values
            raw_glu_list.append(segment_df.loc[i+history_len+ph, 'glucose_value'])
            features_list.append(features)
            
    print("len of features_list " + str(len(features_list)), flush = True)

    return features_list, raw_glu_list



class StackedLSTM(nn.Module):
    """
    Class for a stacked LSTM model for time series prediction. This model consists of two LSTM layers followed by fully connected layers. The first LSTM layer processes the input sequence, and the second LSTM layer processes the output of the first layer. The final output is produced by passing the last time step output through a series of fully connected layers. 
    """
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob):
        """
        Initializes the StackedLSTM model.
        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of features in the hidden state of the LSTM.
            num_layers (int): Number of stacked LSTM layers.
            output_size (int): Number of output features.
            dropout_prob (float): Dropout probability for regularization.
        """
        super(StackedLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True).to(device)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob).to(device)
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True).to(device)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 512).to(device)
        self.fc2 = nn.Linear(512, 128).to(device)
        self.fc3 = nn.Linear(128, output_size).to(device)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    
    def forward(self, x):
        """
        Forward pass through the StackedLSTM model.

        Args:
            x (_type_): The input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            out: The output tensor of shape (batch_size, output_size) after passing through the LSTM layers and fully connected layers.
        """
        batch_size = x.size(0)  # Get the batch size from the input tensor

        # Initialize hidden and cell state for the first LSTM layer
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        
        # First LSTM layer
        out, (hn, cn) = self.lstm1(x, (h0, c0))
        
        # Dropout layer
        out = self.dropout(out)
        
        # Initialize hidden and cell state for the second LSTM layer
        h1 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        c1 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        
        # Second LSTM layer
        out, (hn, cn) = self.lstm2(out, (h1, c1))
        
        # Fully connected layers
        out = out[:, -1, :]  # Get the last time step output
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out

def get_gdata(filename):
    """
    Function to process the DiaTrend data file and segment it into smaller chunks.

    Args:
        filename (_type_): _description_

    Returns:
        _type_: _description_
    """
    glucose = preprocess_t1dexi_cgm(filename, False)
    glucose_dict = {entry[0]['ts']: entry[0]['value'] for entry in glucose}

    # Create the multi-channel database
    g_data = []
    for timestamp in glucose_dict:
        record = {
            'timestamp': timestamp,
            'glucose_value': glucose_dict[timestamp],
        }
            
        g_data.append(record)

    # Create DataFrame
    glucose_df = pd.DataFrame(g_data)

    # Convert glucose values to numeric type for analysis
    glucose_df['glucose_value'] = pd.to_numeric(glucose_df['glucose_value'])

    # Calculate percentiles
    lower_percentile = np.percentile(glucose_df['glucose_value'], 2)
    upper_percentile = np.percentile(glucose_df['glucose_value'], 98)

    # This is to ensure overlapping of filename keys does not happen
    filename = os.path.basename(j)
    file_number = int(filename.split('.')[0])  # Extract numeric part before '.
    segments = segement_data_as_6_min(glucose_df, file_number)

    return segments

def get_test_rmse(model, test_loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions.append(outputs)
            actuals.append(targets)

    predictions = torch.cat(predictions).cpu().numpy()
    actuals = torch.cat(actuals).cpu().numpy()


    rmse = np.sqrt(mean_squared_error(actuals,predictions))
    print(f'RMSE on the folds: {rmse}')
    return  rmse

##############################################################################
#                     
#
#                                 TRAINING 
#
#
##############################################################################

def main(): 
    input_size = 1
    hidden_size = 128  
    num_layers = 2  
    output_size = 1  
    dropout_prob = 0.2  
    ph = 6
    num_epochs =100
    batch_size = 128

    splits = [0, 248, 1201, 1348, 1459, 1726]

    # if the name of the file is between 0 and 248, don't include it in the training set
    if len(sys.argv) > 2: 
        fold = sys.argv[1]
        bot_range = splits[int(fold) -1]
        top_range = splits[int(fold)]
        history_len = int(sys.argv[2])
    else: 
        fold = 1
        bot_range = 0
        top_range = 248

    segment_list = [] 
    test_segment_list = []

    for j in glob.glob('../../../../data/T1DEXI/T1DEXI_processed/*.csv'):
        # don't use overlap
        filename = os.path.basename(j)
        file_number = int(filename.split('.')[0])  # Extract numeric part before '.csv'
        # Exclude files within the range 0 to 248
        if bot_range <= file_number <= top_range:
            continue
        else: 
            print("Processing train file ", filename, flush=True)
            segments = get_gdata(j)
            segment_list.append(segments)

    # merge the list so that it's one list of dictionaries
    merged_segments = {}
    for segment in segment_list:
        for key, value in segment.items():
            merged_segments[key] = value


    features_list, raw_glu_list = prepare_dataset(merged_segments, ph, history_len)
    print(len(features_list), flush = True)
    # Assuming features_list and raw_glu_list are already defined
    features_array = np.array(features_list)
    labels_array = np.array(raw_glu_list)

    # Step 1: Split into 80% train+val and 20% test
    X_train, X_val, y_train, y_val = train_test_split(features_array, labels_array, test_size=0.2, shuffle=False)


    # Convert the splits to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle=False)

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    model = StackedLSTM(input_size, hidden_size, num_layers, output_size, dropout_prob) # input_size, hidden_size, num_layers, output_size, dropout_prob
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005)

    for epoch in range(num_epochs):
        model.train()
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # make inputs and targets same size
            targets = targets.view(-1, 1)
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}', flush=True)


        model.eval()
        with torch.no_grad():
            total_loss = 0
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                outputs = outputs.squeeze(1)

                loss = criterion(outputs, targets.float())
                total_loss += loss.item()
            
            avg_loss = total_loss / len(val_loader)
            print(f'Test Loss: {avg_loss:.4f}')

    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions.append(outputs)
            actuals.append(targets)

    predictions = torch.cat(predictions).cpu().numpy()
    actuals = torch.cat(actuals).cpu().numpy()


    rmse = np.sqrt(mean_squared_error(actuals,predictions))
    print(f'RMSE on validation set: {rmse}')

    # save the model c
    torch.save(model.state_dict(), f'../outputs/T1DEXI/models/Stacked_T1DEXI_{fold}_{history_len}_model.pth')

    ########################
    # TEST THE MODEL
    ########################



    segment_list = [] 
    test_segment_list = []
    new_test_rmse_list = []

    for j in glob.glob('../../../../data/T1DEXI/*.csv'):
        # don't use overlap
        filename = os.path.basename(j)
        file_number = int(filename.split('.')[0])  # Extract numeric part before '.csv'
        # Exclude files within the range 0 to 248
        if bot_range <= file_number <= top_range:
            print("Processing test file ", filename, flush=True)
            test_segments = get_gdata(j)
            test_features, test_glu = prepare_dataset(test_segments, ph, history_len)
            test_features_array = np.array(test_features)
            test_labels_array = np.array(test_glu)

            X_test = test_features_array
            y_test = test_labels_array

            # Assuming features_list and raw_glu_list are already defined
            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.float32)

            test_dataset = TensorDataset(X_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            new_test_rmse_list.append([filename.split('.')[0], get_test_rmse(model, test_loader)])

    df = pd.DataFrame(new_test_rmse_list, columns = ['rmse', 'filenumber']).to_csv(f'Stacked_T1DEXI_{fold}_{history_len}_rmse.csv', index = False)

if __name__ == "__main__":
    main()