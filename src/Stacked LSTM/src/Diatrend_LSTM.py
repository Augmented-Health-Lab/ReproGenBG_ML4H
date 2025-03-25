from __future__ import division, print_function

import collections
import csv
import datetime
import torch
import os
import pickle
import glob 
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

from datetime import datetime, timedelta
from scipy.interpolate import CubicSpline
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Check if the GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}', flush = True)

# Set backends to true for faster training
torch.backends.cudnn.benchmark = True


##############################################################################
#                     
#
#                                 FUNCTIONS 
#
#
##############################################################################



def preprocess_DiaTrend(path, round=False):
    """
    Preprocess the DiaTrend data from a CSV file.
    Args:
        path (str): Path to the CSV file containing DiaTrend data.
    
    Returns:
        list: A list of dictionaries containing the processed data with timestamps and glucose values.
    """
    
    # Reads path to the diatrend file and processes the date
    subject = pd.read_csv(path)
    subject['date'] = pd.to_datetime(subject['date'], errors='coerce')  
    subject.sort_values('date', inplace=True)  

    if round:
        rounded_timestamp = []
        for ts in subject["date"]:
            rounded_timestamp.append(ts)
        subject["rounded_date"] = rounded_timestamp
        formatted_data = [[{'ts': row['rounded_date'], 'value': row['mg/dl']}] for _, row in subject.iterrows()]
    else:
        # Convert each row to the desired format
        formatted_data = [[{'ts': row['date'].to_pydatetime(), 'value': row['mg/dl']}] for _, row in subject.iterrows()]

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
    df['time_diff'] = df['timestamp'].diff()

    # Identifies gaps over 6 minutes
    df['new_segment'] = df['time_diff'] > pd.Timedelta(hours=0.1)
    segment_starts = df[df['new_segment']].index

    # Initializes an empty dictionary to store segments
    segments = {}
    prev_index = 0

    # Loops through each segment start and slice the DataFrame accordingly
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
    raw_glu_list = []
    
    # Iterate over each segment
    for segment_name, segment_df in segments.items():

        segment_df.fillna(0, inplace=True)

        # Ensures that the loop does not go out of bounds
        max_index = len(segment_df) - (history_len + ph)  
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
    glucose = preprocess_DiaTrend(filename)
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
    glucose_df['glucose_value'] = pd.to_numeric(glucose_df['glucose_value'])
    lower_percentile = np.percentile(glucose_df['glucose_value'], 2)
    upper_percentile = np.percentile(glucose_df['glucose_value'], 98)

    # Print thresholds
    filename = os.path.basename(j)
    file_number = int(filename.split('Subject')[-1].split('.')[0])  
    segments = segement_data_as_6_min(glucose_df, file_number)
    
    return segments


##############################################################################
#                     
#
#                                 TRAINING 
#
#
##############################################################################


# HYPERPARAMETERS
input_size = 1
hidden_size = 128
num_layers = 2  
output_size = 1 
dropout_prob = 0.2  
ph = 6
num_epochs =100
batch_size = 128


# input_size, hidden_size, num_layers, output_size, dropout_prob
model = StackedLSTM(input_size, hidden_size, num_layers, output_size, dropout_prob) 
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)

# This is to split the data into 5 folds for cross-validation
splits = [0, 11, 22, 33, 44, 54]

# Ensures that a fold is provided as a command line argument, if not, it defaults to fold 1
if len(sys.argv) > 2: 
    fold = sys.argv[1]
    bot_range = splits[int(fold) -1]
    top_range = splits[int(fold)]
    history_len = int(sys.argv[2])
else: 
    print ("DEFAULTING TO FOLD 1")
    fold = 1
    bot_range = splits[0]
    top_range = splits[1]
    history_len = 6

segment_list = []

# For each diatrend subject, process the data and segment it
for j in glob.glob('../../../../data/Diatrend/diatrend_subset/*.csv'):
    filename = os.path.basename(j)
    file_number = int(filename.split('Subject')[-1].split('.')[0])  
    
    # Exclude files within the range 0 to 248
    if bot_range <= file_number <= top_range:
        pass
    else: 
        print("Processing train file ", filename, flush=True)
        segments = get_gdata(j)
        segment_list.append(segments)

# merge the list so that it's one list of dictionaries
merged_segments = {}
for segment in segment_list:
    for key, value in segment.items():
        merged_segments[key] = value
        
# prepare the dataset. features_list contains the input features and raw_glu_list will contain the target glucose values
features_list, raw_glu_list = prepare_dataset(merged_segments, ph, history_len)
print(f"running with {history_len} as the history and {ph} as the prediction horizon and {fold} is the fold number", flush = True)

features_array = np.array(features_list)
labels_array = np.array(raw_glu_list)

# Step 1: Split into 80% train+val and 20% test
X_train, X_val, y_train, y_val = train_test_split(features_array, labels_array,
                                                  test_size=0.2, shuffle=False)

# Step 2: Split the 80% into 70% train and 10% val (0.7/0.8 = 0.875)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)


# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size, shuffle=False, 
                          pin_memory=True, num_workers=4)


val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, 
                        batch_size=128, shuffle=False)

print("Dataset's created", flush = True)

# Training the model 
for epoch in range(num_epochs):
    model.train()
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        outputs = outputs.squeeze()
        loss = criterion(outputs, targets)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}', flush = True)


    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze()
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
print(f'RMSE on validation set: {rmse}', flush = True)

# Save the model
print(f"saved in outputs/Diatrend/models/{fold}_{ph}_Diatrend_model.pth", flush = True)

torch.save(model.state_dict(), f'../outputs/Diatrend/models/{fold}_{history_len}_Diatrend_model.pth')

##############################################################################
#
#                                TESTING
#
##############################################################################

def get_test_rmse(model, test_loader):
    """
    Function to calculate the RMSE on the test set using the trained model.

    Args:
        model (_StackedLSTM): The trained StackedLSTM model.
        test_loader (DataLoader): DataLoader for the test dataset.

    Returns:
        rmse (float): The root mean square error of the model predictions on the test set.
    """
    
    # Test the model
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            predictions.append(outputs)
            actuals.append(targets)

    predictions = torch.cat(predictions).cpu().numpy()
    actuals = torch.cat(actuals).cpu().numpy()

    # Get the RMSE
    rmse = np.sqrt(mean_squared_error(actuals,predictions))
    print(f'RMSE on the folds: {rmse}')
    return rmse


segment_list = [] 
test_segment_list = []
new_test_rmse_list = []

# Test the model on the test set. Same as above but now we are using the test set
for j in glob.glob('../../../../data/Diatrend/diatrend_subset/*.csv'):
    filename = os.path.basename(j)
    file_number = int(filename.split('Subject')[-1].split('.')[0]) 
    
    # Exclude files within the range 0 to 248
    if bot_range <= file_number <= top_range:
        print("Processing test file ", filename, flush=True)
        
        # Get the segments for the test set
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

# Convert the list of RMSE values to a DataFrame and save it as a CSV file
df = pd.DataFrame(new_test_rmse_list, columns = ['rmse', 'filenumber']).to_csv(f'../outputs/Diatrend/outputs/Diatrend_{fold}_{history_len}_rmse.csv', index = False)


