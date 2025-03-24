# %%
from __future__ import division, print_function

import collections
import csv
import datetime
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import pickle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Using device: {device}', flush = True)
# %%

def preprocess_DiaTrend(path, round=False):
    subject = pd.read_csv(path)
    subject['date'] = pd.to_datetime(subject['date'], errors='coerce')  # Convert 'date' column to datetime if not already
    subject.sort_values('date', inplace=True)  # Sort the DataFrame by the 'date' column

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
    ph = 6, 30 minutes ahead
    ph = 12, 60 minutes ahead
    '''
    features_list = []
    raw_glu_list = []
    
    
    # Iterate over each segment
    for segment_name, segment_df in segments.items():
       

        # Fill NaNs that might have been introduced by conversion errors
        segment_df.fillna(0, inplace=True)

        # Maximum index for creating a complete feature set
        # print("len of segment_df is ", len(segment_df))
        max_index = len(segment_df) - (history_len + ph)  # Subtracting only 15+ph to ensure i + 15 + ph is within bounds
        # print("for segment ", segment_name, "max_index is ", max_index, flush = True)
        # Iterate through the data to create feature-label pairs
        for i in range(max_index):
            # Extracting features from index i to i+15
            segment_df = segment_df.reset_index(drop = True)
            features = segment_df.loc[i:i+history_len, ['glucose_value']].values
            raw_glu_list.append(segment_df.loc[i+history_len+ph, 'glucose_value'])
            features_list.append(features)
            # labels_list.append(label)
            
    print("len of features_list " + str(len(features_list)), flush = True)

    return features_list, raw_glu_list


# %%
import torch
import torch.nn as nn
import torch.optim as optim


class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob):
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
    glucose = preprocess_DiaTrend(filename)
    glucose_dict = {entry[0]['ts']: entry[0]['value'] for entry in glucose}

    # Create the multi-channel database
    g_data = []
    for timestamp in glucose_dict:
        record = {
            'timestamp': timestamp,
            'glucose_value': glucose_dict[timestamp],
            # 'meal_type': None,
            # 'meal_carbs': 0
        }
            
        g_data.append(record)

    # Create DataFrame
    glucose_df = pd.DataFrame(g_data)

    # Convert glucose values to numeric type for analysis
    glucose_df['glucose_value'] = pd.to_numeric(glucose_df['glucose_value'])

    # Calculate percentiles
    lower_percentile = np.percentile(glucose_df['glucose_value'], 2)
    upper_percentile = np.percentile(glucose_df['glucose_value'], 98)

    # Print thresholds
    # print(f"2% lower threshold: {lower_percentile}")
    # print(f"98% upper threshold: {upper_percentile}")
    filename = os.path.basename(j)
    file_number = int(filename.split('Subject')[-1].split('.')[0])  # Extract numeric part before '.
    segments = segement_data_as_6_min(glucose_df, file_number)

    return segments

# %%
input_size = 1# Number of input features
hidden_size = 128  # Hidden vector size
num_layers = 2  # Number of LSTM layers
output_size = 1  # Single output
dropout_prob = 0.2  # Dropout probability


model = StackedLSTM(input_size, hidden_size, num_layers, output_size, dropout_prob) # input_size, hidden_size, num_layers, output_size, dropout_prob
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)

# load the model
# model.load_state_dict(torch.load('./fold_1_model.pth'))

# %%
import glob 
import os
import sys

splits = [0, 11, 22, 33, 44, 54]

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
# test_segment_list = []

for j in glob.glob('../diatrend_subset/*.csv'):
    # don't use overlap
    filename = os.path.basename(j)
    file_number = int(filename.split('Subject')[-1].split('.')[0])  # Extract numeric part before '.csv'
    # Exclude files within the range 0 to 248
    if bot_range <= file_number <= top_range:
        # print("Processing test file ", filename, flush=True)
        # segments = get_gdata(j)
        # test_segment_list.append(segments)
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
        
# test_merged_segments = {}
# for segment in test_segment_list:
#     for key, value in segment.items():
#         test_merged_segments[key] = value
        

# %%
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
    print(f'RMSE on the Fold: {rmse}')
    return  rmse

ph = 6

features_list, raw_glu_list = prepare_dataset(merged_segments, ph, history_len)
print(f"running with {history_len} as the history and {ph} as the prediction horizon and {fold} is the fold number", flush = True)
# Assuming features_list and raw_glu_list are already defined
features_array = np.array(features_list)
labels_array = np.array(raw_glu_list)

# Step 1: Split into 80% train+val and 20% test
X_train, X_val, y_train, y_val = train_test_split(features_array, labels_array, test_size=0.2, shuffle=False)


# Step 2: Split the 80% into 70% train and 10% val (0.7/0.8 = 0.875)
# X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, shuffle=False)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)


# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, 
                         pin_memory=True, num_workers=4)
torch.backends.cudnn.benchmark = True

val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
print("Dataset's created", flush = True)
num_epochs =100
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

# %%
# save the model c
print(f"saved in ./diatrend/{fold}_{ph}_Diatrend_model.pth", flush = True)

torch.save(model.state_dict(), f'./diatrend/{fold}_{history_len}_Diatrend_model.pth')

########################
# TEST THE MODEL
########################

def get_test_rmse(model, test_loader):
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


    rmse = np.sqrt(mean_squared_error(actuals,predictions))
    print(f'RMSE on the folds: {rmse}')
    return  rmse


segment_list = [] 
test_segment_list = []
new_test_rmse_list = []

for j in glob.glob('../diatrend_subset/*.csv'):
    # don't use overlap
    filename = os.path.basename(j)
    file_number = int(filename.split('Subject')[-1].split('.')[0])  # Extract numeric part before '.csv'
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

df = pd.DataFrame(new_test_rmse_list, columns = ['rmse', 'filenumber']).to_csv(f'Diatrend_{fold}_{history_len}_rmse.csv', index = False)


