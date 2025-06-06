from __future__ import division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import collections
import csv
import datetime
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import sys 
import glob
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device, flush = True)
ph = 6
fold = int(sys.argv[1])
history_len = int(sys.argv[2])

class WaveNetBlock(nn.Module):
    def __init__(self, in_channels, dilation):
        super(WaveNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=2, dilation=dilation, padding=1+dilation - 2^(dilation-1))
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=2, dilation=dilation, padding=dilation)
        self.res_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        # print("shape of x: ", x.shape)
        out = F.relu(self.conv1(x))
        # print("shape of first out: ", out.shape)
        out = F.relu(self.conv2(out))
        # print("shape of second out: ", out.shape)
        res = self.res_conv(x)
        # print("shape of res: ", res.shape)
        return out + res

class WaveNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, dilations):
        super(WaveNet, self).__init__()
        self.initial_conv = nn.Conv1d(in_channels, 32, kernel_size=2, padding=1)
        self.blocks = nn.ModuleList([WaveNetBlock(32, dilation) for dilation in dilations])
        self.final_conv1 = nn.Conv1d(32, 128, kernel_size=2, padding=0)
        self.final_conv2 = nn.Conv1d(128, 256, kernel_size=2, padding=0)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, out_channels)
        
    def forward(self, x):
        x = F.relu(self.initial_conv(x))
        for block in self.blocks:
            # print("enter the block loop")
            x = block(x)
        x = F.relu(self.final_conv1(x))
        x = F.relu(self.final_conv2(x))
        x = x[:, :, -1]  # Get the last time step
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

input_channels = 1  # Number of features
output_channels = 1  # Predicting a single value (glucose level)
num_blocks = 4  # Number of WaveNet blocks
dilations = [2**i for i in range(num_blocks)]  # Dilation rates: 1, 2, 4, 8

model = WaveNet(input_channels, output_channels, num_blocks, dilations)
print(model, flush=True)
model.load_state_dict(torch.load(f'./GluNet_DIATREND_{fold}_model_{history_len}.pth'))

splits = [0, 11, 22, 33, 44, 54]

if len(sys.argv) > 2: 
    fold = sys.argv[1]
    bot_range = splits[int(fold) -1]
    top_range = splits[int(fold)]
    history_len = int(sys.argv[2])
    print(f'fold number is {fold} and history length is {history_len}', flush = True)
    print(f'bot range is {bot_range} and top range is {top_range}', flush = True)
else: 
    fold = 1
    bot_range = splits[0]
    top_range = splits[1]
    history_len = 7 
    
# Example of how to define the loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)

print("Testing the model", flush=True)
# Training Loop
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

# %%
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


def prepare_dataset(segments, history_len):
    '''
    ph = 6, 30 minutes ahead
    ph = 12, 60 minutes ahead
    '''
    ph = 6
    features_list = []
    labels_list = []
    raw_glu_list = []
    
    
    # Iterate over each segment
    for segment_name, segment_df in segments.items():
       

        # Fill NaNs that might have been introduced by conversion errors
        segment_df.fillna(0, inplace=True)

        # Maximum index for creating a complete feature set
        # print("len of segment_df is ", len(segment_df))
        max_index = len(segment_df) - (history_len + ph)  # Subtracting only 15+ph to ensure i + 15 + ph is within bounds
        
        # Iterate through the data to create feature-label pairs
        for i in range(max_index):
            # Extracting features from index i to i+15
            segment_df = segment_df.reset_index(drop = True)
            features = segment_df.loc[i:i+history_len, ['glucose_value']].values
            raw_glu_list.append(segment_df.loc[i+history_len+ph, 'glucose_value'])
            features_list.append(features)
            # labels_list.append(label)
            
    print("len of features_list " + str(len(features_list)))

    return features_list, raw_glu_list



def get_test_rmse(model, test_loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to('cpu'), targets.to('cpu')
            outputs = model(inputs.permute(0, 2, 1))
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

for j in glob.glob('../diatrend_processed/*.csv'):
    # don't use overlap
    filename = os.path.basename(j)
    file_number = int(filename.split('Subject')[-1].split('.')[0])  # Extract numeric part before '.csv'
    # Exclude files within the range 0 to 248
    if bot_range <= file_number <= top_range:
        print("Processing test file ", filename, flush=True)
        test_segments = get_gdata(j)
        test_features, test_glu = prepare_dataset(test_segments, history_len)
        test_features_array = np.array(test_features)
        test_labels_array = np.array(test_glu)

        X_test = test_features_array
        y_test = test_labels_array

        # Assuming features_list and raw_glu_list are already defined
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        new_test_rmse_list.append([filename.split('.')[0], get_test_rmse(model, test_loader)])

df = pd.DataFrame(new_test_rmse_list, columns = ['rmse', 'filenumber']).to_csv(f'Glunet_Diatrend_Fold{fold}_HL{history_len}_rmse.csv', index = False)
