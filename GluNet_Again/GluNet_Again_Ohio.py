
from __future__ import division, print_function
import sys
import collections
import csv
import datetime
import xml.etree.ElementTree as ET
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import glob

from GlucNet_functions import *

PH = 6
HISTORY = 7

# # %%
# filepath = f"../OhioT1DM/2018/train/559-ws-training.xml"
# tree = ET.parse(filepath)
# root = tree.getroot()
# glucose = read_ohio(filepath, "glucose_level", False)


# print("Working with file: ", filepath, flush = True)
# # %% [markdown]
# # # Preprocessing

# # %% [markdown]
# # #### P1

# # %%
# glucose_dict = {entry[0]['ts']: entry[0]['value'] for entry in glucose}

# # Create the multi-channel database
# g_data = []
# for timestamp in glucose_dict:
#     record = {
#         'timestamp': timestamp,
#         'glucose_value': glucose_dict[timestamp],
#         'meal_type': None,
#         'meal_carbs': 0
#     }
    
#     g_data.append(record)
# glucose_df = pd.DataFrame(g_data)

# # Convert glucose values to numeric type for analysis
# glucose_df['glucose_value'] = pd.to_numeric(glucose_df['glucose_value'])

# # Calculate percentiles
# lower_percentile = np.percentile(glucose_df['glucose_value'], 2)
# upper_percentile = np.percentile(glucose_df['glucose_value'], 98)

# # Print thresholds
# print(f"2% lower threshold: {lower_percentile}")
# print(f"98% upper threshold: {upper_percentile}")

# # %% [markdown]
# # ### P2, P3

# # %%
# segments = segement_data_as_1hour(glucose_df)
# interpolated_segements = detect_missing_and_spline_interpolate(segments)

# %%
# meal = read_ohio(filepath, "meal", False)

# flattened_meal_data = [item[0] for item in meal]  # Take the first (and only) item from each sublist

# # Convert to DataFrame
# meal_df = pd.DataFrame(flattened_meal_data)

# meal_df['assigned'] = False

# meal_updated_segments = update_segments_with_meals(interpolated_segements, meal_df)


# # %%
# basal = read_ohio(filepath, "basal", False)

# flattened_basal_data = [item[0] for item in basal]  # Take the first (and only) item from each sublist

# # Convert to DataFrame
# basal_df = pd.DataFrame(flattened_basal_data)

# basal_df['assigned'] = False
# basal_df['end_ts'] = basal_df['ts'].shift(-1)
# basal_df[:10]

# # %%
# basal_updated_segments = update_segments_with_basal(meal_updated_segments, basal_df)


# # %%
# # Merge Bolus into the dataframe
# bolus = read_ohio_bolus_tempbasal(filepath, "bolus", False)

# flattened_bolus_data = [item[0] for item in bolus]  # Take the first (and only) item from each sublist

# # # Convert to DataFrame
# bolus_df = pd.DataFrame(flattened_bolus_data)

# bolus_df['assigned'] = False
# bolus_df[:10]

# # %%
# bolus_updated_segments = update_segments_with_bolus(basal_updated_segments, bolus_df)

# # %%
# tempbasal = read_ohio_bolus_tempbasal(filepath, "temp_basal", False)

# # %%
# flattened_tempbasal_data = [item[0] for item in tempbasal]  # Take the first (and only) item from each sublist

# # # Convert to DataFrame
# tempbasal_df = pd.DataFrame(flattened_tempbasal_data)

# tempbasal_df['assigned'] = False
# tempbasal_df[:10]

# # %%


# # # Update the segments with meal data
# final_updated_segments = update_segments_with_tempbasal(bolus_updated_segments, tempbasal_df)

# # %% [markdown]
# # # Training

# # %%


# %%
import torch
from torch.utils.data import DataLoader, TensorDataset

# %%
def label_delta_transform(labels_list):
    # label_lower_percentile = -12.75
    # label_upper_percentile = 12.85
    label_lower_percentile = np.percentile(labels_list, 10)
    label_upper_percentile = np.percentile(labels_list, 90)
    transformed_labels = []
    for label in labels_list:
        if label <= label_lower_percentile:
            transformed_labels.append(1)
        elif label_lower_percentile < label < label_upper_percentile:
            trans_label = round((256/(label_upper_percentile - label_lower_percentile))*(label + abs(label_lower_percentile) + 0.05))
            transformed_labels.append(trans_label)
        elif label >= label_upper_percentile:
            transformed_labels.append(256)
    return transformed_labels


# def prepare_dataset(segments, ph):
#     '''
#     ph = 6, 30 minutes ahead
#     ph = 12, 60 minutes ahead
#     '''
#     features_list = []
#     labels_list = []
#     raw_glu_list = []
    
#     # Iterate over each segment
#     for segment_name, segment_df in segments.items():
#         # Ensure all columns are of numeric type
#         segment_df['carbs'] = pd.to_numeric(segment_df['carbs'], errors='coerce')
#         segment_df['basal_rate'] = pd.to_numeric(segment_df['basal_rate'], errors='coerce')
#         segment_df['bolus_dose'] = pd.to_numeric(segment_df['bolus_dose'], errors='coerce')

#         # Fill NaNs that might have been introduced by conversion errors
#         segment_df.fillna(0, inplace=True)

#         # Maximum index for creating a complete feature set
#         max_index = len(segment_df) - (15+ph+1)  # Subtracting 22 because we need to predict index + 21 and need index + 15 to exist
        
#         # Iterate through the data to create feature-label pairs
#         for i in range(max_index + 1):
#             # Extracting features from index i to i+15
#             features = segment_df.loc[i:i+15, ['glucose_value', 'carbs', 'basal_rate', 'bolus_dose']].values#.flatten()
#             # Extracting label for index i+21
#             # Do the label transform
#             label = segment_df.loc[i+15+ph, 'glucose_value'] - segment_df.loc[i+15, 'glucose_value']
            
#             raw_glu_list.append(segment_df.loc[i+15+ph, 'glucose_value'])
#             features_list.append(features)
#             labels_list.append(label)
            
#     print("len of features_list " + str(len(features_list)))
#     print("len of labels_list " + str(len(labels_list)))
#     new_labels_list = label_delta_transform(labels_list)    
#     print("after label transform. the len of label list "+str(len(new_labels_list)))    
#     return features_list, labels_list, new_labels_list, raw_glu_list

def prepare_dataset(segments,history_len = 6, ph = 6):
    '''
    ph = 6, 30 minutes ahead
    ph = 12, 60 minutes ahead
    '''
    features_list = []
    labels_list = []
    raw_glu_list = []
    
    
    # Iterate over each segment
    for segment_name, segment_df in segments.items():
        
        # rename carb_effect to carbs if carb_effect is in the dictionary 
        if 'carb_effect' in segment_df.columns:
            segment_df = segment_df.rename(columns = {'carb_effect': 'carbs'})
        
        if 'bolus_effect' in segment_df.columns:
            segment_df = segment_df.rename(columns = {'bolus_effect': 'bolus_dose'})
        
        # print all the keys in the dictionary
        print("keys in the dictionary are ", segment_df.keys(), flush = True)
        
        
        # Ensure all columns are of numeric type
        segment_df['carbs'] = pd.to_numeric(segment_df['carbs'], errors='coerce')
        # segment_df['basal_rate'] = pd.to_numeric(segment_df['basal_rate'], errors='coerce')
        segment_df['bolus_dose'] = pd.to_numeric(segment_df['bolus_dose'], errors='coerce')

        # Fill NaNs that might have been introduced by conversion errors
        segment_df.fillna(0, inplace=True)

        # Maximum index for creating a complete feature set
        print("len of segment_df is ", len(segment_df))
        max_index = len(segment_df) - (history_len + ph)  # Subtracting only 15+ph to ensure i + 15 + ph is within bounds
        
        # Iterate through the data to create feature-label pairs
        for i in range(max_index):
            # Extracting features from index i to i+15
            segment_df = segment_df.reset_index(drop = True)
            features = segment_df.loc[i:i+history_len, ['glucose_value', 'carbs', 'bolus_dose']].values
            # Extracting label for index i+15+ph
            # label = segment_df.loc[i+15+ph, 'glucose_value'] - segment_df.loc[i+15, 'glucose_value']
            
            raw_glu_list.append(segment_df.loc[i+history_len+ph, 'glucose_value'])
            features_list.append(features)
            # labels_list.append(label)
            
    print("len of features_list " + str(len(features_list)))
    # print("len of labels_list " + str(len(labels_list)))
    
    # new_labels_list = label_delta_transform(labels_list)    
    # print("after label transform, the len of label list "+str(len(new_labels_list)))    
    
    return features_list, raw_glu_list

# %%
# load train data from the .pkl file
file_num = 'ALL'
HISTORY = int(sys.argv[1])

with open(f'../StackedLSTM_again/processed_data/BIG_training_data.pkl', 'rb') as f:
    final_updated_segments= pickle.load(f)
    
final_updated_segments = {k: final_updated_segments[k] for k in list(final_updated_segments)[:20]}


features_list, labels_list = prepare_dataset(final_updated_segments, history_len=HISTORY)

# %%
features_array = np.array(features_list)
labels_array = np.array(labels_list)

# Step 1: Split into 80% train+val and 20% test
X_temp, X_test, y_temp, y_test = train_test_split(features_array, labels_array, test_size=0.2, shuffle=False)

# Step 2: Split the 80% into 70% train and 10% val (0.7/0.8 = 0.875)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, shuffle=False)

# Convert the splits to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)

val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# %%
# Convert lists to PyTorch tensors
features_array = np.array(features_list)
labels_array = np.array(labels_list)

features_tensor = torch.tensor(features_array, dtype=torch.float32)
labels_tensor = torch.tensor(labels_array, dtype=torch.float32).unsqueeze(1)  # Making labels tensor 2D

feature_label_tensor = TensorDataset(features_tensor, labels_tensor)


train_loader = DataLoader(feature_label_tensor, batch_size=32, shuffle=True)

# Example of using DataLoader in a training loop
for features, labels in train_loader:
    print("Features batch shape:", features.shape)
    print("Label batch shape:", labels.shape)
    # Example: print(features, labels)
    break

# %%
# initialize cuda option
dtype = torch.FloatTensor # data type
ltype = torch.LongTensor # label type

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('use gpu')
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor

# %%
# Build the dilate CNN
import torch
import torch.nn as nn
import torch.nn.functional as F

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

input_channels = 3 # Number of features
output_channels = 1  # Predicting a single value (glucose level)
num_blocks = 4  # Number of WaveNet blocks
dilations = [2**i for i in range(num_blocks)]  # Dilation rates: 1, 2, 4, 8

model = WaveNet(input_channels, output_channels, num_blocks, dilations)

# Example of how to define the loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)


# %%
for inputs, targets in train_loader:
    print("Input tensor shape:", inputs.shape)
    print("Input tensor total elements:", inputs.numel())
    print("Target tensor shape:", targets.shape)
    print("Sequence length:", inputs.shape[1])
    break


# %%

# Training Loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.permute(0, 2, 1))  # Permute to match (batch, channels, seq_len)
        outputs = outputs.squeeze()  # Remove extra dimensions if present
        targets = targets.squeeze()  # Remove extra dimensions if present

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs.permute(0, 2, 1))  # Permute to match (batch, channels, seq_len)
            outputs = outputs.squeeze()  # Remove extra dimensions if present
            targets = targets.squeeze()  # Remove extra dimensions if present

            loss = criterion(outputs, targets)
            val_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Validation Loss: {val_loss / len(val_loader)}', flush=True)

# %%
# save model
torch.save(model, f'./glucnet_model_OHIO_ALL_FEATURES_{file_num}_H_{HISTORY}.pth')

# %%
def prepare_dataset_test_dataset( file_dir, HISTORY = 6):
        # test data
    g_data = []

    print("file_dir is ", file_dir)
    test_file_path = file_dir 
    test_glucose = read_ohio(test_file_path, "glucose_level", False)
    test_glucose_dict = {entry[0]['ts']: entry[0]['value'] for entry in test_glucose}

    for timestamp in test_glucose_dict:
        record = {
            'timestamp': timestamp,
            'glucose_value': test_glucose_dict[timestamp],
        }
        
        g_data.append(record)
    test_glucose_df = pd.DataFrame(g_data)
    test_glucose_df['glucose_value'] = pd.to_numeric(test_glucose_df['glucose_value'])

    test_segmebts = segement_data_as_1hour(test_glucose_df)
    test_interpolated_segments = detect_missing_and_spline_interpolate(test_segmebts)

    test_meal = read_ohio(test_file_path, "meal", False)
    flattened_test_meal_data = [item[0] for item in test_meal]  # Take the first (and only) item from each sublist
    test_meal_df = pd.DataFrame(flattened_test_meal_data)
    test_meal_df['assigned'] = False
    test_meal_updated_segments = update_segments_with_meals(test_interpolated_segments, test_meal_df)

    test_basal = read_ohio(test_file_path, "basal", False)
    flattened_test_basal_data = [item[0] for item in test_basal]  # Take the first (and only) item from each sublist
    test_basal_df = pd.DataFrame(flattened_test_basal_data)
    test_basal_df['assigned'] = False
    test_basal_df['end_ts'] = test_basal_df['ts'].shift(-1)
    test_basal_updated_segments = update_segments_with_basal(test_meal_updated_segments, test_basal_df)

    test_bolus = read_ohio_bolus_tempbasal(test_file_path, "bolus", False)
    flattened_test_bolus_data = [item[0] for item in test_bolus]  # Take the first (and only) item from each sublist
    test_bolus_df = pd.DataFrame(flattened_test_bolus_data)
    test_bolus_df['assigned'] = False
    test_bolus_updated_segments = update_segments_with_bolus(test_basal_updated_segments, test_bolus_df)

    test_tempbasal = read_ohio_bolus_tempbasal(test_file_path, "temp_basal", False)
    flattened_test_tempbasal_data = [item[0] for item in test_tempbasal]  # Take the first (and only) item from each sublist
    test_tempbasal_df = pd.DataFrame(flattened_test_tempbasal_data)
    test_tempbasal_df['assigned'] = False

    test_final_updated_segments = update_segments_with_tempbasal(test_bolus_updated_segments, test_tempbasal_df)
    test_features_list, test_labels_list = prepare_dataset(test_final_updated_segments, history_len= HISTORY)

    test_features_array = np.array(test_features_list)
    test_labels_array = np.array(test_labels_list)

    X_test, y_test = torch.tensor(test_features_array, dtype=torch.float32), torch.tensor(test_labels_array, dtype=torch.float32)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return test_loader



# %%
# # load the model 
# input_channels = 4  # Number of features
# output_channels = 1
# num_blocks = 4  # Number of WaveNet blocks
# dilations = [2**i for i in range(num_blocks)]  # Dilation rates: 1, 2, 4, 8
# model = WaveNet(input_channels, output_channels, num_blocks, dilations)
# # # PH = 12
# # file_num = 'all'

# # model.load_state_dict(torch.load(f'./glucnet_model_100_epoch_ph12.pth'))

# %%

# preds = []
# trues = []
# errors = []
# fname = []

# for file in glob.glob("../OhioT1DM/2018/test/*.xml"):
#     test_filename = file
#     # load the pkl file
#     # with open(test_filename, 'rb') as f:
#     #     test_segments = pickle.load(f)
#     # test_loader = prepare_dataset(test_segments, history_len=HISTORY)
    
#     test_loader = prepare_dataset_test_dataset(test_filename, HISTORY)
    
#     # Verify the content
#     # Calculate RMSE on test set
#     model.eval()
#     predictions = []
#     actuals = []
#     with torch.no_grad():
#         for inputs, targets in test_loader:
#             outputs = model(inputs.permute(0, 2, 1))
#             predictions.append(outputs)
#             actuals.append(targets)
            
#     predictions = torch.cat(predictions).cpu().numpy()
#     actuals = torch.cat(actuals).cpu().numpy()

#     rmse = np.sqrt(mean_squared_error(actuals,predictions))
#     print(f'RMSE on test set: {rmse}')



#     preds.append(predictions)
#     trues.append(actuals)
#     errors.append(rmse)
#     fname.append(test_filename.split('-ws')[0][-3:])


# # %%
# curr_dat = pd.DataFrame({'fname': fname, 'rmse': errors})
# curr_dat.to_csv(f'wavenet_ph_{PH}_{file_num}_all_values.csv', index=False)

# %%
