#!/usr/bin/env python
# coding: utf-8

# # Replicate GluNet on T1DEXI
# 
# GluNet was mainly reported as a personalized model

# In[1]:


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
import os

from datetime import datetime, timedelta
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error


# In[ ]:


ph = 6
history_len = 15


# In[20]:


def round_up_to_nearest_five_minutes(ts):
    # Parse the timestamp
    dt = datetime.strptime(ts, "%d-%m-%Y %H:%M:%S")
    
    # Calculate minutes to add to round up to the nearest 5 minutes
    minutes_to_add = (5 - dt.minute % 5) % 5
    if minutes_to_add == 0 and dt.second == 0:
        # If exactly on a 5 minute mark and second is 0, no need to add time
        minutes_to_add = 5
    
    # Add the necessary minutes
    new_dt = dt + timedelta(minutes=minutes_to_add)
    
    # Return the new timestamp in the same format
    return new_dt.strftime( "%d-%m-%Y %H:%M:%S")


def preprocess_t1dexi_cgm(path, round):

    subject = pd.read_csv(path)
    # Group by 'Category' column
    grouped = subject.groupby('LBCAT')
    # Create a dictionary to store the split DataFrames
    split_dfs = {category: group for category, group in grouped}
    selected_cgm = split_dfs["CGM"][["LBORRES", "LBDTC"]]
    new_df_cgm = pd.DataFrame(selected_cgm)

    new_df_cgm['LBDTC'] = pd.to_datetime(new_df_cgm['LBDTC'], errors='coerce')  # Convert 'date' column to datetime if not already
    new_df_cgm.sort_values('LBDTC', inplace=True)  # Sort the DataFrame by the 'date' column

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
    
    # # Assuming self.interval_timedelta is set, for example:
    # interval_timedelta = datetime.timedelta(minutes=6)  # Example timedelta of 6 minutes, providing a range for latency

    # # Create a list to store the results
    # res = []

    # # Initialize the first group
    # if not subject.empty:
    #     current_group = [subject.iloc[0]['LBORRES']]
    #     last_time = subject.iloc[0]['LBDTC']

    # # Iterate over rows in DataFrame starting from the second row
    # for index, row in subject.iloc[1:].iterrows():
    #     current_time = row['LBDTC']
    #     if (current_time - last_time) <= interval_timedelta:
    #         # If the time difference is within the limit, add to the current group
    #         current_group.append(row['LBORRES'])
    #     else:
    #         # Otherwise, start a new group
    #         res.append(current_group)
    #         current_group = [row['LBORRES']]
    #     last_time = current_time

    # # Add the last group if it's not empty
    # if current_group:
    #     res.append(current_group)
    
    # # Filter out groups with fewer than 10 glucose readings
    # res = [group for group in res if len(group) >= 10]



    # return res


# In[21]:


def segement_data_as_1hour(data):
    df = pd.DataFrame(data)

    # Calculate time differences
    df['time_diff'] = df['timestamp'].diff()

    # Identify large gaps
    df['new_segment'] = df['time_diff'] > pd.Timedelta(hours=1)

    # Find indices where new segments start
    segment_starts = df[df['new_segment']].index

    # Initialize an empty dictionary to store segments
    segments = {}
    prev_index = 0

    # Loop through each segment start and slice the DataFrame accordingly
    for i, start in enumerate(segment_starts, 1):
        segments[f'segment_{i}'] = df.iloc[prev_index:start].reset_index(drop=True)
        prev_index = start

    # Add the last segment from the last gap to the end of the DataFrame
    segments[f'segment_{len(segment_starts) + 1}'] = df.iloc[prev_index:].reset_index(drop=True)

    # Optionally remove helper columns from each segment
    for segment in segments.values():
        segment.drop(columns=['time_diff', 'new_segment'], inplace=True)
    
    return segments


# In[4]:


def detect_missing_and_spline_interpolate(segments):
    for sequence in segments:
        # sequence = "segment_3"
        detected_missing = 0
        for ts in range(len( segments[sequence]['timestamp'])-1):
            if segments[sequence]['timestamp'][ts+1] - segments[sequence]['timestamp'][ts] > timedelta(minutes = 6):
                print(sequence)
                print("before: ", segments[sequence]['timestamp'][ts])
                print("after: ", segments[sequence]['timestamp'][ts+1])
                detected_missing = 1
            
        if detected_missing == 1:
            datetime_list = np.array(pd.date_range(start=min(segments[sequence]['timestamp']), end=max(segments[sequence]['timestamp']), freq='5T').tolist())
            reference_time = min(segments[sequence]['timestamp'])

            # Convert datetime objects to the number of seconds since the reference time
            datetime_seconds_since_start = [((dt - reference_time).total_seconds())/60 for dt in datetime_list] # Make it into minute
            original_timestamp_in_segement = [((dt - reference_time).total_seconds())/60 for dt in segments[sequence]['timestamp']]

            x = original_timestamp_in_segement
            y = np.array(segments[sequence]['glucose_value'])
            cs = CubicSpline(x, y)
            xs = datetime_seconds_since_start

            interpolated_xs = cs(xs)
            time_index_interpolated = pd.date_range(start=reference_time, periods=len(interpolated_xs), freq='5T')

            # Create DataFrame from the time index and glucose values
            df_interpolated = pd.DataFrame({'timestamp': time_index_interpolated, 'glucose_value': interpolated_xs})
            segments[sequence] = df_interpolated

    return segments


# In[5]:


# Function to align and update segments with meal data
def update_segments_with_meals(segments, meal_df):
    for segment_name, segment_df in segments.items():
        # Initialize the 'carbs' column to zeros
        segment_df['carbs'] = 0

        # Iterate through each timestamp in the segment
        for i, row in segment_df.iterrows():
            # Find the closest meal timestamp and its carb information
            meal_df['time_difference'] = abs(meal_df['ts'] - row['timestamp'])
            closest_meal = meal_df.loc[meal_df['time_difference'].idxmin()]
            
            # Check if the closest meal is within 5 minutes
            if closest_meal['time_difference'] <= pd.Timedelta(minutes=5):
                # Ensure that the meal is assigned to only one segment and is the closest
                if not meal_df.at[closest_meal.name, 'assigned']:
                    segment_df.at[i, 'carbs'] = closest_meal['carbs']
                    meal_df.at[closest_meal.name, 'assigned'] = True  # Mark as assigned
                else:
                    # Check if the current timestamp is closer than the one it was assigned to
                    assigned_index = segment_df[segment_df['carbs'] == closest_meal['carbs']].index[0]
                    if row['timestamp'] - closest_meal['ts'] < segment_df.at[assigned_index, 'timestamp'] - closest_meal['ts']:
                        # Reassign the meal to the new closer timestamp
                        segment_df.at[assigned_index, 'carbs'] = 0  # Remove carbs from previously assigned timestamp
                        segment_df.at[i, 'carbs'] = closest_meal['carbs']  # Assign carbs to the new closer timestamp
            # else:
            #     print(f"Meal type {meal['type']} on {meal['ts']} is too far from closest timestamp in {closest_segment} with a difference of {closest_diff}.")

    return segments


# In[6]:


# Function to align and update segments with meal data
def update_segments_with_basal(segments, basal_df):
    for segment_name, segment_df in segments.items():
        # Initialize the 'carbs' column to zeros
        segment_df['basal_rate'] = None

        # Iterate through each timestamp in the segment
        for i, row in segment_df.iterrows():
            # Find the closest meal timestamp and its carb information
            for _, basal_row in basal_df.iterrows():
                if basal_row['ts'] <= row['timestamp'] < (basal_row['end_ts'] if pd.notna(basal_row['end_ts']) else pd.Timestamp('2099-12-31')):
                    segment_df.at[i, 'basal_rate'] = basal_row['value']
                    break

    return segments


# In[7]:


# Read in bolus and temp basal information
# Need to set the 
def preprocess_t1dexi_bolus_tempbasal(filepath, round):
    subject_facm = pd.read_csv(filepath)
    # Group by 'Category' column
    grouped = subject_facm.groupby('FACAT')

    split_dfs = {category: group for category, group in grouped}
    # Step 1: Extract the desired columns
    new_df_bolus = split_dfs["BOLUS"][["FAORRES", "FADTC"]]
    new_df_bolus['FADTC'] = pd.to_datetime(new_df_bolus['FADTC'], format="%Y-%m-%d %H:%M:%S")
    new_df_bolus.reset_index(drop=True, inplace=True)
    new_df_bolus = new_df_bolus.rename(columns={'FAORRES': 'dose', 'FADTC': 'ts_begin'})
    new_df_bolus['assigned'] = False
    # new_df_bolus['end_ts'] = new_df_bolus['ts_begin'].shift(-1)
    return new_df_bolus


# In[8]:


def update_segments_with_bolus(segments, bolus_df):
    for segment_name, segment_df in segments.items():
        # Initialize the 'dose' column to zeros
        segment_df['bolus_dose'] = 0

        # Iterate through each timestamp in the segment
        for i, row in segment_df.iterrows():
            # Find the closest bolus timestamp and its carb information
            bolus_df['time_difference'] = abs(bolus_df['ts_begin'] - row['timestamp'])
            closest_bolus = bolus_df.loc[bolus_df['time_difference'].idxmin()]
            
            # Check if the closest bolus is within 5 minutes
            if closest_bolus['time_difference'] <= pd.Timedelta(minutes=5):
                # Ensure that the bolus is assigned to only one segment and is the closest
                if not bolus_df.at[closest_bolus.name, 'assigned']:
                    segment_df.at[i, 'bolus_dose'] = closest_bolus['dose']
                    bolus_df.at[closest_bolus.name, 'assigned'] = True  # Mark as assigned
                else:
                    # Check if the current timestamp is closer than the one it was assigned to
                    assigned_index = segment_df[segment_df['bolus_dose'] == closest_bolus['dose']].index[0]
                    if row['timestamp'] - closest_bolus['ts_begin'] < closest_bolus['ts_begin'] - segment_df.at[assigned_index, 'timestamp']:
                        # Reassign the bolus to the new closer timestamp
                        segment_df.at[assigned_index, 'bolus_dose'] = 0  # Remove dose from previously assigned timestamp
                        segment_df.at[i, 'bolus_dose'] = closest_bolus['dose']  # Assign dose to the new closer timestamp
            # else:
            #     print(f"bolus type {bolus['type']} on {bolus['ts']} is too far from closest timestamp in {closest_segment} with a difference of {closest_diff}.")

    return segments


# In[9]:


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

def prepare_dataset(segments, ph):
    '''
    ph = 6, 30 minutes ahead
    ph = 12, 60 minutes ahead
    '''
    features_list = []
    labels_list = []
    raw_glu_list = []
    
    # Iterate over each segment
    for segment_name, segment_df in segments.items():
        # Ensure all columns are of numeric type
        segment_df['carbs'] = pd.to_numeric(segment_df['carbs'], errors='coerce')
        segment_df['basal_rate'] = pd.to_numeric(segment_df['basal_rate'], errors='coerce')
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
            features = segment_df.loc[i:i+history_len, ['glucose_value', 'carbs', 'basal_rate', 'bolus_dose']].values
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


# In[11]:


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

input_channels = 4  # Number of features
output_channels = 1  # Predicting a single value (glucose level)
num_blocks = 4  # Number of WaveNet blocks
dilations = [2**i for i in range(num_blocks)]  # Dilation rates: 1, 2, 4, 8

model = WaveNet(input_channels, output_channels, num_blocks, dilations)
print(model)

# Example of how to define the loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)


# # Implementation

# In[12]:


overlap = ['854.csv',
 '979.csv',
 '816.csv',
 '953.csv',
 '981.csv',
 '1617.csv',
 '1343.csv',
 '987.csv',
 '255.csv',
 '907.csv',
 '856.csv',
 '354.csv',
 '894.csv',
 '862.csv',
 '900.csv',
 '695.csv']

#  '1287.csv','1112.csv' no basal  '85.csv', '911.csv',


# In[13]:


subject = pd.read_csv(f"../LB_split/854.csv")


# In[14]:


glucose = preprocess_t1dexi_cgm(f"../LB_split/854.csv", False)
glucose[:5]


# In[15]:


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
print(f"2% lower threshold: {lower_percentile}")
print(f"98% upper threshold: {upper_percentile}")

glucose_df


# # Spline interpolation and extrapolation

# In[16]:


# Example: print each segment
segments = segement_data_as_1hour(glucose_df)
interpolated_segements = detect_missing_and_spline_interpolate(segments)


# # Align other factors with the glucose information
# 
# ## Include meal info

# In[17]:


meal = pd.read_csv("../ML_split/854.csv")
selected_meal_column = meal[["MLDOSE", "MLDTC"]]

meal_df = selected_meal_column.rename(columns={'MLDOSE': 'carbs', 'MLDTC': 'ts'})
meal_df['ts'] = pd.to_datetime(meal_df['ts'], format="%Y-%m-%d %H:%M:%S")

meal_df['assigned'] = False

# Extract unique dates
unique_dates = meal_df['ts'].dt.date.unique()

# Convert to list
meal_avaiable_dates_list = unique_dates.tolist()

cleaned_segments = {}

# Iterate through each segment and filter by unique dates
for segment_name, df in interpolated_segements.items():
    # Convert timestamp column to datetime and then extract the date part
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    
    # Filter the DataFrame to only include rows where the date is in unique_dates_list
    filtered_df = df[df['date'].isin(meal_avaiable_dates_list)]
    
    # Drop the 'date' column as it's no longer needed
    filtered_df = filtered_df.drop(columns=['date'])
    
    # Store the filtered DataFrame in the cleaned_segments dictionary
    cleaned_segments[segment_name] = filtered_df

# Update the segments with meal data
meal_updated_segments = update_segments_with_meals(cleaned_segments, meal_df)


# # Include basal

# In[18]:


subject_facm = pd.read_csv(f"../FACM_split/854.csv")
# Group by 'Category' column
grouped = subject_facm.groupby('FACAT')

split_dfs = {category: group for category, group in grouped}
# Step 1: Extract the desired columns
new_df_basal = split_dfs["BASAL"][["FAORRES", "FADTC"]]
new_df_basal['FADTC'] = pd.to_datetime(new_df_basal['FADTC'], format="%Y-%m-%d %H:%M:%S")
new_df_basal.reset_index(drop=True, inplace=True)
new_df_basal = new_df_basal.rename(columns={'FAORRES': 'value', 'FADTC': 'ts'})
new_df_basal['assigned'] = False
new_df_basal['end_ts'] = new_df_basal['ts'].shift(-1)
new_df_basal[:10]


# In[19]:


# Update the segments with meal data
basal_updated_segments = update_segments_with_basal(meal_updated_segments, new_df_basal)


# # Include bolus

# In[ ]:


new_df_bolus = preprocess_t1dexi_bolus_tempbasal(f"../FACM_split/854.csv", False)
bolus_updated_segments = update_segments_with_bolus(basal_updated_segments, new_df_bolus)


# ## Method to deal with large meal missingness
# 
# 1. Use 0 to impute 
# 2. Only use the days with meal record

# In[ ]:


# Try with the method 2
# Therefore we have the meal_avaiable_dates_list


# # Construct X and y, training and test

# In[ ]:


# # Build training and validation loader
# features_array = np.array(features_list)
# labels_array = np.array(raw_glu_list) # Maybe need to replace this

# X_train, X_val, y_train, y_val = train_test_split(features_array, labels_array, test_size=0.2, shuffle= False)

# # Data Preparation (assuming X_train, y_train, X_val, y_val are numpy arrays)
# X_train = torch.tensor(X_train, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.float32)
# X_val = torch.tensor(X_val, dtype=torch.float32)
# y_val = torch.tensor(y_val, dtype=torch.float32)

# # Create DataLoader
# train_dataset = TensorDataset(X_train, y_train)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
# val_dataset = TensorDataset(X_val, y_val)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# In[ ]:


bolus_updated_segments


# In[ ]:


features_list, raw_glu_list = prepare_dataset(bolus_updated_segments, ph)
# Assuming features_list and raw_glu_list are already defined
features_array = np.array(features_list)
labels_array = np.array(raw_glu_list)

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
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# In[ ]:


# Training Loop
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.permute(0, 2, 1))  # Permute to match (batch, channels, seq_len)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs.permute(0, 2, 1))  # Permute to match (batch, channels, seq_len)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Validation Loss: {val_loss / len(val_loader)}')


# In[ ]:


# Calculate RMSE on test set
model.eval()
predictions = []
actuals = []
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs.permute(0, 2, 1))
        predictions.append(outputs)
        actuals.append(targets)

predictions = torch.cat(predictions).cpu().numpy()
actuals = torch.cat(actuals).cpu().numpy()


rmse = root_mean_squared_error(actuals,predictions)
print(f'RMSE on test set: {rmse}')


# # Implement on the group

# In[ ]:


overlap


# In[ ]:


test_rmse_list = []
for ffile in overlap:

    subject = pd.read_csv(f"../LB_split/{ffile}")
    glucose = preprocess_t1dexi_cgm(f"../LB_split/{ffile}", False)
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
    segments = segement_data_as_1hour(glucose_df)
    interpolated_segements = detect_missing_and_spline_interpolate(segments)
    meal = pd.read_csv(f"../ML_split/{ffile}")
    selected_meal_column = meal[["MLDOSE", "MLDTC"]]

    meal_df = selected_meal_column.rename(columns={'MLDOSE': 'carbs', 'MLDTC': 'ts'})
    meal_df['ts'] = pd.to_datetime(meal_df['ts'], format="%Y-%m-%d %H:%M:%S")

    meal_df['assigned'] = False

    # Extract unique dates
    unique_dates = meal_df['ts'].dt.date.unique()

    # Convert to list
    meal_avaiable_dates_list = unique_dates.tolist()

    cleaned_segments = {}

    # Iterate through each segment and filter by unique dates
    for segment_name, df in interpolated_segements.items():
        # Convert timestamp column to datetime and then extract the date part
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        
        # Filter the DataFrame to only include rows where the date is in unique_dates_list
        filtered_df = df[df['date'].isin(meal_avaiable_dates_list)]
        
        # Drop the 'date' column as it's no longer needed
        filtered_df = filtered_df.drop(columns=['date'])
        
        # Store the filtered DataFrame in the cleaned_segments dictionary
        cleaned_segments[segment_name] = filtered_df

    # Update the segments with meal data
    meal_updated_segments = update_segments_with_meals(cleaned_segments, meal_df)

    subject_facm = pd.read_csv(f"../FACM_split/{ffile}")
    # Group by 'Category' column
    grouped = subject_facm.groupby('FACAT')

    split_dfs = {category: group for category, group in grouped}
    # Step 1: Extract the desired columns
    new_df_basal = split_dfs["BASAL"][["FAORRES", "FADTC"]]
    new_df_basal['FADTC'] = pd.to_datetime(new_df_basal['FADTC'], format="%Y-%m-%d %H:%M:%S")
    new_df_basal.reset_index(drop=True, inplace=True)
    new_df_basal = new_df_basal.rename(columns={'FAORRES': 'value', 'FADTC': 'ts'})
    new_df_basal['assigned'] = False
    new_df_basal['end_ts'] = new_df_basal['ts'].shift(-1)
    
    basal_updated_segments = update_segments_with_basal(meal_updated_segments, new_df_basal)

    new_df_bolus = preprocess_t1dexi_bolus_tempbasal(f"../FACM_split/{ffile}", False)
    bolus_updated_segments = update_segments_with_bolus(basal_updated_segments, new_df_bolus)

    features_list, raw_glu_list = prepare_dataset(bolus_updated_segments, ph)
    # Assuming features_list and raw_glu_list are already defined
    features_array = np.array(features_list)
    labels_array = np.array(raw_glu_list)

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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = WaveNet(input_channels, output_channels, num_blocks, dilations)

    # Example of how to define the loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)

    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.permute(0, 2, 1))  # Permute to match (batch, channels, seq_len)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs.permute(0, 2, 1))  # Permute to match (batch, channels, seq_len)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Validation Loss: {val_loss / len(val_loader)}')

    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs.permute(0, 2, 1))
            predictions.append(outputs)
            actuals.append(targets)

    predictions = torch.cat(predictions).cpu().numpy()
    actuals = torch.cat(actuals).cpu().numpy()


    rmse = root_mean_squared_error(actuals,predictions)
    print(f'RMSE on test set: {rmse}')
    test_rmse_list.append(rmse)


# In[ ]:




