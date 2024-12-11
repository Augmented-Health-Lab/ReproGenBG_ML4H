


from __future__ import division, print_function

import collections
import csv
import datetime
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from GlucNet_functions import *


# In[3]:


filepath = f"../OhioT1DM/2018/train/559-ws-training.xml"
tree = ET.parse(filepath)
root = tree.getroot()
glucose = read_ohio(filepath, "glucose_level", False)


# # Preprocessing

# #### P1

# In[4]:


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
glucose_df = pd.DataFrame(g_data)

# Convert glucose values to numeric type for analysis
glucose_df['glucose_value'] = pd.to_numeric(glucose_df['glucose_value'])

# Calculate percentiles
lower_percentile = np.percentile(glucose_df['glucose_value'], 2)
upper_percentile = np.percentile(glucose_df['glucose_value'], 98)

# Print thresholds
print(f"2% lower threshold: {lower_percentile}")
print(f"98% upper threshold: {upper_percentile}")


# ### P2, P3

# In[5]:


segments = segement_data_as_1hour(glucose_df)
interpolated_segements = detect_missing_and_spline_interpolate(segments)


# In[6]:


meal = read_ohio(filepath, "meal", False)

flattened_meal_data = [item[0] for item in meal]  # Take the first (and only) item from each sublist

# Convert to DataFrame
meal_df = pd.DataFrame(flattened_meal_data)

meal_df['assigned'] = False

meal_updated_segments = update_segments_with_meals(interpolated_segements, meal_df)


# In[7]:


basal = read_ohio(filepath, "basal", False)

flattened_basal_data = [item[0] for item in basal]  # Take the first (and only) item from each sublist

# Convert to DataFrame
basal_df = pd.DataFrame(flattened_basal_data)

basal_df['assigned'] = False
basal_df['end_ts'] = basal_df['ts'].shift(-1)
basal_df[:10]


# In[8]:


basal_updated_segments = update_segments_with_basal(meal_updated_segments, basal_df)


# In[9]:


# Merge Bolus into the dataframe
bolus = read_ohio_bolus_tempbasal(filepath, "bolus", False)

flattened_bolus_data = [item[0] for item in bolus]  # Take the first (and only) item from each sublist

# Convert to DataFrame
bolus_df = pd.DataFrame(flattened_bolus_data)

bolus_df['assigned'] = False
bolus_df[:10]


# In[10]:


bolus_updated_segments = update_segments_with_bolus(basal_updated_segments, bolus_df)


# In[11]:


tempbasal = read_ohio_bolus_tempbasal(filepath, "temp_basal", False)


# In[12]:


flattened_tempbasal_data = [item[0] for item in tempbasal]  # Take the first (and only) item from each sublist

# Convert to DataFrame
tempbasal_df = pd.DataFrame(flattened_tempbasal_data)

tempbasal_df['assigned'] = False
tempbasal_df[:10]


# In[13]:


# Update the segments with meal data
final_updated_segments = update_segments_with_tempbasal(bolus_updated_segments, tempbasal_df)


# # Training

# In[14]:


import torch
from torch.utils.data import DataLoader, TensorDataset


# In[15]:


features_list, labels_list = prepare_dataset(segments)


# In[16]:


# Convert lists to PyTorch tensors
features_tensor = torch.tensor(features_list, dtype=torch.float32)
labels_tensor = torch.tensor(labels_list, dtype=torch.float32).unsqueeze(1)  # Making labels tensor 2D

feature_label_tensor = TensorDataset(features_tensor, labels_tensor)


train_loader = DataLoader(feature_label_tensor, batch_size=32, shuffle=True)

# Example of using DataLoader in a training loop
for features, labels in train_loader:
    print("Features batch shape:", features.shape)
    print("Label batch shape:", labels.shape)
    # Example: print(features, labels)
    break


# In[17]:


# initialize cuda option
dtype = torch.FloatTensor # data type
ltype = torch.LongTensor # label type

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('use gpu')
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor


# In[18]:


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


# In[20]:


for inputs, targets in train_loader:
    print("Input tensor shape:", inputs.shape)
    print("Input tensor total elements:", inputs.numel())
    print("Target tensor shape:", targets.shape)
    print("Sequence length:", inputs.shape[1])
    break


# In[19]:


features_list, raw_glu_list = prepare_dataset(final_updated_segments)
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

