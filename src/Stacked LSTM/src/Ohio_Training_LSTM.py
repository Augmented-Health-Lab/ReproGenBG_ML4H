from __future__ import division, print_function

import os
import collections
import csv
import datetime
import pickle
import torch
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import torch.nn as nn
import torch.optim as optim

from datetime import datetime, timedelta
from scipy.interpolate import CubicSpline
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device, flush=True)

def prepare_dataset(segments, ph, history, input_size):
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

        segment_df.fillna(0, inplace=True)  # Fill NaNs with 0s

        # implemented for evaluating the difference for different input sizes            
        if input_size == 4: 
            segment_df['carb_effect'] = pd.to_numeric(segment_df['carb_effect'], errors='coerce')
            segment_df['steps'] = pd.to_numeric(segment_df['steps'], errors='coerce')
            segment_df['steps'] = segment_df['steps'] 
            segment_df['bolus_effect'] = pd.to_numeric(segment_df['bolus_effect'], errors='coerce')


        # Maximum index for creating a complete feature set
        max_index = len(segment_df) - (history-1+ph+1) 
        
        # Iterate through the data to create feature-label pairs
        for i in range(max_index + 1):
            # Extracting features from index i to i+history-1
            if input_size == 4:
                features = segment_df.loc[i:i+history-1, ['glucose_value',  'bolus_effect', 'carb_effect', 'steps']] 
            else: 
                features = segment_df.loc[i:i+history-1, ['glucose_value']]
            
            label = segment_df.loc[i+history-1+ph, 'glucose_value'] - segment_df.loc[i+history-1, 'glucose_value']
            
            raw_glu_list.append(segment_df.loc[i+history-1+ph, 'glucose_value'])
            features_list.append(features)
            labels_list.append(label)
            
    print("len of features_list " + str(len(features_list)),flush=True)
    
    return features_list, labels_list, raw_glu_list



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


##############################################################################
#                     
#
#                                 TRAINING 
#
#
##############################################################################

# HYPERPARAMETERS
input_size = 1 # Number of input features (Only using CGM values for now)
hidden_size = 128  # Hidden vector size
num_layers = 2  # Number of LSTM layers
output_size = 1  # Single output
dropout_prob = 0.2  # Dropout probability
ph = 6
history = int(sys.argv[1])
folder_dir = os.getcwd()

# The input training file
training_filename = "../StackedLSTM_again/processed_data/BIG_training_onlyCGM.pkl"
model_save_name = 'Ohio_HISTORY_{history}.pth'.format(history=history)
results_save_name = 'Ohio_HISTORY_{history}.csv'.format(history=history)

model_save_path = os.path.join(folder_dir, model_save_name)
results_save_path = os.path.join(folder_dir, results_save_name)

print(f'Model name: {model_save_path}', flush=True)
print(f'Results name: {results_save_path}', flush=True)

# Load the Processed training data 
with open(training_filename, 'rb') as f:
    loaded_df_dict = pickle.load(f)

step_updated_segments = loaded_df_dict
features_list, labels_list, raw_glu_list = prepare_dataset(step_updated_segments, ph, history, input_size) 

# Build training and validation loader
features_array = np.array(features_list)
labels_array = np.array(raw_glu_list) 

X_train, X_val, y_train, y_val = train_test_split(features_array, labels_array, test_size=0.2, shuffle= False)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

# Build the Tensor Dataset
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

model = StackedLSTM(input_size, hidden_size, num_layers, output_size, dropout_prob) # input_size, hidden_size, num_layers, output_size, dropout_prob
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)

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
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}', flush=True)


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
        print(f'Test Loss: {avg_loss:.4f}', flush= True)

#  save the model 
current = os.getcwd()
model_save_path = os.path.join(current, model_save_name)
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}', flush=True)

##############################################################################
#
#                                TESTING
#
##############################################################################

# First we test on the validation set
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

rmse = mean_squared_error(actuals,predictions)
print(f'RMSE on validation set: {np.sqrt(rmse)}')

plt.plot(predictions[:700], label = 'predictions')
plt.plot(actuals[:700], label = 'actuals')
plt.legend()

# Next we test on the test set

def test_model(model, test_step_updated_segments, ph, history, input_size):
    """
    Function to test the model on the test dataset.
    
    Args:
        model (nn.Module): The trained LSTM model.
        test_step_updated_segments (dict): Dictionary containing segmented DataFrames for testing.
        ph (int): Prediction horizon (in minutes).
        history (int): Length of the history to consider for each prediction (in minutes).
        input_size (int): Number of input features.
    
    Returns:
        predictions (np.ndarray): The model predictions on the test dataset.
        actuals (np.ndarray): The actual glucose values from the test dataset.
        rmse (float): The root mean squared error of the predictions. 
    """
    # Prepare for training
    features_list_test, labels_list_test, raw_glu_list_test = prepare_dataset(test_step_updated_segments, ph, history, input_size) # segments, ph, history, input_size
    
    # Build training and validation loader for test
    features_array_test = np.array(features_list_test)
    labels_array_test = np.array(raw_glu_list_test) 

    X_test, y_test = features_array_test, labels_array_test

    # Data Preparation (assuming X_train, y_train, X_val, y_val are numpy arrays)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoader
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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
    print(f'RMSE on validation set: {rmse}')
    
    return predictions, actuals, rmse


preds = []
trues = []
errors = []
fname = []

# Test the model for each subject individually. 
for file in glob.glob("../../../data/OhioT1DM/2018/test/*test.pkl"):
    
    # just use the first 3 letters of the file name which corresponds to the subject ID
    test_filename = file.split('/')[-1][:3]
    print(file, flush=True)
    with open(file, 'rb') as f:
        test_loaded_df_dict = pickle.load(f)

    print(f'Loaded test data for subject: {test_filename}', flush=True)

    # Verify the content
    test_step_updated_segments = test_loaded_df_dict
    pred, true, rmse = test_model(model, test_step_updated_segments, ph, history, input_size)
    preds.append(pred)
    trues.append(true)
    errors.append(rmse)
    print(f'RMSE on {test_filename}: {rmse}', flush=True)
    fname.append(test_filename)

# Export data to a csv file 
curr_dat = pd.DataFrame({'fname': fname, 'rmse': errors})
curr_dat.to_csv(results_save_path, index=False)
 