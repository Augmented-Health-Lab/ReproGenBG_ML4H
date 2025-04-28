
from __future__ import division, print_function

import collections
import csv
import datetime

import os
import sys 
import glob
import torch
import torch

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from scipy.interpolate import CubicSpline
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device, flush = True)

"""
Not all functions are used in this script. They are included for completeness and can be used to replicate the original functionality of Li et Al
For the purposes of our final model, we only need the glucose data.
"""


def round_up_to_nearest_five_minutes(ts):
    """
    Function to round up a timestamp to the nearest 5 minutes.
    """    

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
    """Function to preprocess the CGM data from T1DEXI dataset.

    Args:
        path (str): Path to the CGM data file.
        round (bool): Whether to round the timestamps to the nearest 5 minutes.

    Returns:
        list: A list of dictionaries containing the processed CGM data.
    """

    subject = pd.read_csv(path)
    # Group by 'Category' column
    # grouped = subject.groupby('LBCAT')
    # Create a dictionary to store the split DataFrames
    # split_dfs = {category: group for category, group in grouped}
    # selected_cgm = split_dfs["CGM"][["LBORRES", "LBDTC"]]
    # new_df_cgm = pd.DataFrame(selected_cgm)
    new_df_cgm = subject[["LBORRES", "LBDTC"]]

    # new_df_cgm['LBDTC'] = pd.to_datetime(new_df_cgm['LBDTC'], errors='coerce')  # Convert 'date' column to datetime if not already
    new_df_cgm = new_df_cgm.copy()  # Create explicit copy
    new_df_cgm.loc[:, 'LBDTC'] = pd.to_datetime(new_df_cgm['LBDTC'], errors='coerce')
    
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

def segement_data_as_6_min(data, user_id):
    """
    Segments the data into smaller chunks based on time differences.
    
    Args:
        data (pd.DataFrame): The input DataFrame containing the data to be segmented.
        user_id (int): The user ID for naming the segments.
    
    Returns:
        dict: A dictionary where keys are segment names and values are DataFrames of the segments.
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

def detect_missing_and_spline_interpolate(segments):
    """
    Performs the spline interpolation on the segments to fill in missing data points, as mentioned
    in Li et Al. This function was not used in the final model, but is included for completeness. 
    Reasons for not using it are discussed in the paper.
    """
    
    
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

def update_segments_with_meals(segments, meal_df):
    """Function to update segments with meal information.

    Args:
        segments (dict): Dictionary of segments.
        meal_df (pd.DataFrame): DataFrame containing meal information.

    Returns:
        dict: Updated segments with meal information.
    """
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

def update_segments_with_basal(segments, basal_df):
    """
    Function to update segments with basal information.
    
    Args:
        segments (dict): Dictionary of segments.
        basal_df (pd.DataFrame): DataFrame containing basal information.
    
    Returns: 
        dict: Updated segments with basal information.
    """
    
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

def preprocess_t1dexi_bolus_tempbasal(filepath, round):
    """
    Function to preprocess the bolus and temp basal data from T1DEXI dataset.
    
    Args:
        filepath (str): Path to the bolus and temp basal data file.
        round (bool): Whether to round the timestamps to the nearest 5 minutes.
    
    Return: 
        pd.DataFrame: DataFrame containing the processed bolus data.
    """
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

def update_segments_with_bolus(segments, bolus_df):
    """
    Function to update segments with bolus information.
    
    Args: 
        segments (dict): Dictionary of segments.
        bolus_df (pd.DataFrame): DataFrame containing bolus information.
    
    Return: 
        dict: Updated segments with bolus information.
    """
    
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

def label_delta_transform(labels_list):
    """
    Function to transform labels based on their percentiles, as mentioned in Li et Al
    """
    
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



def prepare_dataset(segments, history_len):
    '''
    Function to prepare the dataset for training and testing.
    
    Args:
        segments (dict): Dictionary containing segmented data.
        history_len (int): Length of the history to consider for features.

    Returns:
        features_list (list): List of feature arrays.
        raw_glu_list (list): List of raw glucose values.
    
    ph = 6, 30 minutes ahead
    ph = 12, 60 minutes ahead
    '''
    ph = 6
    
    features_list = []
    labels_list = []
    raw_glu_list = []
    
    # Iterate over each segment
    for segment_name, segment_df in segments.items():
        # Ensure all columns are of numeric type
        # segment_df['carbs'] = pd.to_numeric(segment_df['carbs'], errors='coerce')
        # segment_df['basal_rate'] = pd.to_numeric(segment_df['basal_rate'], errors='coerce')
        # segment_df['bolus_dose'] = pd.to_numeric(segment_df['bolus_dose'], errors='coerce')

        # Fill NaNs that might have been introduced by conversion errors
        segment_df.fillna(0, inplace=True)

        # Maximum index for creating a complete feature set
        print("len of segment_df is ", len(segment_df))
        max_index = len(segment_df) - (history_len + ph)  # Subtracting only 15+ph to ensure i + 15 + ph is within bounds
        
        # Iterate through the data to create feature-label pairs
        for i in range(max_index):
            # Extracting features from index i to i+15
            segment_df = segment_df.reset_index(drop = True)
            features = segment_df.loc[i:i+history_len, ['glucose_value']].values
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
def get_gdata(filename):
    """
    Function to get glucose data from a file and preprocess it.

    Args:
        filename (str): Path to the file containing glucose data.

    Returns:
        segments (dict): Dictionary containing segmented glucose data.
    """
    
    glucose = preprocess_t1dexi_cgm(filename, False)
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
    file_number = int(filename.split('.')[0])  # Extract numeric part before '.
    segments = segement_data_as_6_min(glucose_df, file_number)

    return segments

def get_test_rmse(model, test_loader):
    """
    Function to calculate RMSE on the test set.
    
    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test set.
    
    Returns:
        float: RMSE value.
    """
    
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.permute(0, 2, 1))
            predictions.append(outputs)
            actuals.append(targets)

    predictions = torch.cat(predictions).cpu().numpy()
    actuals = torch.cat(actuals).cpu().numpy()


    rmse = np.sqrt(mean_squared_error(actuals,predictions))
    print(f'RMSE on the folds: {rmse}')
    return  rmse


class WaveNetBlock(nn.Module):
    """
    Class WaveNet Block for dilated convolutions.
    """
    
    def __init__(self, in_channels, dilation):
        """
        Initialize the WaveNet block with two dilated convolutions and a residual connection.
        
        Args:
            in_channels (int): Number of input channels.
            dilation (int): Dilation rate for the convolutions.
        
        Returns:
            None
        
        """
        super(WaveNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=2, dilation=dilation, padding=1+dilation - 2^(dilation-1))
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=2, dilation=dilation, padding=dilation)
        self.res_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        """
        Forward pass through the WaveNet block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, seq_len).
        
        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        # print("shape of x: ", x.shape)
        out = F.relu(self.conv1(x))
        # print("shape of first out: ", out.shape)
        out = F.relu(self.conv2(out))
        # print("shape of second out: ", out.shape)
        res = self.res_conv(x)
        # print("shape of res: ", res.shape)
        return out + res


class WaveNet(nn.Module):
    """
    Class WaveNet for the entire model.
    """
    
    def __init__(self, in_channels, out_channels, num_blocks, dilations):
        """
        Initialize the WaveNet model with an initial convolution, multiple blocks, and final convolutions.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_blocks (int): Number of WaveNet blocks.
            dilations (list): List of dilation rates for the blocks.
        
        Returns:
            None
        """
        
        super(WaveNet, self).__init__()
        self.initial_conv = nn.Conv1d(in_channels, 32, kernel_size=2, padding=1)
        self.blocks = nn.ModuleList([WaveNetBlock(32, dilation) for dilation in dilations])
        self.final_conv1 = nn.Conv1d(32, 128, kernel_size=2, padding=0)
        self.final_conv2 = nn.Conv1d(128, 256, kernel_size=2, padding=0)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, out_channels)
        
    def forward(self, x):
        """
        Input tensor of shape (batch_size, in_channels, seq_len).
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, seq_len).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels).
        """
        
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


def main(): 

    splits = [0, 248, 1201, 1348, 1459, 1726]
    import sys


    ##############################################################################
    #
    #                                PREPROCESSING
    #
    ##############################################################################

    # if the name of the file is between 0 and 248, don't include it in the training set
    if len(sys.argv) > 2: 
        fold = sys.argv[1]
        bot_range = splits[int(fold) -1]
        top_range = splits[int(fold)]
        history_len = int(sys.argv[2])
    else: 
    # Default to first fold
        fold = 1
        bot_range = 0
        top_range = 248
        history_len = 6

    segment_list = [] 
    test_segment_list = []

    for j in glob.glob('../T1DEXI_processed/*.csv'):
        # don't use overlap
        filename = os.path.basename(j)
        file_number = int(filename.split('.')[0])  # Extract numeric part before '.csv'
        # Exclude files within the range 0 to 248
        if bot_range < file_number <= top_range:
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



    ##############################################################################
    #
    #                                TRAINING
    #
    ##############################################################################
    
    input_channels = 1  # Number of features
    output_channels = 1  # Predicting a single value (glucose level)
    
    if history_len <= 6: 
        num_blocks = 3
    else: 
        num_blocks = 4
    
    dilations = [2**i for i in range(num_blocks)]  # Dilation rates: 1, 2, 4, 8
    
    model = WaveNet(input_channels, output_channels, num_blocks, dilations)
    model = model.to(device)

    # Example of how to define the loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)


    import sys
    try:
        fold_number = int(sys.argv[1])
        history_len = int(sys.argv[2])
    except:
        fold_number = 1
        history_len = 7
        
    print(f'fold number is {fold_number} and history length is {history_len}', flush = True)
    bolus_updated_segments = merged_segments
    features_list, raw_glu_list = prepare_dataset(bolus_updated_segments, history_len)
    # Assuming features_list and raw_glu_list are already defined
    features_array = np.array(features_list)
    labels_array = np.array(raw_glu_list)

    # Step 1: Split into 80% train+val and 20% test
    X_train, X_val, y_train, y_val = train_test_split(features_array, labels_array, test_size=0.2, shuffle=False)


    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    # Convert the splits to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    # optimze torch 
    torch.backends.cudnn.benchmark = True
    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=12,     persistent_workers=True, prefetch_factor=2     )

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=12,     persistent_workers=True, prefetch_factor=2)


    # print the device being used
    print('Using device:', device, flush = True)
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)    # Move inputs to GPU
            targets = targets.to(device)  # Move targets to GPU

            optimizer.zero_grad()
            outputs = model(inputs.permute(0, 2, 1))  # Permute to match (batch, channels, seq_len)
            # use squeeze
            outputs = outputs.squeeze()
            targets = targets.squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)    # Move inputs to GPU
                targets = targets.to(device)  # Move targets to GPU

                outputs = model(inputs.permute(0, 2, 1))  # Permute to match (batch, channels, seq_len)
                outputs = outputs.squeeze()
                targets = targets.squeeze()

                loss = criterion(outputs, targets)
                val_loss += loss.item() 
        
        print(f'Epoch {epoch+1}, Validation Loss: {val_loss / len(val_loader)}', flush=True)


    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.permute(0, 2, 1))
            predictions.append(outputs)
            actuals.append(targets)

    predictions = torch.cat(predictions).cpu().numpy()
    actuals = torch.cat(actuals).cpu().numpy()


    rmse = np.sqrt(mean_squared_error(actuals,predictions))
    print(f'RMSE on validation set: {rmse}')

    # save the model 
    print(f'Saving in GluNet_{fold}_model_{history_len}.pth', flush = True)
    torch.save(model.state_dict(), f'GluNet_{fold}_model_{history_len}.pth')


    ##############################################################################
    #
    #                                TESTING
    #
    ##############################################################################


    segment_list = [] 
    test_segment_list = []
    new_test_rmse_list = []

    print(f'history length is {history_len}', flush = True)
    print(f'Fold number is {fold}', flush = True)
    print(f'prediction horizon is 6', flush = True)
    
    for j in glob.glob('../T1DEXI_processed/*.csv'):
        # don't use overlap
        filename = os.path.basename(j)
        file_number = int(filename.split('.')[0])  # Extract numeric part before '.csv'
        
        # Exclude files within the required ranges
        if bot_range <= file_number <= top_range:
            print("Processing test file ", filename, flush=True)
            test_segments = get_gdata(j)
            test_features, test_glu = prepare_dataset(test_segments, history_len=history_len)
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

    print('This is now placed in, ', f'Glunet_t1dex_Fold{fold}_HL{history_len}_rmse.csv')

    df = pd.DataFrame(new_test_rmse_list, columns = ['rmse', 'filenumber']).to_csv(f'Glunet_t1dex_Fold{fold}_HL{history_len}_rmse.csv', index = False)



if main() == '__main__':
    main()