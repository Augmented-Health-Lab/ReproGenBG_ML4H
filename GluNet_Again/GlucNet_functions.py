# LSTM Functions
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
import pickle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

                
def round_up_to_nearest_five_minutes(ts):
    # Parse the timestamp
    dt = datetime.strptime(ts, "%d-%m-%Y %H:%M:%S")
    
    # Calculate minutes to add to round up to the nearest 5 minutes
    minutes_to_add = (5 - dt.minute % 5) % 5
    if minutes_to_add == 0 and dt.second == 0:
        # If exactly on a 5 minute mark and second is 0, no need to add time
        minutes_to_add = 0
    
    # Add the necessary minutes
    new_dt = dt + timedelta(minutes=minutes_to_add)
    
    # Return the new timestamp in the same format
    return new_dt.strftime( "%d-%m-%Y %H:%M:%S")

# Need to set the 
def read_ohio(filepath, category, round):
    tree = ET.parse(filepath)
    root = tree.getroot()
    # interval_timedelta = datetime.timedelta(minutes=interval_timedelta)

    res = []
    for item in root.findall(category):
        entry0 = item[0].attrib
        if round == True:
            adjusted_ts = round_up_to_nearest_five_minutes(entry0['ts'])
            entry0['ts'] = adjusted_ts
        ts = entry0['ts']
        entry0['ts'] = datetime.strptime(ts, "%d-%m-%Y %H:%M:%S")
        res.append([entry0])
        for i in range(1, len(item)):
            # last_entry = item[i - 1].attrib
            entry = item[i].attrib
            # t1 = datetime.datetime.strptime(entry["ts"], "%d-%m-%Y %H:%M:%S")
            # t0 = datetime.datetime.strptime(last_entry["ts"], "%d-%m-%Y %H:%M:%S")
            # delt = t1 - t0
            # if category == "glucose_level":
            #     if delt <= interval_timedelta:
            #         res[-1].append([entry])
            #     else:
            #         res.append([entry])
            # else:
            ts = entry['ts']
            if round == True:
                adjusted_ts = round_up_to_nearest_five_minutes(ts)
                entry['ts'] = adjusted_ts
            ts = entry['ts']
            entry['ts'] = datetime.strptime(ts, "%d-%m-%Y %H:%M:%S")
            res.append([entry])
    return res

def transfer_into_table(glucose):
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
    glucose_df['glucose_value'] = glucose_df['glucose_value'] # Shrink to its 1/100 for scaling

    return glucose_df


def segement_data_as_15min(data):
    df = pd.DataFrame(data)

    # Calculate time differences
    df['time_diff'] = df['timestamp'].diff()

    # Identify large gaps
    df['new_segment'] = df['time_diff'] > pd.Timedelta(minutes=15)

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


def find_closest_glucose_index(glucose_df, meal_time, threshold_seconds=300):
    time_diffs = (glucose_df['timestamp'] - meal_time).abs()
    within_threshold = time_diffs < pd.Timedelta(seconds=threshold_seconds)
    if within_threshold.any():
        closest_index = time_diffs[within_threshold].idxmin()
        return closest_index
    return None

def update_segments_with_meals(segments, meal_df):
    for segment_name, segment_df in segments.items():
        # Initialize the 'carbs' column to zeros
        segment_df['carb_effect'] = 0

        for index, meal_row in meal_df.iterrows():
            meal_time = meal_row['ts']
            closest_glucose_idx = find_closest_glucose_index(segment_df, meal_time)
            
            if closest_glucose_idx is not None:
                segment_df.loc[closest_glucose_idx, 'carb_effect'] = int(meal_row['carb_effect'])
                meal_df.loc[index, 'assigned'] = True


    return segments


def expand_meal_entry(meal_row):
    meal_time = meal_row['ts']
    end_effect_time = meal_time + timedelta(hours=3)
    carb = float(meal_row['carbs'])

    c_eff_list = [0, 0, 0, ]

    for i in range(1, 10):
        c_eff = (i * 0.111) * carb
        if c_eff > carb:
            print("C_eff > carb")
            c_eff = carb
        c_eff_list.append(c_eff)

    for j in range(1, 25):
        c_eff = (1 - (j * 0.028)) * carb
        if c_eff < 0:
            print("C_eff < 0")
            c_eff = 0
        c_eff_list.append(c_eff)

    timestamp_list = pd.date_range(start=meal_time, end=end_effect_time, freq='5min')
    d = {"ts": timestamp_list[:-1], "carb_effect": c_eff_list}
    meal_effect_df = pd.DataFrame(data = d)

    return meal_effect_df

    
def read_ohio_bolus_tempbasal(filepath, category, round):
    tree = ET.parse(filepath)
    root = tree.getroot()
    # interval_timedelta = datetime.timedelta(minutes=interval_timedelta)

    res = []
    for item in root.findall(category):
        if len(item) == 0:
            continue  # Skip if the item has no children
            
        entry0 = item[0].attrib
        if round == True:
            adjusted_ts = round_up_to_nearest_five_minutes(entry0['ts_begin'])
            entry0['ts_begin'] = adjusted_ts
            adjusted_ts = round_up_to_nearest_five_minutes(entry0['ts_end'])
            entry0['ts_end'] = adjusted_ts
        
        entry0['ts_begin'] = datetime.strptime(entry0['ts_begin'], "%d-%m-%Y %H:%M:%S")
        entry0['ts_end'] = datetime.strptime(entry0['ts_end'], "%d-%m-%Y %H:%M:%S")

        res.append([entry0])
        for i in range(1, len(item)):
            # last_entry = item[i - 1].attrib
            entry = item[i].attrib
            ts_begin = entry['ts_begin']
            ts_end = entry['ts_end']
            if round == True:
                adjusted_ts_begin = round_up_to_nearest_five_minutes(ts_begin)
                entry['ts_end'] = adjusted_ts_begin
                adjusted_ts_end = round_up_to_nearest_five_minutes(ts_end)
                entry['ts_end'] = adjusted_ts_end
            entry['ts_begin'] = datetime.strptime(entry['ts_begin'], "%d-%m-%Y %H:%M:%S")
            entry['ts_end'] = datetime.strptime(entry['ts_end'], "%d-%m-%Y %H:%M:%S")
            if category == "bolus":
                if entry['ts_begin'] != entry['ts_end']:
                    print("Unequal: begin: " + str(entry['ts_begin']) + " end: " + str(entry['ts_end']))
            res.append([entry])
    return res
# Function to align and update segments with meal data
def update_segments_with_tempbasal(segments, tempbasal_df):
    for segment_name, segment_df in segments.items():
        # Initialize the 'carbs' column to zeros
        # segment_df['basal_rate'] = None

        # Iterate through each timestamp in the segment
        for i, row in segment_df.iterrows():
            # Find the closest meal timestamp and its carb information
            for _, tempbasal_row in tempbasal_df.iterrows():
                if tempbasal_row['ts_begin'] <= row['timestamp'] < tempbasal_row['ts_end']:
                    segment_df.at[i, 'basal_rate'] = tempbasal_row['value']
                    break

    return segments

def expand_bolus_entry(bolus_row):
    bolus_time = bolus_row['ts_begin']
    timestamp_list = [bolus_time, ]
    # end_effect_time = bolus_time + timedelta(hours=3)
    dose = float(bolus_row['dose'])

    b_eff_list = [dose, ]
    b_eff = dose

    i = 1
    while b_eff > 0:
        b_eff = dose - (i * 0.07)
        b_eff_list.append(b_eff)
        timestamp_list.append(bolus_time + timedelta(minutes=5 * i))
        i += 1
    # print(len(timestamp_list[:-1]))
    # print(len(b_eff_list[:-1]))


    d = {"ts": timestamp_list[:-1], "bolus_effect": b_eff_list[:-1]}
    bolus_effect_df = pd.DataFrame(data = d)

    return bolus_effect_df


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

    return segments

def compute_accumulated_step(window_list, step_df):
    start_time = window_list[0]
    end_time = window_list[-1]

    step_list = []
    counter = 1
    for idx, step_row in step_df.iterrows():
        
        if step_row['ts'] >= start_time and step_row['ts'] < end_time:
            step_list.append(counter * float(step_row['value']))
            counter += 1

        if step_row['ts'] >= end_time:
            break
    # print("length of step_list ", len(step_list))
    if len(step_list) == 0:
        return None
    accumulate_step = sum(step_list)/len(step_list)
    return accumulate_step

########################################################################
#
#                           IMPORTANT FUNCTIONS
#
########################################################################
    
def prepare_dataset(segments):
    features_list = []
    labels_list = []
    
    # Iterate over each segment
    for segment_name, segment_df in segments.items():
        # Ensure all columns are of numeric type
        segment_df['carbs'] = pd.to_numeric(segment_df['carbs'], errors='coerce')
        segment_df['basal_rate'] = pd.to_numeric(segment_df['basal_rate'], errors='coerce')
        segment_df['bolus_dose'] = pd.to_numeric(segment_df['bolus_dose'], errors='coerce')

        # Fill NaNs that might have been introduced by conversion errors
        segment_df.fillna(0, inplace=True)

        # Maximum index for creating a complete feature set
        max_index = len(segment_df) - 22  # Subtracting 22 because we need to predict index + 21 and need index + 15 to exist
        
        # Iterate through the data to create feature-label pairs
        for i in range(max_index + 1):
            # Extracting features from index i to i+15
            features = segment_df.loc[i:i+15, ['glucose_value', 'carbs', 'basal_rate', 'bolus_dose']].values.flatten()
            # Extracting label for index i+21
            # Do the label transform
            label = segment_df.loc[i+21, 'glucose_value'] - segment_df.loc[i+15, 'glucose_value']
            
            
            features_list.append(features)
            labels_list.append(label)
            
    print("len of features_list " + str(len(features_list)))
    print("len of labels_list " + str(len(labels_list)))
    labels_list = label_delta_transform(labels_list)    
    print("after label transform. the len of label list "+str(len(labels_list)))    
    return features_list, labels_list
    
    # # Convert lists to PyTorch tensors
    # features_tensor = torch.tensor(features_list, dtype=torch.float32)
    # labels_tensor = torch.tensor(labels_list, dtype=torch.float32).unsqueeze(1)  # Making labels tensor 2D
    
    # return TensorDataset(features_tensor, labels_tensor)
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

        # # Iterate through each timestamp in the segment
        # for i, row in segment_df.iterrows():
        #     # Find the closest meal timestamp and its carb information
        #     unassigned_meals = meal_df[meal_df['assigned'] == False]
        #     if not unassigned_meals.empty:
        #         unassigned_meals['time_difference'] = abs(unassigned_meals['ts'] - row['timestamp'])
        #         closest_meal = unassigned_meals.loc[unassigned_meals['time_difference'].idxmin()]

        #         if closest_meal['time_difference'] <= pd.Timedelta(minutes=5):
        #             segment_df.at[i, 'carbs'] = closest_meal['carbs']
        #             meal_df.at[closest_meal.name, 'assigned'] = True  # Mark as assigned


    return segments



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



# Optimize step information
# Process step information
def optimize_step_processing(bolus_updated_segments, step_df):
    # Convert step_df timestamps to datetime if they aren't already
    step_df['ts'] = pd.to_datetime(step_df['ts'])
    step_df['value'] = pd.to_numeric(step_df['value'])
    
    # Pre-calculate weights for step accumulation (1 to 10 for 50 minutes window)
    weights = np.arange(1, 11)
    
    for segment_name, segment_df in bolus_updated_segments.items():
        # Convert timestamps if needed
        segment_df['timestamp'] = pd.to_datetime(segment_df['timestamp'])
        
        # Create array to store accumulated steps
        accumulate_step_list = []
        
        # Get all unique window starts for this segment
        window_starts = segment_df['timestamp'].apply(lambda x: x - timedelta(minutes=50))
        window_ends = segment_df['timestamp']
        
        # Process each window
        for start, end in zip(window_starts, window_ends):
            # Filter steps within the window
            mask = (step_df['ts'] >= start) & (step_df['ts'] < end)
            window_steps = step_df.loc[mask, 'value']
            
            if len(window_steps) == 0:
                accumulate_step_list.append(None)
            else:
                # Take last 10 steps (or pad with zeros if less than 10)
                last_steps = window_steps.iloc[-10:] if len(window_steps) > 10 else window_steps
                weighted_sum = (last_steps.values * weights[:len(last_steps)]).sum()
                accumulate_step_list.append(weighted_sum / len(last_steps))
        
        # Assign accumulated steps to segment
        segment_df['steps'] = accumulate_step_list
    
    return bolus_updated_segments

# Main execution
# steps = read_ohio(filepath, "basis_steps", True)
# flattened_steps_data = [item[0] for item in steps]
# step_df = pd.DataFrame(flattened_steps_data)
# step_updated_segments = optimize_step_processing(bolus_updated_segments, step_df)

# Example DataFrame setup
# data = {
#     'timestamp': pd.to_datetime([
#         '2021-12-07 01:17:00', '2021-12-07 01:22:00', '2021-12-07 01:27:00', '2021-12-07 01:32:00', '2021-12-07 01:37:00',
#         '2021-12-07 04:00:00',  # Large gap example
#         '2021-12-07 06:00:00',  # Another large gap example
#         '2022-01-17 23:36:00', '2022-01-17 23:41:00', '2022-01-17 23:46:00', '2022-01-17 23:51:00', '2022-01-17 23:56:00'
#     ]),
#     'glucose_value': [101, 98, 104, 112, 120, 130, 135, 161, 164, 168, 172, 176]
# }
# glucose_df

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



def prepare_dataset_test_dataset( file_dir = f'../OhioT1DM/2018/test/559-ws-testing.xml', ph = 6):
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
    test_features_list, test_labels_list = prepare_dataset(test_final_updated_segments, ph)

    test_features_array = np.array(test_features_list)
    test_labels_array = np.array(test_labels_list)

    X_test, y_test = torch.tensor(test_features_array, dtype=torch.float32), torch.tensor(test_labels_array, dtype=torch.float32)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return test_loader

