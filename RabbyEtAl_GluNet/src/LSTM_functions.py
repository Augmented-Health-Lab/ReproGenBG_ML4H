from __future__ import division, print_function

import collections
import csv
import datetime
import torch
import pickle

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta
from scipy.interpolate import CubicSpline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def round_up_to_nearest_five_minutes(ts):
    """
    Round a timestamp string to the nearest 5 minutes.
    Args:
        ts (str): Timestamp string in the format "dd-mm-yyyy HH:MM:SS".
    Returns:
        str: Rounded timestamp string in the same format.
    """
    dt = datetime.strptime(ts, "%d-%m-%Y %H:%M:%S")
    
    minutes_to_add = (5 - dt.minute % 5) % 5
    if minutes_to_add == 0 and dt.second == 0:
        minutes_to_add = 0
    
    new_dt = dt + timedelta(minutes=minutes_to_add)
    return new_dt.strftime( "%d-%m-%Y %H:%M:%S")

def read_ohio(filepath, category, round):
    """
    Function to read the ohio XML file and extract data for the specific categories
    
    Args: 
        filepath (str): Path to the XML file.
        category (str): The category of data to extract (e.g., "glucose_level", "meal").
        round (bool): Whether to round the timestamps to the nearest 5 minutes.
    
    Returns:
        res (list): A list of lists containing the extracted data for the specified category.
    """
    tree = ET.parse(filepath)
    root = tree.getroot()
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
            entry = item[i].attrib
            ts = entry['ts']
            if round == True:
                adjusted_ts = round_up_to_nearest_five_minutes(ts)
                entry['ts'] = adjusted_ts
            ts = entry['ts']
            entry['ts'] = datetime.strptime(ts, "%d-%m-%Y %H:%M:%S")
            res.append([entry])
    return res

def transfer_into_table(glucose):
    """
    Function to convert the glucose data into a pandas DataFrame.
    
    Args:
        glucose (list): List of lists containing glucose data.
    
    Returns: 
        glucose_df (DataFrame): A pandas DataFrame containing the glucose data with timestamps and values.
    """
    glucose_dict = {entry[0]['ts']: entry[0]['value'] for entry in glucose}
    g_data = []
    for timestamp in glucose_dict:
        record = {
            'timestamp': timestamp,
            'glucose_value': glucose_dict[timestamp],
        }
        
        g_data.append(record)

    glucose_df = pd.DataFrame(g_data)
    glucose_df['glucose_value'] = pd.to_numeric(glucose_df['glucose_value'])
    glucose_df['glucose_value'] = glucose_df['glucose_value'] 

    return glucose_df


def segement_data_as_6_min(data):
    """
    Function to segment the glucose data into segments based on time gaps greater than 6 minutes.
    
    Args:
        data (DataFrame): A pandas DataFrame containing glucose data with timestamps and values.
    
    Returns:
        segments (dict): A dictionary where each key is a segment name and the value is a DataFrame for that segment.
    """
    df = pd.DataFrame(data)

    # Calculate time differences
    df['time_diff'] = df['timestamp'].diff()

    # Identify large gaps
    df['new_segment'] = df['time_diff'] > pd.Timedelta(minutes=6)

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


##############################################################################
#
#                         UNUSED FUNCTIONS BELOW
#
##############################################################################

# +
# + The below functions were not necessarily used in the main processing pipeline but are provided for completeness.
# +



def find_closest_glucose_index(glucose_df, meal_time, threshold_seconds=300):
    """
    Find the index of the closest glucose reading to a given meal time within a specified threshold.
    
    Args:
        glucose_df (DataFrame): A pandas DataFrame containing glucose data with timestamps.
        meal_time (datetime): The timestamp of the meal.
        threshold_seconds (int): The maximum time difference in seconds to consider a glucose reading as "close".
    
    Returns:
        closest_index (int or None): The index of the closest glucose reading if found, otherwise None.
    """
    time_diffs = (glucose_df['timestamp'] - meal_time).abs()
    within_threshold = time_diffs < pd.Timedelta(seconds=threshold_seconds)
    if within_threshold.any():
        closest_index = time_diffs[within_threshold].idxmin()
        return closest_index
    return None

def update_segments_with_meals(segments, meal_df):
    """
    Update each segment DataFrame with the meal effect based on the closest glucose reading.
    
    Args: 
        segments (dict): A dictionary where each key is a segment name and the value is a DataFrame for that segment.
        meal_df (DataFrame): A pandas DataFrame containing meal data with timestamps and carb effects.
    
    Returns:
        segments (dict): The updated segments with the meal effect applied.
    """
    
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
    """
    Expand the meal entry to calculate the carb effect over time.
    
    Args:
        meal_row (Series): A pandas Series containing the meal data with timestamp and carb value.
    
    Returns:
        meal_effect_df (DataFrame): A pandas DataFrame containing the timestamp and corresponding carb effect over time.
    """
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
    """
    Function to read the ohio XML file and extract bolus data for the specific categories
    Args:
        filepath (str): Path to the XML file.
        category (str): The category of data to extract (e.g., "bolus").
        round (bool): Whether to round the timestamps to the nearest 5 minutes.
    Returns:
        res (list): A list of lists containing the extracted bolus data for the specified category.
    """
    
    tree = ET.parse(filepath)
    root = tree.getroot()

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


def expand_bolus_entry(bolus_row):
    """
    Expand the bolus entry to calculate the bolus effect over time.
    
    Args:
        bolus_row (Series): A pandas Series containing the bolus data with timestamp and dose value.
        
    Returns:
        bolus_effect_df (DataFrame): A pandas DataFrame containing the timestamp and corresponding bolus effect over time.
    """
    
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

    d = {"ts": timestamp_list[:-1], "bolus_effect": b_eff_list[:-1]}
    bolus_effect_df = pd.DataFrame(data = d)

    return bolus_effect_df


def update_segments_with_bolus(segments, bolus_df):
    """
    Update each segment DataFrame with the bolus effect based on the closest glucose reading.
    Args:
        segments (dict): A dictionary where each key is a segment name and the value is a DataFrame for that segment.
        bolus_df (DataFrame): A pandas DataFrame containing bolus data with timestamps and bolus effects.
    
    Returns:
        segments (dict): The updated segments with the bolus effect applied.
    """
    for segment_name, segment_df in segments.items():
        # Initialize the 'carbs' column to zeros
        segment_df['bolus_effect'] = 0

        for index, bolus_row in bolus_df.iterrows():
            bolus_time = bolus_row['ts']
            closest_glucose_idx = find_closest_glucose_index(segment_df, bolus_time)
            
            if closest_glucose_idx is not None:
                segment_df.loc[closest_glucose_idx, 'bolus_effect'] = float(bolus_row['bolus_effect'])
                bolus_df.loc[index, 'assigned'] = True
    return segments


def compute_accumulated_step(window_list, step_df):
    """
    Function to compute the accumulated step count over a specified time window.
    
    Args:
        window_list (list): A list containing the start and end timestamps of the window.
        step_df (DataFrame): A pandas DataFrame containing step data with timestamps and values.
    
    Returns:
        float: The average accumulated step count over the specified window, or None if no steps are found.
    """
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

def add_meal_segments(filepath):
    """
    Function to read the Ohio XML file and extract meal data, then update glucose segments with meal effects.
    
    Args:
        filepath (str): Path to the Ohio XML file.
        
    Returns:
        meal_updated_segments (dict): A dictionary where each key is a segment name and the value is a DataFrame for that segment, updated with meal effects.
    """
    glucose = read_ohio(filepath, "glucose_level", True)
    glucose_df = transfer_into_table(glucose)
    segments = segement_data_as_15min(glucose_df)

    #Include meal:
    meal = read_ohio(filepath, "meal", True)
    flattened_meal_data = [item[0] for item in meal]  # Take the first (and only) item from each sublist
    # Convert to DataFrame
    meal_df = pd.DataFrame(flattened_meal_data)
    meal_df['assigned'] = False
    empty_d = {"ts": [], "carb_effect": []}
    whole_meal_effect_df = pd.DataFrame(data = empty_d)
    for index, meal_row in meal_df.iterrows():
        meal_effect_df = expand_meal_entry(meal_row)

        # Merge the DataFrames on the 'ts' column with an outer join
        merged_df = pd.merge(whole_meal_effect_df, meal_effect_df, on='ts', how='outer', suffixes=('_df1', '_df2'))

        # Fill NaN values with 0 for the carb_effect columns
        merged_df['carb_effect_df1'] = merged_df['carb_effect_df1'].fillna(0)
        merged_df['carb_effect_df2'] = merged_df['carb_effect_df2'].fillna(0)

        # Sum the carb_effect values
        merged_df['carb_effect'] = merged_df['carb_effect_df1'] + merged_df['carb_effect_df2']

        # Keep only the required columns
        whole_meal_effect_df = merged_df[['ts', 'carb_effect']]

    whole_meal_effect_df['assigned'] = False
    meal_updated_segments = update_segments_with_meals(segments, whole_meal_effect_df)
    return meal_updated_segments


def add_bolus_segments(filepath, meal_updated_segments):
    """
    Function to read the Ohio XML file and extract bolus data, then update meal segments with bolus effects.
    
    Args:
        filepath (str): Path to the Ohio XML file.
        meal_updated_segments (dict): A dictionary where each key is a segment name and the value is a DataFrame for that segment, updated with meal effects.
    
    Returns:
        bolus_updated_segments (dict): A dictionary where each key is a segment name and the value is a DataFrame for that segment, updated with both meal and bolus effects.
    """
    bolus = read_ohio_bolus_tempbasal(filepath, "bolus", True)
    flattened_bolus_data = [item[0] for item in bolus]  # Take the first (and only) item from each sublist
    # Convert to DataFrame
    bolus_df = pd.DataFrame(flattened_bolus_data)

    empty_b = {"ts": [], "bolus_effect": []}
    whole_bolus_effect_df = pd.DataFrame(data = empty_b)

    for index, bolus_row in bolus_df.iterrows():
        bolus_effect_df = expand_bolus_entry(bolus_row)

        # Merge the DataFrames on the 'ts' column with an outer join
        merged_df = pd.merge(whole_bolus_effect_df, bolus_effect_df, on='ts', how='outer', suffixes=('_df1', '_df2'))

        # Fill NaN values with 0 for the carb_effect columns
        merged_df['bolus_effect_df1'] = merged_df['bolus_effect_df1'].fillna(0)
        merged_df['bolus_effect_df2'] = merged_df['bolus_effect_df2'].fillna(0)
        

        # Sum the carb_effect values
        merged_df['bolus_effect'] = merged_df['bolus_effect_df1'] + merged_df['bolus_effect_df2']

        # Keep only the required columns
        whole_bolus_effect_df = merged_df[['ts', 'bolus_effect']]

    whole_bolus_effect_df["assigned"] = False
    bolus_updated_segments = update_segments_with_bolus(meal_updated_segments, whole_bolus_effect_df)
    
    return bolus_updated_segments


# Process step information
def optimize_step_processing(bolus_updated_segments, step_df):
    
    """
    Function to optimize the processing of step data by accumulating steps over 50 minutes.
    
    Args:
        bolus_updated_segments (dict): A dictionary where each key is a segment name and the value is a DataFrame for that segment, updated with bolus effects.
        step_df (DataFrame): A pandas DataFrame containing step data with timestamps and values.
    
    Returns:
        bolus_updated_segments (dict): The updated segments with accumulated step counts added.
    """
    
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