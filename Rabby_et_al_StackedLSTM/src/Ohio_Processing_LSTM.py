from __future__ import division, print_function

import collections
import csv
import datetime
import torch
import pickle
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

from datetime import datetime, timedelta
from scipy.interpolate import CubicSpline
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from LSTM_functions import *


# Check if GPU is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_all_training_files(directory_path):
    """
    Processes all the training XML files in the specified directory.
    
    Args:
        directory_path (str): Path to the directory containing the XML files.
        
    Returns:
        all_processed_data: A list of dictionaries containing processed data for each file.
    """
    xml_files = glob.glob(os.path.join(directory_path, "*-ws-training.xml"))
    
    all_processed_data = []
    
    for filepath in xml_files:
        try:
            # Process each file
            glucose = read_ohio(filepath, "glucose_level", True)
            glucose_df = transfer_into_table(glucose)
            segments = segement_data_as_6_min(glucose_df)
            
            # # Uncomment the following lines if you want to process additional data like insulin, meal, and steps
            # meal = add_meal_segments(filepath)
            # bolus = add_bolus_segments(filepath, meal)

            # steps = read_ohio(filepath, "basis_steps", True)
            # flattened_steps_data = [item[0] for item in steps]
            # step_df = pd.DataFrame(flattened_steps_data)
            # step_updated_segments = optimize_step_processing(bolus, step_df)
            
            # Add to list of processed data
            all_processed_data.append({
                'filepath': filepath,
                'segments': segments,
            })
            
            print(f"Successfully processed {filepath}")
            
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")
    
    return all_processed_data

def main(): 
    """
    This is the main function that processes the training and test data. It should be run
    before running Ohio_Training_LSTM.py.
    """
    ##############################################################################
    #
    #                         PROCESSING  TRAINING DATA
    #
    ##############################################################################

    directory_path = "../../../data/OhioT1DM/2018/train/"
    training_data = process_all_training_files(directory_path)


    # Have all data for all subjects in one dictionary
    # Add a counter to ensure each segment key is unique

    segment_dict = {}
    count = 0

    segment_name_list = []
    segment_data_list= []
    for i in training_data: 
        count += 1
        for j in i['segments']:
            segment_dict[str(count)+j] = i['segments'][j]

    # As we only use CGM, the suffix onlyCGM is placed
    filename = './processed_data/BIG_training_onlyCGM.pkl'

    # Save the dictionary to a file
    if not os.path.exists(filename):
        open(filename, 'wb').close()
        
    with open(filename, 'wb') as f:
        pickle.dump(segment_dict, f)


    ##############################################################################
    #
    #                         PROCESSING TEST DATA
    #
    ##############################################################################

    test_dir = '../OhioT1DM/2018/test/*'

    for test_file in glob.glob(test_dir): 
        print(f'Processing {test_file}')
        glucose = read_ohio(test_file, "glucose_level", True)
        glucose_df = transfer_into_table(glucose)
        segments = segement_data_as_6_min(glucose_df)
        filename = f'{os.path.basename(test_file)}_test_noshrink.pkl'
        pickle.dump(segments, open(filename, 'wb'))


if __name__ == "__main__":
    main()