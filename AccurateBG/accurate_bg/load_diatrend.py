import argparse
import json
import os
import pandas as pd
import os

import datetime
import numpy as np
from cgms_data_seg_diatrend import CGMSDataSeg
from cnn_ohio import regressor, regressor_transfer, test_ckpt
from data_reader_DiaTrend import DataReader

def preprocess_DiaTrend(path):

    subject = pd.read_csv(path)
    subject['date'] = pd.to_datetime(subject['date'], errors='coerce')  # Convert 'date' column to datetime if not already
    # print(subject['date'][0])
    subject.sort_values('date', inplace=True)  # Sort the DataFrame by the 'date' column

    # Assuming self.interval_timedelta is set, for example:
    interval_timedelta = datetime.timedelta(minutes=6)  # Example timedelta of 6 minutes, providing a range for latency

    # Create a list to store the results
    res = []

    # Initialize the first group
    if not subject.empty:
        current_group = [subject.iloc[0]['mg/dl']]
        last_time = subject.iloc[0]['date']

    # Iterate over rows in DataFrame starting from the second row
    for index, row in subject.iloc[1:].iterrows():
        current_time = row['date']
        if (current_time - last_time) <= interval_timedelta:
            # If the time difference is within the limit, add to the current group
            current_group.append(row['mg/dl'])
        else:
            # Otherwise, start a new group
            res.append(current_group)
            current_group = [row['mg/dl']]
        last_time = current_time

    # Add the last group if it's not empty
    if current_group:
        res.append(current_group)
    
    # Filter out groups with fewer than 10 glucose readings
    # res = [group for group in res if len(group) >= 10]

    return res


def main():
    epoch = 10
    ph = 12
    path = "../diatrend_results"
    # Define the directory path
    train_directory_path = f'../../../dataset/DiaTrend/train'  # Use a raw string for paths on Windows

    # List files without their extensions
    train_file_names = [os.path.splitext(file)[0] for file in os.listdir(train_directory_path)
                if os.path.isfile(os.path.join(train_directory_path, file))]
    
    # Define the directory path
    test_directory_path = f'../../../dataset/DiaTrend/test'  # Use a raw string for paths on Windows

    # List files without their extensions
    test_file_names = [os.path.splitext(file)[0] for file in os.listdir(test_directory_path)
                if os.path.isfile(os.path.join(test_directory_path, file))]
    
    cleaned_subjects = [s.replace("_training_data", "") for s in train_file_names]
    cleaned_subjects.sort()
    
    
    train_data = dict()
    for subj in train_file_names:
        subj_path = f'C:/Users/username/OneDrive/Desktop/BGprediction/DiaTrend/train/{subj}.csv'
        reader = preprocess_DiaTrend(subj_path)
        train_data[subj] = reader

    test_data = dict()
    for subj in test_file_names:
        subj_path = f'C:/Users/username/OneDrive/Desktop/BGprediction/DiaTrend/test/{subj}.csv'
        reader = preprocess_DiaTrend(subj_path)
        test_data[subj] = reader

    # a dumb dataset instance
    train_dataset = CGMSDataSeg(
        "ohio", "C:/Users/username/OneDrive/Desktop/BGprediction/DiaTrend/train/Subject10_training_data.csv", 6
    )
    sampling_horizon = 7
    prediction_horizon = ph
    scale = 0.01
    outtype = "Same"
    # train on training dataset
    # k_size, nblock, nn_size, nn_layer, learning_rate, batch_size, epoch, beta
    with open(f'../diatrend_results/config.json') as json_file:
        config = json.load(json_file)
    argv = (
        config["k_size"],
        config["nblock"],
        config["nn_size"],
        config["nn_layer"],
        config["learning_rate"],
        config["batch_size"],
        epoch,
        config["beta"],
    )
    l_type = config["loss"]
    # test on patients data
    outdir = os.path.join(path, f"ph_{prediction_horizon}_{l_type}")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    all_errs = []

    # Loop
    standard = False  # do not use standard
    all_errs = []
    for pid in cleaned_subjects[:9]:
        local_train_data = []
        
        for k in train_data.keys():
            if k != pid:
                local_train_data += train_data[k]

        train_dataset.data = local_train_data
        train_dataset.set_cutpoint = -1
        train_dataset.reset(
            sampling_horizon,
            prediction_horizon,
            scale,
            100,
            False,
            outtype,
            1,
            standard,
        )
        regressor(train_dataset, *argv, l_type, outdir)
        # Fine-tune and test
        # target_test_dataset = CGMSDataSeg(
        #     "ohio", f"C:/Users/username/OneDrive/Desktop/BGprediction/OhioT1DM/2018/test/{pid}-ws-testing.xml", 5
        # )
        target_test_dataset = CGMSDataSeg(
        "ohio", f"C:/Users/username/OneDrive/Desktop/BGprediction/DiaTrend/test/{pid}_testing_data.csv", 6
        )
        target_test_dataset.set_cutpoint = 1
        target_test_dataset.reset(
            sampling_horizon,
            prediction_horizon,
            scale,
            0.01,
            False,
            outtype,
            1,
            standard,
        )
        # target_train_dataset = CGMSDataSeg(
        #     "ohio", f"C:/Users/username/OneDrive/Desktop/BGprediction/OhioT1DM/2018/test/{pid}-ws-testing.xml", 5
        # )
        target_train_dataset = CGMSDataSeg(
        "ohio", f"C:/Users/username/OneDrive/Desktop/BGprediction/DiaTrend/train/{pid}_training_data.csv", 6
        )
        target_train_dataset.set_cutpoint = -1
        target_train_dataset.reset(
            sampling_horizon,
            prediction_horizon,
            scale,
            100,
            False,
            outtype,
            1,
            standard,
        )
        err, labels = test_ckpt(target_test_dataset, outdir)
        errs = [err]
        transfer_res = [labels]
        for i in range(1, 2):
            err, labels = regressor_transfer(
                target_train_dataset,
                target_test_dataset,
                config["batch_size"],
                epoch,
                outdir,
                i,
            )
            errs.append(err)
            transfer_res.append(labels)
        transfer_res = np.concatenate(transfer_res, axis=1)
        print(transfer_res)
        np.savetxt(
            f"{outdir}/{pid}.txt",
            transfer_res,
            fmt="%.4f %.4f %.4f %.4f", #%.4f %.4f %.4f %.4f
        )
        all_errs.append([pid] + errs)
    all_errs = np.array(all_errs)
    np.savetxt(f"{outdir}/errors.txt", all_errs, fmt="%d %.4f %.4f") # %.4f %.4f
    # label pair:(groundTruth, y_pred)


if __name__ == "__main__":
    main()
