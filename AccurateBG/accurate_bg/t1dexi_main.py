import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from cgms_data_seg import CGMSDataSeg  # Assuming CGMSDataSeg is defined in cgms_data_seg.py
from regressor import regressor  # Assuming regressor is defined in regressor.py
from test_ckpt import test_ckpt  # Assuming test_ckpt is defined in test_ckpt.py


def read_preprocess_cgm_training(subj):
    """
    Read and preprocess CGM training data.

    Parameters
    ----------
    subj : str
        Path to the subject's CGM data file.

    Returns
    -------
    pd.DataFrame
        A concise DataFrame with glucose readings and timestamps.
    """
    subject_cgm = pd.read_csv(subj)
    subject_cgm['LBDTC'] = pd.to_datetime(subject_cgm['LBDTC'])
    subject_cgm['date'] = subject_cgm['LBDTC'].dt.date
    subject_cgm_concise = subject_cgm[["LBORRES", "LBDTC"]]
    return subject_cgm_concise


def preprocess_T1DEXI(subject_id):
    """
    Preprocess the T1DEXI dataset by grouping glucose readings based on time intervals.

    Parameters
    ----------
    subject_id : str
        Path to the subject's CGM data file.

    Returns
    -------
    list
        A list of grouped glucose readings.
    """
    subject_cgm = read_preprocess_cgm_training(subject_id)
    interval_timedelta = timedelta(minutes=6)

    res = []
    if not subject_cgm.empty:
        current_group = [subject_cgm.iloc[0]['LBORRES']]
        last_time = subject_cgm.iloc[0]['LBDTC']

    for _, row in subject_cgm.iloc[1:].iterrows():
        current_time = row['LBDTC']
        if (current_time - last_time) <= interval_timedelta:
            current_group.append(row['LBORRES'])
        else:
            res.append(current_group)
            current_group = [row['LBORRES']]
        last_time = current_time

    if current_group:
        res.append(current_group)

    return res


def main():
    """
    Main function to execute the T1DEXI dataset processing, training, and evaluation.
    """
    epoch = 80
    ph = 6
    path = "../t1dexi_results"

    for sh in [6, 12, 18, 24]:
        for fold_num in range(1, 6):
            # Load training data
            train_directory_path = f'C:/Users/baiyi/OneDrive/Desktop/Modify_GenBG/modified_t1dexi_subset/T1DEXI_cgm_processed/fold{fold_num}_training'
            train_file_names = [os.path.splitext(file)[0] for file in os.listdir(train_directory_path)
                                if os.path.isfile(os.path.join(train_directory_path, file))]

            # Load test data
            test_directory_path = f'C:/Users/baiyi/OneDrive/Desktop/Modify_GenBG/modified_t1dexi_subset/T1DEXI_cgm_processed/fold{fold_num}_test'
            test_file_names = [os.path.splitext(file)[0] for file in os.listdir(test_directory_path)
                               if os.path.isfile(os.path.join(test_directory_path, file))]

            # Preprocess training data
            train_data = {}
            for subj in train_file_names:
                subj_path = f'{train_directory_path}/{subj}.csv'
                train_data[subj] = preprocess_T1DEXI(subj_path)

            # Preprocess test data
            test_data = {}
            for subj in test_file_names:
                subj_path = f'{test_directory_path}/{subj}.csv'
                test_data[subj] = preprocess_T1DEXI(subj_path)

            # Initialize dataset
            train_dataset = CGMSDataSeg(
                "t1dexi", f"{train_directory_path}/252.csv", 5
            )
            sampling_horizon = sh
            prediction_horizon = ph
            scale = 0.01
            outtype = "Same"

            # Load configuration
            with open(f'../t1dexi_results/config.json') as json_file:
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

            # Create output directory
            outdir = os.path.join(path, f"ph_{prediction_horizon}_sh{sampling_horizon}_fold{fold_num}_{l_type}")
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            # Train on the training fold
            train_file_names.sort()
            standard = False
            train_pids = set(train_file_names)
            local_train_data = []
            for k in train_pids:
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

            # Evaluate on the test patients
            all_errs = []
            for pid in test_file_names:
                target_test_dataset = CGMSDataSeg(
                    "t1dexi", f"{test_directory_path}/{pid}.csv", 5
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

                err, labels = test_ckpt(target_test_dataset, outdir)
                np.savetxt(
                    f"{outdir}/{pid}.txt",
                    [err],
                    fmt="%.4f",
                )
                all_errs.append([str(pid), err])

            all_errs = np.array(all_errs, dtype=object)
            np.savetxt(f"{outdir}/errors.txt", all_errs, fmt="%s %.4f")


if __name__ == "__main__":
    main()