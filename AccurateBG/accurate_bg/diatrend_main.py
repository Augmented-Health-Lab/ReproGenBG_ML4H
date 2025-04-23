import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from cgms_data_seg import CGMSDataSeg  # Assuming CGMSDataSeg is defined in cgms_data_seg.py
from regressor import regressor  # Assuming regressor is defined in regressor.py
from test_ckpt import test_ckpt  # Assuming test_ckpt is defined in test_ckpt.py


def preprocess_DiaTrend(path):
    """
    Preprocess the DiaTrend dataset by grouping glucose readings based on time intervals.

    Parameters
    ----------
    path : str
        Path to the CSV file containing glucose readings.

    Returns
    -------
    list
        A list of grouped glucose readings.
    """
    subject = pd.read_csv(path)
    subject['date'] = pd.to_datetime(subject['date'], errors='coerce')  # Convert 'date' column to datetime
    subject.sort_values('date', inplace=True)  # Sort the DataFrame by the 'date' column

    interval_timedelta = timedelta(minutes=6)  # Example timedelta of 6 minutes

    res = []
    if not subject.empty:
        current_group = [subject.iloc[0]['mg/dl']]
        last_time = subject.iloc[0]['date']

    for _, row in subject.iloc[1:].iterrows():
        current_time = row['date']
        if (current_time - last_time) <= interval_timedelta:
            current_group.append(row['mg/dl'])
        else:
            res.append(current_group)
            current_group = [row['mg/dl']]
        last_time = current_time

    if current_group:
        res.append(current_group)

    return res


def main():
    """
    Main function to execute the DiaTrend dataset processing, training, and evaluation.
    """
    epoch = 80
    ph = 6
    path = "../diatrend_results"
    sh = 6

    for sh in [6, 12, 18, 24]:
        for fold_num in range(1, 6):
            # Load training data
            train_directory_path = f'C:/Users/baiyi/OneDrive/Desktop/Modify_GenBG/modified_diatrend_subset/fold{fold_num}_training'
            train_file_names = [os.path.splitext(file)[0] for file in os.listdir(train_directory_path)
                                if os.path.isfile(os.path.join(train_directory_path, file))]
            cleaned_subjects = [s.replace("processed_cgm_data_", "") for s in train_file_names]

            # Load test data
            test_directory_path = f'C:/Users/baiyi/OneDrive/Desktop/Modify_GenBG/modified_diatrend_subset/fold{fold_num}_test'
            test_file_names = [os.path.splitext(file)[0] for file in os.listdir(test_directory_path)
                               if os.path.isfile(os.path.join(test_directory_path, file))]
            cleaned_test_subjects = [s.replace("processed_cgm_data_", "") for s in test_file_names]

            # Preprocess training data
            train_data = {}
            for subj in train_file_names:
                subj_path = f'C:/Users/baiyi/OneDrive/Desktop/Modify_GenBG/modified_diatrend_subset/fold{fold_num}_training/{subj}.csv'
                train_data[subj] = preprocess_DiaTrend(subj_path)

            # Preprocess test data
            test_data = {}
            for subj in test_file_names:
                subj_path = f'C:/Users/baiyi/OneDrive/Desktop/Modify_GenBG/modified_diatrend_subset/fold{fold_num}_test/{subj}.csv'
                test_data[subj] = preprocess_DiaTrend(subj_path)

            # Initialize dataset
            train_dataset = CGMSDataSeg(
                "diatrend", "C:/Users/baiyi/OneDrive/Desktop/Modify_GenBG/modified_diatrend_subset/fold1_training/processed_cgm_data_Subject12.csv", 5
            )
            sampling_horizon = sh
            prediction_horizon = ph
            scale = 0.01
            outtype = "Same"

            # Load configuration
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

            # Create output directory
            outdir = os.path.join(path, f"ph_{prediction_horizon}_sh{sampling_horizon}_fold{fold_num}_{l_type}")
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            # Train on the training fold
            cleaned_subjects.sort()
            standard = False
            train_pids = set(cleaned_subjects)
            local_train_data = []
            for k in train_pids:
                local_train_data += train_data["processed_cgm_data_" + k]

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
            for pid in cleaned_test_subjects:
                target_test_dataset = CGMSDataSeg(
                    "diatrend", f"C:/Users/baiyi/OneDrive/Desktop/Modify_GenBG/modified_diatrend_subset/fold{fold_num}_test/processed_cgm_data_{pid}.csv", 5
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