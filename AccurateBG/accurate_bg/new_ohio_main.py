import os
import json
import numpy as np
from data_reader import DataReader  # Assuming DataReader is defined in data_reader.py
from cgms_data_seg import CGMSDataSeg  # Assuming CGMSDataSeg is defined in cgms_data_seg.py
from cnn_ohio import regressor, regressor_transfer, test_ckpt


def main():
    """
    Main function to execute the training and evaluation process for the dataset.
    """
    # Fixed sampling horizon
    sampling_horizon = 6
    epoch = 80

    # Patient IDs for 2018 and 2020
    pid_2018 = [559, 563, 570, 588, 575, 591]
    pid_2020 = [540, 552, 544, 567, 584, 596]
    pid_year = {2018: pid_2018, 2020: pid_2020}

    # Load training data
    train_data = dict()
    for year in pid_year.keys():
        pids = pid_year[year]
        for pid in pids:
            reader = DataReader(
                "ohio", f"C:/Users/baiyi/OneDrive/Desktop/Modify_GenBG/OhioT1DM 2020/{year}/train/{pid}-ws-training.xml", 5
            )
            train_data[pid] = reader.read()

    # Load test data for 2018 patients
    use_2018_test = False
    standard = False
    test_data_2018 = []
    for pid in pid_2018:
        reader = DataReader(
            "ohio", f"C:/Users/baiyi/OneDrive/Desktop/Modify_GenBG/OhioT1DM 2020/2018/test/{pid}-ws-testing.xml", 5
        )
        test_data_2018 += reader.read()

    # Initialize a dummy dataset instance
    train_dataset = CGMSDataSeg(
        "ohio", "C:/Users/baiyi/OneDrive/Desktop/Modify_GenBG/OhioT1DM 2020/2018/train/559-ws-training.xml", 5
    )

    # Load configuration
    path = "../ohio_results" # Replace with the actual path to your config directory
    with open(os.path.join(path, "config.json")) as json_file:
        config = json.load(json_file)

    prediction_horizon = config["prediction_horizon"]
    scale = 0.01
    outtype = "Same"
    epoch = config["epoch"]
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

    # Output directory
    outdir = os.path.join(path, f"ph_{prediction_horizon}_sh{sampling_horizon}_{l_type}")
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Train and evaluate for each patient
    all_errs = []
    for year in pid_year.keys():
        pids = pid_year[year]
        for pid in pids:
            train_pids = set(pid_2018 + pid_2020) - {pid}
            local_train_data = []
            if use_2018_test:
                local_train_data += test_data_2018
            for k in train_pids:
                local_train_data += train_data[k]

            print(f"Pretrain data: {sum([sum(x) for x in local_train_data])}")
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

            # Fine-tune on personal data
            target_test_dataset = CGMSDataSeg(
                "ohio", f"C:/Users/baiyi/OneDrive/Desktop/Modify_GenBG/OhioT1DM 2020/{year}/test/{pid}-ws-testing.xml", 5
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
            target_train_dataset = CGMSDataSeg(
                "ohio", f"C:/Users/baiyi/OneDrive/Desktop/Modify_GenBG/OhioT1DM 2020/{year}/train/{pid}-ws-training.xml", 5
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

            # Evaluate and transfer learning
            err, labels = test_ckpt(target_test_dataset, outdir)
            errs = [err]
            transfer_res = [labels]
            for i in range(1, 4):
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
            np.savetxt(
                f"{outdir}/{pid}.txt",
                transfer_res,
                fmt="%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f",
            )
            all_errs.append([pid] + errs)

    # Save all errors
    all_errs = np.array(all_errs)
    np.savetxt(f"{outdir}/errors.txt", all_errs, fmt="%d %.4f %.4f %.4f %.4f")

if __name__ == "__main__":
    main()