# Stacked LSTM for Blood Glucose Prediction

This repository contains the implementation of a Stacked LSTM model for blood glucose prediction using multiple datasets: T1DEXI, OhioT1DM, and Diatrend.

## Directory Structure

```
Stacked LSTM/
├── README.md                # This file with project documentation
├── outputs/                 # Model outputs and results directory
│   ├── Diatrend/            # Results from Diatrend dataset experiments
│   │   ├── models/          # Saved model weights for Diatrend
│   │   └── outputs/         # Prediction results and evaluation metrics
│   ├── OhioT1DM/            # Results from OhioT1DM dataset experiments
│   │   ├── data/            # Processed data specific to OhioT1DM
│   │   ├── models/          # Saved model weights for OhioT1DM
│   │   └── outputs/         # Prediction results and evaluation metrics
│   └── T1DEXI/              # Results from T1DEXI dataset experiments
│       ├── models/          # Saved model weights for T1DEXI
│       └── outputs/         # Prediction results and evaluation metrics
├── processed_data/          # Preprocessed datasets directory
│   └── BIG_training_onlyCGM.pkl  # Processed training data with only CGM values
└── src/                     # Source code directory
    ├── diatrend_job.sh      # Job script for running Diatrend experiments on HPC
    ├── Diatrend_LSTM.py     # Implementation for Diatrend dataset processing and model
    ├── LSTM_functions.py    # Common utility functions for all LSTM models
    ├── Ohio_Processing_LSTM.py  # Data processing script for Ohio dataset
    ├── Ohio_Training_LSTM.py    # Training script for Ohio dataset models
    ├── t1dexi_job.sh        # Job script for running T1DEXI experiments on HPC
    ├── T1DEXI_LSTM.py       # Implementation for T1DEXI dataset processing and model
    ├── testing_Diatrend.ipynb   # Notebook for testing and visualizing Diatrend models
    ├── testing_OhioT1DM.ipynb   # Notebook for testing and visualizing Ohio models
    └── testing_T1DEXI.ipynb     # Notebook for testing and visualizing T1DEXI models
```

## Model Overview

This project implements a Stacked LSTM (Long Short-Term Memory) neural network architecture for predicting future blood glucose values in diabetic patients. As referenced in Rabby et Al.The model uses historical continuous glucose monitoring (CGM) data to forecast glucose levels over specific prediction horizons.

link: https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-021-01462-5

Please cite this work if you find it helpful: 

```
Rabby MF, Tu Y, Hossen MI, Lee I, Maida AS, Hei X. Stacked LSTM based deep recurrent neural network with kalman smoothing for blood glucose prediction. BMC Med Inform Decis Mak. 2021 Mar 16;21(1):101. doi: 10.1186/s12911-021-01462-5. PMID: 33726723; PMCID: PMC7968367.
```

## Datasets

The model is trained and evaluated on three distinct datasets:

1. **T1DEXI**: A dataset containing continuous glucose monitoring data from Type 1 diabetes patients.
2. **OhioT1DM**: A dataset with CGM readings, insulin dosing, meal information, and physical activity data.
3. **Diatrend**: A dataset with glucose measurements from a different monitoring system.

## Getting Started

To use this project:

1. Process the raw data using the corresponding processing scripts
2. Train models using the training scripts for each dataset
3. Evaluate and visualize results using the testing notebooks