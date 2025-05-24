# Stacked LSTM for Blood Glucose Prediction

This repository contains the implementation of a Stacked LSTM model for blood glucose prediction using multiple datasets: T1DEXI, OhioT1DM, and Diatrend.

## Directory Structure
```
Li_et_al_GluNet/                  # Root directory for the GluNet blood glucose prediction project
│
├── README.md                     # Project documentation and overview
│
├── __pycache__/                  # Python compiled bytecode cache
│   ├── GlucNet_functions.cpython-311.pyc
│   ├── GlucNet_functions.cpython-312.pyc
│   └── GlucNet_functions.cpython-39.pyc
│
├── data/                         # Directory for training datasets
│   ├── BIG_training_data.pkl     # Complete training data (likely includes CGM and other features)
│   └── BIG_training_onlyCGM.pkl  # Training data with only CGM (Continuous Glucose Monitoring) values
│
├── outputs/                      # Directory to store model outputs and results
│   ├── Diatrend/                 # Results for Diatrend dataset experiments
│   ├── OhioT1DM/                 # Results for Ohio T1DM dataset experiments
│   └── T1DEXI/                   # Results for T1DEXI dataset experiments
│
└── src/                          # Source code directory
    ├── diatrend_GluNet.ipynb     # Jupyter notebook for Diatrend dataset experiments
    ├── diatrend_job_2.sh         # Batch job script for Diatrend experiments (alternative version)
    ├── diatrend_job.sh           # Batch job script for Diatrend experiments on HPC/cluster
    ├── GlucNet_functions.py      # Core functions used across different datasets
    ├── GluNet_Again.ipynb        # Additional experimentation notebook
    ├── Glunet_Diatrend_Tester.py # Testing script for Diatrend dataset
    ├── Glunet_Diatrend.py        # Implementation for Diatrend dataset
    ├── GluNet_Ohio.py            # Implementation for Ohio T1DM dataset
    ├── GluNet_Processing.ipynb   # Notebook for data preprocessing
    ├── GluNet_T1DEXI.py          # Implementation for T1DEXI dataset
    ├── ohio_job.sh               # Batch job script for Ohio dataset experiments
    ├── t1dexi_GluNet copy.ipynb  # Copy of the T1DEXI experiment notebook
    └── t1dexi_GluNet.ipynb       # Jupyter notebook for T1DEXI dataset experiments
```

## Model Overview
The GluNet model is a deep learning architecture designed specifically for blood glucose prediction in diabetes management. It utilizes a stacked LSTM (Long Short-Term Memory) network to capture temporal dependencies in glucose time series data. The model architecture includes:

### Key Features
- Time-series analysis of continuous glucose monitoring (CGM) data
- Ability to incorporate multiple input features beyond glucose measurements
- Prediction horizons from 30 minutes to several hours
- Trained on diverse datasets to ensure generalizability

### Performance Metrics
- Evaluated using Root Mean Square Error (RMSE)
- Tested on three different datasets: T1DEXI, OhioT1DM, and Diatrend
- Comparable or superior performance to existing state-of-the-art glucose prediction models

### Implementation
The implementation is done in PyTorch, with separate scripts for each dataset to account for their specific characteristics and formats. Training involves batch processing, validation, and hyperparameter optimization for optimal prediction accuracy.

### How to run this code

#### OhioT1DM:

```bash
python ./src/Ohio_Processing_LSTM.py 
bash ./src/ohio_job.sh
```
#### DiaTrend:

```bash
bash ./src/diatrend_job.sh
```

#### T1DEXI:

```bash
bash ./src/t1dexi_job.sh
```

Link to the original paper: https://ieeexplore.ieee.org/document/8779644

Citation to the original paper: 

```
K. Li, C. Liu, T. Zhu, P. Herrero and P. Georgiou, "GluNet: A Deep Learning Framework for Accurate Glucose Forecasting," in IEEE Journal of Biomedical and Health Informatics, vol. 24, no. 2, pp. 414-423, Feb. 2020, doi: 10.1109/JBHI.2019.2931842.
```