# 2021 Deng et al. CNN Implementation

This directory contains the replication work of the CNN-based blood glucose prediction model from Deng et al. (2021)[1]. The replication depends on the original code provided by the original work [2]. However, not all files or functions from the original repository were used in this study.

## Project Structure
```
2021Deng_et_al_CNN/
├── AccurateBG/                          # The original code repository from Deng et al. 2021
│   └── accurate_bg/                     # The primary folder for all the functions
│       ├── cgms_data_seg.py             # Functions for data preprocessing (for OhioT1DM)
│       ├── CGMSData.py                 # Functions for data construction (for OhioT1DM)
│       ├── cnn_ohio.py                  # Models and training
│       ├── data_reader.py              # Dataset reader for OhioT1DM
│       ├── helper.py                   # Helper functions
│       ├── mixup.py                    # Data Augmentation function
│       ├── ohio_main.py                # Original main function
│       └── utils.py                    # Utility functions   
├── diatrend_results/
│   ├── result_tables/                  # Include the Diatrend result tables with different sampling horizon
│   └── config.json                     # Config files for DiaTrend training and evaluation
├── ohio_results/
│   ├── result_tables/                  # Include the Ohio result tables with different sampling horizon
│   └── config.json                     # Config files for Ohio training and evaluation (Replication)
├── diatrend_main.py                    # The main execution file for the replication on DiaTrend dataset
├── new_ohio_main.py                    # The main execution file for the replication on OhioT1DM dataset
├── t1dexi_main.py                    # The main execution file for the replication on T1DEXI dataset
├── cgms_data_seg_diatrend.py             # Functions for data preprocessing for DiaTrend
├── cgms_data_seg_t1dexi.py             # Functions for data preprocessing for T1DEXI
├── CGMSDataDiaTrend.py                 # Functions for data construction for DiaTrend
├── CGMSDataT1DEXI.py                 # Functions for data construction for T1DEXI
├── data_reader_DiaTrend.py             # Dataset reader for DiaTrend
├── data_reader_T1DEXI.py             # Dataset reader for T1DEXI
├── run_ohio.sh                         # Bash file to run replication code on ohioT1DM
├── run_diatrend.sh                     # Bash file to run replication code on DiaTrend
├── run_t1dexi.sh                     # Bash file to run replication code on T1DEXI
└── README.md
```

## Running the Code

### For Ohio Dataset
1. Make the script executable:
```bash
chmod +x run_ohio.sh
```

2. Run the script:
```bash
./run_ohio.sh
```

### For DiaTrend Dataset

#### Running Single Fold
1. Make the script executable:
```bash
chmod +x run_diatrend.sh
```

2. Run a specific fold (replace paths accordingly):
```bash
./run_diatrend.sh -p "/path/to/diatrend_dataset" -f 1 # Fold 1
```
This example is for the training and evalution on the first fold of the DiaTrend dataset. For the other fold, please replace the number after -f.

3. Run a specific fold (replace paths accordingly):
```bash
./run_t1dexi.sh -p "/path/to/t1dexi_dataset" -f 1 # Fold 1
```
This example is for the training and evalution on the first fold of the T1DEXI dataset. For the other fold, please replace the number after -f.

### Windows Users
For Windows users using Command Prompt:
```batch
python new_ohio_main.py
```
or
```batch
python diatrend_main.py --dataset_path "path\to\dataset" --fold_num 1
```

## Configuration
- Configuration files for both Ohio and DiaTrend implementations are located in their respective results folders
- Modify `config.json` in each folder to adjust model parameters

## Model Parameters
- Prediction Horizon (PH): 6 time steps
- Sampling Horizon (SH): 6 time steps
- Training Epochs: 80

## Output
Results will be saved in:
- `ohio_results/` for Ohio dataset
- `diatrend_results/` for DiaTrend dataset
- `t1dexi_results/` for T1DEXI dataset

## Reference
- [1] Deng, Y., Lu, L., Aponte, L. et al. Deep transfer learning and data augmentation improve glucose levels prediction in type 2 diabetes patients. npj Digit. Med. 4, 109 (2021). https://doi.org/10.1038/s41746-021-00480-x
- [2] Deng et al., 2021: https://github.com/yixiangD/AccurateBG 

