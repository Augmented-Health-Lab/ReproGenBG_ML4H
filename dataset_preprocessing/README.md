# Dataset Preprocessing Scripts

This directory contains preprocessing scripts for the DiaTrend and T1DEXI datasets used in blood glucose prediction models.

## Directory Structure
```
dataset_preprocessing/
├── diatrend_preprocess.sh       # Shell script to run DiaTrend preprocessing
├── diatrend_preprocessing.py    # Python script for DiaTrend data processing
├── fold_split.csv              # Configuration file for dataset splits
├── selected_t1dexi.txt         # List of selected T1DEXI subject IDs
├── t1dexi_preprocess.sh        # Shell script to run T1DEXI preprocessing
└── t1dexi_preprocessing.py     # Python script for T1DEXI data processing
```

## Prerequisites

- Python 3.8+
- Required Python packages:
  - pandas
  - numpy
  - openpyxl (for Excel file support)

## DiaTrend Dataset Preprocessing

The DiaTrend preprocessing script processes raw Excel files and organizes them into training folds.

### Usage

1. Using bash script:
```bash
chmod +x diatrend_preprocess.sh
./diatrend_preprocess.sh -i /path/to/raw/diatrend/data -o /path/to/output/directory
```

2. Direct Python execution:
```bash
python diatrend_preprocessing.py --input_dir /path/to/raw/diatrend/data --output_dir /path/to/output/directory
```

### Expected Input/Output

- Input: Directory containing raw DiaTrend Excel files
- Output: 
  - Processed CSV files
  - 5-fold split directories (fold1_training through fold5_training)

Note: The 5-fold split is only for these following methods replication: 2019Martinsson_et_al_LSTM, 2021vanDoorn_et_al_LSTM, and 2021Deng_et_al_CNN. The rest methods have fold split functions included in their code.

## T1DEXI Dataset Preprocessing

The T1DEXI preprocessing script processes the raw CSV file and extracts data for selected subjects.

### Usage

1. Using bash script:
```bash
chmod +x t1dexi_preprocess.sh
./t1dexi_preprocess.sh -i /path/to/LB.csv -o /path/to/output/directory -s /path/to/selected_t1dexi.txt
```

2. Direct Python execution:
```bash
python t1dexi_preprocessing.py --input_file /path/to/LB.csv --output_dir /path/to/output/directory --selected_subjects selected_t1dexi.txt
```

### Expected Input/Output

- Input:
  - LB.csv file containing raw T1DEXI data
  - selected_t1dexi.txt containing subject IDs to process
- Output:
  - Individual CSV files for each processed subject

## Configuration Files

- `fold_split.csv`: Contains the fold assignments for both DiaTrend and T1DEXI datasets
- `selected_t1dexi.txt`: Lists the T1DEXI subject IDs to include in processing

## Windows Users

For Windows users using Command Prompt:

```batch
python diatrend_preprocessing.py --input_dir "path\to\raw\data" --output_dir "path\to\output"
python t1dexi_preprocessing.py --input_file "path\to\LB.csv" --output_dir "path\to\output" --selected_subjects "selected_t1dexi.txt"
```

## Notes

- The preprocessing includes data cleaning, validation, and organization into proper format
- Default parameters can be modified in the Python scripts
- Make sure input directories and files exist before running the scripts
- Output directories will be created if they don't exist

## Error Handling

The scripts include basic error checking for:
- Missing input directories/files
- Invalid file formats
- Missing required data fields