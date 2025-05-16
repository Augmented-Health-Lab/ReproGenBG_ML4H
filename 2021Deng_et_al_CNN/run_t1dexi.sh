#!/bin/bash

# Default dataset path
DATASET_PATH="C:\Users\baiyi\OneDrive\Desktop\Modify_GenBG\modified_t1dexi_subset\T1DEXI_cgm_processed"


# Get command line arguments
while getopts "p:f:" opt; do
  case $opt in
    p) DATASET_PATH="$OPTARG"
    ;;
    f) FOLD_NUM="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac
done

if [ -z "$FOLD_NUM" ]; then
    echo "Fold number (-f) is required"
    exit 1
fi

# Run the Python script
python t1dexi_main.py --dataset_path "$DATASET_PATH" --fold_num "$FOLD_NUM"