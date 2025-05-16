#!/bin/bash

DATA_DIR=${1:-"../../../ReproGenBG_Dataset/T1DEXI_processed"}

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Directory $DATA_DIR not found"
    exit 1
fi

python t1dexi_main.py --data_dir "$DATA_DIR"