#!/bin/bash

DEFAULT_INPUT_DIR="../diatrend_dataset"
DEFAULT_OUTPUT_DIR="../datasets/diatrend_subset"

while getopts "i:o:h" opt; do
    case $opt in
        i) INPUT_DIR=$OPTARG ;;
        o) OUTPUT_DIR=$OPTARG ;;
        h) 
            echo "Usage: $0 [-i input_dir] [-o output_dir]"
            exit 0
            ;;
        *) 
            echo "Invalid option"
            exit 1
            ;;
    esac
done

# Use defaults if not specified
INPUT_DIR=${INPUT_DIR:-$DEFAULT_INPUT_DIR}
OUTPUT_DIR=${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}

# Run preprocessing script
python diatrend_preprocessing.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR"