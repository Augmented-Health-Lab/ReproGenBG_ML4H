#!/bin/bash

DEFAULT_INPUT_FILE="../datasets/t1dexi/LB.csv"
DEFAULT_OUTPUT_DIR="../datasets/t1dexi_subset"
DEFAULT_SELECTED_SUBJECTS="selected_t1dexi.txt"

while getopts "i:o:s:h" opt; do
    case $opt in
        i) INPUT_FILE=$OPTARG ;;
        o) OUTPUT_DIR=$OPTARG ;;
        s) SELECTED_SUBJECTS=$OPTARG ;;
        h) 
            echo "Usage: $0 [-i input_file] [-o output_dir] [-s selected_subjects_file]"
            exit 0
            ;;
        *) 
            echo "Invalid option"
            exit 1
            ;;
    esac
done

# Use defaults if not specified
INPUT_FILE=${INPUT_FILE:-$DEFAULT_INPUT_FILE}
OUTPUT_DIR=${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}
SELECTED_SUBJECTS=${SELECTED_SUBJECTS:-$DEFAULT_SELECTED_SUBJECTS}

# Run preprocessing script
python t1dexi_preprocessing.py \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --selected_subjects "$SELECTED_SUBJECTS"