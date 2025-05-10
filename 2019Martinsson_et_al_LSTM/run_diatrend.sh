#!/bin/bash

# Check if fold number is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide a fold number (1-5)"
    echo "Usage: ./run_diatrend.sh <fold_number>"
    exit 1
fi

# Check if fold number is valid
if [ $1 -lt 1 ] || [ $1 -gt 5 ]; then
    echo "Error: Fold number must be between 1 and 5"
    exit 1
fi

# Run the Python script with the provided fold number
python diatrend_main.py --fold $1