#!/bin/bash

# Default dataset path
DEFAULT_DATA_DIR="../../dataset/diatrend_subset/"

# Help message
show_help() {
    echo "Usage: $0 [-d data_directory]"
    echo "Options:"
    echo "  -d    Specify the path to diatrend dataset directory"
    echo "        Default: $DEFAULT_DATA_DIR"
    echo "  -h    Show this help message"
}

# Parse command line options
DATA_DIR=$DEFAULT_DATA_DIR

while getopts "d:h" opt; do
    case $opt in
        d)
            DATA_DIR=$OPTARG
            ;;
        h)
            show_help
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            show_help
            exit 1
            ;;
    esac
done

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Directory $DATA_DIR does not exist!"
    exit 1
fi

# Run the Python script
echo "Running with data directory: $DATA_DIR"
python diatrend_main.py --data_dir "$DATA_DIR"