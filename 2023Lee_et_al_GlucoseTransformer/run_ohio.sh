#!/bin/bash

DEFAULT_DATA_DIR="../../datasets/OhioT1DM"

show_help() {
    echo "Usage: $0 [-d data_directory]"
    echo "Options:"
    echo "  -d    Path to OhioT1DM dataset directory (default: $DEFAULT_DATA_DIR)"
    echo "  -h    Show this help message"
}

DATA_DIR=$DEFAULT_DATA_DIR

while getopts "d:h" opt; do
    case $opt in
        d) DATA_DIR=$OPTARG ;;
        h) show_help; exit 0 ;;
        *) show_help; exit 1 ;;
    esac
done

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Directory $DATA_DIR does not exist!"
    exit 1
fi

echo "Running Ohio T1DM experiment with data from: $DATA_DIR"
python ohio_main.py --data_dir "$DATA_DIR"