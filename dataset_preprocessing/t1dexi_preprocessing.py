import pandas as pd
import os
import shutil
import argparse

def process_t1dexi(input_file, output_dir, selected_subjects_file):
    """
    Process T1DEXI dataset and save individual subject files
    
    Args:
        input_file (str): Path to T1DEXI LB.csv file
        output_dir (str): Directory for processed output files
        selected_subjects_file (str): Path to file containing selected subject IDs
    """
    # Read CGM data
    cgm_data = pd.read_csv(input_file, low_memory=False)
    useful_columns = ['USUBJID', 'LBORRES', 'LBDTC']
    
    # Read selected subjects
    with open(selected_subjects_file, 'r') as file:
        selected_t1dexi = [int(subject) for subject in file.read().splitlines()]
    
    # Filter data
    selected_df = cgm_data[cgm_data['USUBJID'].isin(selected_t1dexi)][useful_columns]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each subject
    for subj in selected_df.USUBJID.unique():
        subj_df = selected_df[selected_df['USUBJID'] == subj]
        output_file = os.path.join(output_dir, f'{subj}.csv')
        subj_df.to_csv(output_file, index=False)
        print(f"Processed subject {subj}")

def main():
    parser = argparse.ArgumentParser(description='Process T1DEXI dataset')
    parser.add_argument('--input_file', type=str, required=True,
                      help='Path to T1DEXI LB.csv file')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory for processed output files')
    parser.add_argument('--selected_subjects', type=str, default='selected_t1dexi.txt',
                      help='Path to file containing selected subject IDs')
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    if not os.path.exists(args.selected_subjects):
        raise FileNotFoundError(f"Selected subjects file not found: {args.selected_subjects}")
    
    print(f"Processing T1DEXI dataset...")
    print(f"Input file: {args.input_file}")
    print(f"Output directory: {args.output_dir}")
    
    process_t1dexi(args.input_file, args.output_dir, args.selected_subjects)

if __name__ == "__main__":
    main()