import pandas as pd
import os
import shutil
import argparse

# The fold split functions are specificly applied to the following three methods: 
# Martinsson et al. , vanDoorn et al. and Deng et al. 
# Lee et al, Li et al, and Rabby et al. have their fold split functions in their own respective files.
def process_patient_cgm(cgm_df, min_days=28, max_days=42, records_threshold=200, min_records=200, max_records=288):
    """
    Process patient CGM data based on the number of valid recording days.
    
    Args:
        cgm_df: DataFrame with CGM data
        min_days: Minimum required days with >200 records (default: 28)
        max_days: Maximum days to keep (default: 42)
        records_threshold: Minimum records per day threshold (default: 200)
    
    Returns:
        processed_df: Processed DataFrame or None if patient should be excluded
        status: String indicating the processing status
    """
    # Ensure datetime format
    cgm_df = cgm_df.copy()
    cgm_df['date'] = pd.to_datetime(cgm_df['date'])
    cgm_df['date_only'] = cgm_df['date'].dt.date
    
    # Calculate records per day
    records_per_day = cgm_df.groupby('date_only').size()
    valid_days = records_per_day[records_per_day > records_threshold]
    num_valid_days = len(valid_days)
    
    print(f"Number of days with >{records_threshold} records: {num_valid_days}")
    
    # Case 1: Exclude patient
    if num_valid_days < min_days:
        return None, f"Excluded: Only {num_valid_days} valid days (<{min_days})"
    
    # Case 2: Keep as is
    elif min_days <= num_valid_days <= max_days:
        return cgm_df, f"Kept: {num_valid_days} valid days (within range)"
    
    # Case 3: Truncate to max_days
    else:
        valid_dates = sorted(valid_days.index)
        # Calculate the latest possible start date (ensuring 42 days are available after it)
        latest_start_idx = len(valid_dates) - max_days
        
        # Randomly select a start date index
        # start_idx = np.random.randint(0, latest_start_idx + 1)
        start_idx = 0
        selected_dates = valid_dates[start_idx:start_idx + max_days]
        print(selected_dates)
        # Filter dataframe to keep only selected days
        processed_df = cgm_df[cgm_df['date_only'].isin(selected_dates)]
        
        status_msg = (f"Truncated: Selected {max_days} consecutive days\n"
                     f"Date range: {selected_dates[0]} to {selected_dates[-1]}\n"
                     f"Records per day - Min: {valid_days[selected_dates].min():.0f}, "
                     f"Max: {valid_days[selected_dates].max():.0f}, "
                     f"Mean: {valid_days[selected_dates].mean():.0f}")
        
        return processed_df, status_msg
# Example usage
def analyze_patient_data(cgm_df):
    """
    Analyze and display information about the processing results
    """
    processed_df, status = process_patient_cgm(cgm_df)
    
    print("\nProcessing Status:", status)
    
    if processed_df is not None:
        # Calculate records per day in processed data
        records_per_day = processed_df.groupby('date_only').size()
        
        print("\nProcessed Data Statistics:")
        print(f"Total days: {len(records_per_day)}")
        print(f"Days with >200 records: {len(records_per_day[records_per_day > 200])}")
        print(f"Average records per day: {records_per_day.mean():.1f}")

        return processed_df
    else:
        print("Patient excluded from analysis")
        return None
    

def extract_diatrend_cgm(input_dir, output_dir, filename):
    """
    Process and extract CGM data from Excel files
    
    Args:
        input_dir (str): Directory containing raw Diatrend dataset
        output_dir (str): Directory for processed output files
        filename (str): Name of the Excel file to process
    """
    cgm_df = pd.read_excel(os.path.join(input_dir, filename), sheet_name="CGM")
    processed_df = analyze_patient_data(cgm_df)
    
    if processed_df is not None:
        cleaned_df = processed_df[['date', 'mg/dl']].copy()
        cleaned_df = cleaned_df.sort_values('date')
        cleaned_df = cleaned_df.reset_index(drop=True)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"processed_cgm_data_{filename[:-5]}.csv")
        cleaned_df.to_csv(output_path, index=False)
        print(f"Processed data saved to: {output_path}")

def main(input_dir, output_dir):
    """
    Main function to process Diatrend dataset
    
    Args:
        input_dir (str): Directory containing raw Diatrend files
        output_dir (str): Directory for processed output
    """
    # Process all Excel files
    all_diatrend_subjects_filename = [f for f in os.listdir(input_dir) 
                                    if f.endswith('.xlsx') and os.path.isfile(os.path.join(input_dir, f))]
    
    for filename in all_diatrend_subjects_filename:
        print(f"\nProcessing {filename}...")
        extract_diatrend_cgm(input_dir, output_dir, filename)
    
    # Split into folds
    folds_dir = os.path.join(output_dir, "folds")
    split_into_folds(output_dir, folds_dir)

def split_into_folds(input_dir, output_dir, fold_size=11, total_folds=5):
    """
    Split processed CGM data files into foldK_training directories.
    You can also manually put the processed files into the foldK_training folders.

    Args:
        input_dir (str): Directory containing the processed CGM data files.
        output_dir (str): Directory where foldK_training folders will be created.
        fold_size (int): Number of files per fold (default: 11).
        total_folds (int): Total number of folds (default: 5).
    """
    # Get all processed files in the input directory
    all_files = sorted([f for f in os.listdir(input_dir) if f.startswith("processed_cgm_data_") and f.endswith(".csv")])

    # Ensure the number of files matches the expected total
    if len(all_files) != fold_size * total_folds:
        raise ValueError(f"Expected {fold_size * total_folds} files, but found {len(all_files)} in {input_dir}.")

    # Create fold directories and move files
    for fold in range(1, total_folds + 1):
        fold_dir = os.path.join(output_dir, f"fold{fold}_training")
        os.makedirs(fold_dir, exist_ok=True)

        # Determine the range of files for this fold
        start_idx = (fold - 1) * fold_size
        end_idx = start_idx + fold_size
        fold_files = all_files[start_idx:end_idx]

        # Move files to the fold directory
        for file in fold_files:
            src_path = os.path.join(input_dir, file)
            dest_path = os.path.join(fold_dir, file)
            shutil.move(src_path, dest_path)

        print(f"Fold {fold}: Moved {len(fold_files)} files to {fold_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Diatrend dataset and create fold splits')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Directory containing raw Diatrend dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory for processed output files')
    
    args = parser.parse_args()
    
    # Verify directories
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    
    print(f"Processing Diatrend dataset...")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    
    main(args.input_dir, args.output_dir)