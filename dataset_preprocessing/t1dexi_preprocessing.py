import pandas as pd
import os
import shutil


def main(): 
    """
    Main function to process the T1DEXI dataset. Please change the path to the dataset if needed.
    """
    t1dexi_directory = '../datasets/t1dexi/LB.csv' # LB file from T1DEXI dataset
        
    cgm_data = pd.read_csv('t1dexi_directory', low_memory=False)
    useful_columns = ['USUBJID', 'LBORRES', 'LBDTC']

    # read the selected_t1dexi.txt file
    with open('selected_t1dexi.txt', 'r') as file:
        selected_t1dexi = file.read().splitlines()
        
    selected_t1dexi= [int(subject) for subject in selected_t1dexi]
    selected_df = cgm_data[cgm_data['USUBJID'].isin(selected_t1dexi)][useful_columns]

    for subj in selected_df.USUBJID.unique():
        subj_df = selected_df[selected_df['USUBJID'] == subj]
        # Create a directory for each subject
        os.makedirs(f'../datasets/t1dexi_subset/', exist_ok=True)
        
        # Save the subject's data to a CSV file in their directory
        subj_df.to_csv(f'../datasets/t1dexi_subset/{subj}.csv', index=False)

if __name__ == "__main__":
    main()