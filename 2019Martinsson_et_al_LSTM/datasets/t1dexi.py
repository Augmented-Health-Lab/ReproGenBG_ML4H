# import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from Original_Martinsson import utils
import os
import glob

def load_dataset(cfg):
    if os.path.basename(cfg['csv_path']) == 'all':
        print("loading training data for all patients ...")
        csvs = os.path.join(os.path.dirname(cfg['csv_path']), "*.csv")
        csv_paths = glob.glob(csvs)
        tups = []
        for csv_path in csv_paths:
            cfg['csv_path'] = csv_path
            tups.append(load_data(cfg))
        x_train = np.concatenate([t[0] for t in tups], axis=0)
        y_train = np.concatenate([t[1] for t in tups], axis=0)
        x_valid = np.concatenate([t[2] for t in tups], axis=0)
        y_valid = np.concatenate([t[3] for t in tups], axis=0)
        x_test = np.concatenate([t[4] for t in tups], axis=0)
        y_test = np.concatenate([t[5] for t in tups], axis=0)

        cfg['csv_path'] = 'all'
        return x_train, y_train, x_valid, y_valid, x_test, y_test
    else:
        x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(cfg)
        return x_train, y_train, x_valid, y_valid, x_test, y_test

def load_data(cfg):
    csv_path        = cfg['csv_path']
    nb_past_steps   = int(cfg['nb_past_steps'])
    nb_future_steps = int(cfg['nb_future_steps'])
    train_fraction  = float(cfg['train_fraction'])
    valid_fraction  = float(cfg['valid_fraction'])
    test_fraction   = float(cfg['test_fraction'])
    print("nb_future_steps ", nb_future_steps)

    xs, ys = load_glucose_data(csv_path, nb_past_steps, nb_future_steps)
    ys = np.expand_dims(ys, axis=1)

    x_train, x_valid, x_test = utils.split_data(xs, train_fraction,
            valid_fraction, test_fraction)
    y_train, y_valid, y_test = utils.split_data(ys, train_fraction,
            valid_fraction, test_fraction)

    # scale data
    scale = float(cfg['scale'])
    x_train *= scale
    y_train *= scale
    x_valid *= scale
    y_valid *= scale
    x_test  *= scale
    y_test  *= scale

    return x_train, y_train, x_valid, y_valid, x_test, y_test

def load_glucose_data(csv_path, nb_past_steps, nb_future_steps):
    df_glucose_level = load_T1DEXI_series(csv_path)
    dt = df_glucose_level.index.to_series().diff().dropna()
    idx_breaks = np.argwhere(dt>pd.Timedelta(6, 'm'))

    # It would be possible to load more features here
    nd_glucose_level = df_glucose_level.values
    consecutive_segments = np.split(nd_glucose_level, idx_breaks.flatten())

    consecutive_segments = [c for c in consecutive_segments if len(c) >=
            nb_past_steps+nb_future_steps]

    sups = [utils.sequence_to_supervised(c, nb_past_steps, nb_future_steps) for
            c in consecutive_segments]

    xss = [sup[0] for sup in sups]
    yss = [sup[1] for sup in sups]

    xs = np.concatenate(xss)
    ys = np.concatenate(yss)

    return np.expand_dims(xs, axis=2), ys

def parse_date(date_str):
    try:
        return pd.to_datetime(date_str, format="%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        try:
            return pd.to_datetime(date_str, format="%Y-%m-%d %H:%M:%S")
        except ValueError:
            return pd.NaT

def load_T1DEXI_series(path):
    subject = pd.read_csv(path)

    # Lists to store the results
    parsed_dates = []
    values = []

    # Iterate through each row in the DataFrame
    for index, row in subject.iterrows():
        # Parse the date using the custom function
        parsed_date = parse_date(row['LBDTC'])
        
        # Append the parsed date and corresponding value to the lists
        parsed_dates.append(parsed_date)
        values.append(float(row['LBORRES']))

    # Now 'parsed_dates' and 'values' contain your data
    # print(parsed_dates)
    # print(values)
    index = pd.DatetimeIndex(parsed_dates)
    series = pd.Series(values, index=index)
    
    return series