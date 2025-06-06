import os
import pprint
import numpy as np
import yaml
import glob
import argparse
from training_evaluation_functions import (
    load_cfgs,
    load_module,
    train,
    evaluate_diatrend,
)

def main_diatrend(fold_number):
    """
    Main function to execute training and evaluation for the DiaTrend dataset.

    Parameters:
    -----------
    fold_number : int
        The fold number to process (1-5)
    """
    # Training phase
    yaml_filepath = f"./original_diatrend_experiments_60min/all_final_experiment_fold{fold_number}.yaml"
    mode = "train"

    cfgs = load_cfgs(yaml_filepath)
    print(f"Running {len(cfgs)} experiments for fold {fold_number} (Training).")
    for cfg in cfgs:
        seed = int(cfg['train']['seed'])
        np.random.seed(seed)

        # Load modules dynamically
        module_dataset = load_module(cfg['dataset']['script_path'])
        module_model = load_module(cfg['model']['script_path'])
        module_optimizer = load_module(cfg['optimizer']['script_path'])
        module_loss_function = load_module(cfg['loss_function']['script_path'])
        module_train = load_module(cfg['train']['script_path'])

        # Pretty print configuration
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(cfg)

        # Load dataset
        x_train, y_train, x_valid, y_valid, x_test, y_test = module_dataset.load_dataset(cfg['dataset'])

        print("x_train.shape: ", x_train.shape)
        print("y_train.shape: ", y_train.shape)
        print("x_valid.shape: ", x_valid.shape)
        print("y_valid.shape: ", y_valid.shape)
        print("x_test.shape: ", x_test.shape)
        print("y_test.shape: ", y_test.shape)

        # Load optimizer
        optimizer = module_optimizer.load(cfg['optimizer'])

        # Load loss function
        loss_function = module_loss_function.load()

        # Load model
        if 'tf_nll' in loss_function.__name__:
            model = module_model.load(
                x_train.shape[1:],
                y_train.shape[1] * 2,
                cfg['model']
            )
        else:
            model = module_model.load(
                x_train.shape[1:],
                y_train.shape[1],
                cfg['model']
            )

        # Load initial weights if specified
        if 'initial_weights_path' in cfg['train']:
            print("Loading initial weights: ", cfg['train']['initial_weights_path'])
            model.load_weights(cfg['train']['initial_weights_path'])

        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=loss_function
        )

        # Execute training
        if mode == 'train':
            print("Training model ...")
            train(model, module_train, x_train, y_train, x_valid, y_valid, cfg)

    # Evaluation phase
    yaml_files = glob.glob(f"./original_diatrend_experiments_60min/fold{fold_number}_eval/*.yaml")
    mode = "evaluate"
    print(f"Running {len(yaml_files)} experiments for fold {fold_number} (Evaluation).")
    for yaml_filepath in yaml_files:
        cfgs = load_cfgs(yaml_filepath)
        for cfg in cfgs:
            seed = int(cfg['train']['seed'])
            np.random.seed(seed)

            # Load modules dynamically
            module_dataset = load_module(cfg['dataset']['script_path'])
            module_model = load_module(cfg['model']['script_path'])
            module_optimizer = load_module(cfg['optimizer']['script_path'])
            module_loss_function = load_module(cfg['loss_function']['script_path'])
            module_train = load_module(cfg['train']['script_path'])

            # Pretty print configuration
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(cfg)

            # Load dataset
            x_train, y_train, x_valid, y_valid, x_test, y_test = module_dataset.load_dataset(cfg['dataset'])

            print("x_train.shape: ", x_train.shape)
            print("y_train.shape: ", y_train.shape)
            print("x_valid.shape: ", x_valid.shape)
            print("y_valid.shape: ", y_valid.shape)
            print("x_test.shape: ", x_test.shape)
            print("y_test.shape: ", y_test.shape)

            # Load optimizer
            optimizer = module_optimizer.load(cfg['optimizer'])

            # Load loss function
            loss_function = module_loss_function.load()

            # Load model
            if 'tf_nll' in loss_function.__name__:
                model = module_model.load(
                    x_train.shape[1:],
                    y_train.shape[1] * 2,
                    cfg['model']
                )
            else:
                model = module_model.load(
                    x_train.shape[1:],
                    y_train.shape[1],
                    cfg['model']
                )

            # Load initial weights if specified
            if 'initial_weights_path' in cfg['train']:
                print("Loading initial weights: ", cfg['train']['initial_weights_path'])
                model.load_weights(cfg['train']['initial_weights_path'])

            # Compile model
            model.compile(
                optimizer=optimizer,
                loss=loss_function
            )

            # Execute evaluation
            if mode == 'evaluate':
                print("Evaluating model ...")
                evaluate_diatrend(model, x_test, y_test, cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run DiaTrend experiments with specific fold number')
    parser.add_argument('--fold', type=int, choices=range(1, 6), 
                      help='Fold number (1-5)', required=True)
    args = parser.parse_args()
    
    main_diatrend(args.fold)
