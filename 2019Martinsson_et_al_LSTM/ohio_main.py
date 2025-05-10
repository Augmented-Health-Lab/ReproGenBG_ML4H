import os
import pprint
import numpy as np
import yaml
import glob
from training_evaluation_functions import (
    load_cfgs,
    load_module,
    train,
    evaluate_ohio,
    # plot_nll,
    # plot_noise_experiment,
    # plot_seg,
    # plot_target_distribution
)
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


def main_training(yaml_filepath, mode):
    """
    Main function to execute the process based on the provided configuration file and mode.

    Parameters
    ----------
    yaml_filepath : str
        Path to the YAML configuration file.
    mode : str
        Mode of operation: 'train', 'evaluate', 'plot_nll', 'plot_noise_experiment', 'plot_seg', 'plot_dist'.

    Returns
    -------
    None
    """
    # Load configurations
    cfgs = load_cfgs(yaml_filepath)
    print("Running {} experiments.".format(len(cfgs)))

    for cfg in cfgs:
        # Set random seed
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
        print("loading optimizer ...")
        optimizer = module_optimizer.load(cfg['optimizer'])

        # Load loss function
        loss_function = module_loss_function.load()

        # Load model
        print("loading model ...")
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
            loss=loss_function,

        )

        # Execute based on mode
        if mode == 'train':
            print("training model ...")
            train(model, module_train, x_train, y_train, x_valid, y_valid, cfg)
        # elif mode == 'plot_nll':
        #     plot_nll(model, x_test, y_test, cfg)
        # elif mode == 'plot_noise_experiment':
        #     plot_noise_experiment(model, x_test, y_test, cfg)
        # elif mode == 'plot_seg':
        #     plot_seg(model, x_test, y_test, cfg)
        # elif mode == 'plot_dist':
        #     plot_target_distribution(y_test, cfg)
        elif mode == 'evaluate':
            evaluate_ohio(model, x_test, y_test, cfg)
        else:
            print(f"Unknown mode: {mode}")

def main_evaluate(yaml_files, mode):
    """
    Main function to execute the process based on the provided configuration files and mode.

    Parameters
    ----------
    yaml_files : list
        List of paths to the YAML configuration files.
    mode : str
        Mode of operation: 'train', 'evaluate', 'plot_nll', 'plot_noise_experiment', 'plot_seg', 'plot_dist'.

    Returns
    -------
    None
    """
    for yaml_filepath in yaml_files[:-1]:
        # Load configurations
        cfgs = load_cfgs(yaml_filepath)
        print("Running {} experiments.".format(len(cfgs)))

        for cfg in cfgs:
            # Set random seed
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
            print("loading optimizer ...")
            optimizer = module_optimizer.load(cfg['optimizer'])

            # Load loss function
            loss_function = module_loss_function.load()

            # Load model
            print("loading model ...")
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

            # Execute based on mode
            if mode == 'train':
                print("training model ...")
                train(model, module_train, x_train, y_train, x_valid, y_valid, cfg)
            elif mode == 'evaluate':
                print("Evaluating model ...")
                evaluate_ohio(model, x_test, y_test, cfg)
            else:
                print(f"Unknown mode: {mode}")



if __name__ == "__main__":
    # Example usage

    # Replace the path with the path to your configuration file.
    # Make sure to the sampling horizon is set to be the length you defined.
    yaml_filepath = f"./original_ohio_experiments_60min/all_final_experiment.yaml"
    mode = "train"
    main_training(yaml_filepath, mode)

    # Get all yaml files in the directory
    yaml_files = glob.glob("./original_ohio_experiments_60min/*.yaml")
    mode = "evaluate"
    main_evaluate(yaml_files, mode)
    

