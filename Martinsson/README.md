# Replicate the Model Proposed by Martionsson et al., 2019

The replication work relies on the code from the original study repository: https://github.com/johnmartinsson/blood-glucose-prediction. Not all of the files or functions from the original study are used in this study.

## Structure

```
Martinsson/
├── datasets/
│   ├── ohio.py  # Functions for loading OhioT1DM dataset
│   ├── mixed.py  # Not used in this study
│   ├── diatrend.py  # Functions for loading DiaTrend dataset
│   ├── t1dexi.py  # Functions for loading T1DEXI dataset
│   ├── normal_experiment.py  # Not used in this study
├── loss functions/
│   ├── gmse_keras.py # GMSE loss function
│   ├── mse_keras.py # MSE loss function
│   ├── nll_keras.py # **NLL loss function: Used in this method.
│   ├── nll_mse_keras.py # NLL MSE loss function
├── models/
│   ├── basic_lstm_independent_keras.py # Not used in this study
│   ├── basic_lstm_keras.py # Not used in this study
│   ├── lstm_experiment_keras.py # The LSTM model used in this model
├── notebooks/ # Include the exploration notebooks.
├── optimizers/ # The adam optimizer used in this model
├── original_diatrend_experiments_60min/ # Include the config yaml files for models trained on DiaTrend
│   ├── fold1_eval # Fold 1 for evaluation
│   ├── fold2_eval # Fold 2 for evaluation
│   ├── fold3_eval # Fold 3 for evaluation
│   ├── fold4_eval # Fold 4 for evaluation
│   ├── fold5_eval # Fold 5 for evaluation
│   ├── all_final_experiment_fold1.yaml # Config for training fold 1
│   ├── all_final_experiment_fold2.yaml # Config for training fold 2
│   ├── all_final_experiment_fold3.yaml # Config for training fold 3
│   ├── all_final_experiment_fold4.yaml # Config for training fold 4
│   ├── all_final_experiment_fold5.yaml # Config for training fold 5
├── original_t1dexi_experiments_60min/ # Include the config yaml files for models trained on T1DEXI
│   ├── fold1_eval # Fold 1 for evaluation
│   ├── fold2_eval # Fold 2 for evaluation
│   ├── fold3_eval # Fold 3 for evaluation
│   ├── fold4_eval # Fold 4 for evaluation
│   ├── fold5_eval # Fold 5 for evaluation
│   ├── all_final_experiment_fold1.yaml # Config for training fold 1
│   ├── all_final_experiment_fold2.yaml # Config for training fold 2
│   ├── all_final_experiment_fold3.yaml # Config for training fold 3
│   ├── all_final_experiment_fold4.yaml # Config for training fold 4
│   ├── all_final_experiment_fold5.yaml # Config for training fold 5
├── original_ohio_experiments_60min/ # Include the config yaml files for models trained on OhioT1DM
│   ├── 540_all_final_evaluation.yaml # Config for evaluating subject 540
│   ├── ......
│   ├── all_final_experiment.yaml # Config for training on the OhioT1DM training set
├── result_tables/ # results for each individual on different sampling horizons
├── train/ # The training function of this study
├── created_table.py # Not used in this study
├── demo.py # Run a demo example
├── diatrend_main.py # Include the entire training and evaluation process on DiaTrend dataset
├── *.sh # Not used in this study
├── generate_final_experiments_*.py # Not used in this study
├── generate_new_yaml.ipynb # To generate # To generate new yaml for training
├── ohio_main.py # Include the entire training and evaluation process on OhioT1DM dataset
├── t1dexi_main.py # Include the entire training and evaluation process on T1DEXI dataset
├── process_yaml_files.ipynb  # Notebook for processing and updating YAML files.
├── process_the_result_txt.ipynb  # Notebook for processing result text files.
├── README_OriginalStudy.md  # Instructions for reproducing results from the original study.
├── training_evaluation_functions.py # Including the functions for modeling and training. Mainly called in the *main.py files
├── utils.py # Include utils functions.
├── Other files not mentioned are not used in this study.
```


# Original README

## Citation

- Article: https://link.springer.com/article/10.1007%2Fs41666-019-00059-y

Please cite this work if you find this repository useful for your research:

    Martinsson, J., Schliep, A., Eliasson, B. et al. J Healthc Inform Res (2019). https://doi.org/10.1007/s41666-019-00059-y

## Prerequisites
The code is designed to be run on the OhioT1DM Dataset. So to use it the xml_path in e.g. the example experiment YAML configuration need to point to the path on disk where the XML data files are. E.g., change "/home/ubuntu/ohio_data/OhioT1DM-training/" to point to Ohiot1DM-training folder containing the XML files for the ohio dataset.

It would of cource be possible to write a new dataset module which loads the data into the required format and train the models on other data as well.

## Installation
    $> chmod +x setup.sh
    $> ./setup.sh

## Running an experiment
Note that this is designed to run on the Ohio Diabetes dataset. You need to
explicitly state the absolute file path to the XML file of the patient you want
to train the model for in the experiment configuration file (YAML file).

Except for that, everything should run out of the box.

    $> chmod +x run.py
    $> ./run.py --file experiments/example.yaml -m train

All results are collected and stored in the 'artifacts' directory. To visualize the training session you can run

    $> tensorboard --logdir artifacts/<artifacts-path>

and fire up tensorboard.

## Reproduce final results

    $> sh run_final_experiments.sh
    $> sh evaluate_final_experiments.sh
    $> python create_table.py

## Reproduce plots

    # Run the hyperparameter search
    $> python run.py --file=experiments/all_nb_lstm_state_nb_past_steps_search.yaml -m train
    # Evaluate the trained models
    $> python run.py --file=experiments/all_nb_lstm_state_nb_past_steps_search.yaml -m ealuate
    # Hyperparam search plots
    $> python plot_parameter_search.py artifacts/all_nb_lstm_states_nb_past_steps/

The plots will be in the working directory.
    
    # Surveillance error grid plots and prediction plots
    $> sh run_final_plots.sh
    
The plots will be in the artifacts folders.

## Versions

To reproduce the results in [Automatic blood glucose prediction with confidence
using recurrent neural networks](http://ceur-ws.org/Vol-2148/paper10.pdf) revert to commit: [a5f0ebcf45f87b63d118dcad5e96eb505bb4269a](https://github.com/johnmartinsson/blood-glucose-prediction/commit/a5f0ebcf45f87b63d118dcad5e96eb505bb4269a) and follow the README.
