# Replicate the Model Proposed by van Doorn et al., 2021

When replicating this work, we referred to the framework from Martinsson et al. due to similarities in the model structure and training policies, but independently recreated the models, loss function, and related configuration files based on the descriptions provided in the manuscript.
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
│   ├── mse_keras.py # **MSE loss function: Used in this method.
│   ├── nll_keras.py # NLL loss function
│   ├── nll_mse_keras.py # NLL MSE loss function
├── models/
│   ├── basic_lstm_independent_keras.py # Not used in this study
│   ├── basic_lstm_keras.py # Not used in this study
│   ├── lstm_experiment_keras.py # The LSTM model used in this model
├── notebooks/ # Include the exploration notebooks.
├── optimizers/ # The adam optimizer used in this model
├── vandoorn_diatrend_experiments_30min/ # Include the config yaml files for models trained on DiaTrend
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
├── vandoorn_t1dexi_experiments_30min/ # Include the config yaml files for models trained on T1DEXI
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
├── original_vandoorn_experiments/ # Include the config yaml files for models trained on OhioT1DM
│   ├── 540_all_final_evaluation.yaml # Config for evaluating subject 540
│   ├── ......
│   ├── all_final_experiment.yaml # Config for training on the OhioT1DM training set
├── result_tables/ # results for each individual on different sampling horizons
├── train/ # The training function of this study
├── created_table.py # Not used in this study
├── demo.py # Run a demo example
├── vandoorn_diatrend_main.py # Include the entire training and evaluation process on DiaTrend dataset
├── *.sh # Not used in this study
├── generate_final_experiments_*.py # Not used in this study
├── generate_new_yaml.ipynb # To generate # To generate new yaml for training
├── vandoorn_ohio_main.py # Include the entire training and evaluation process on OhioT1DM dataset
├── vandoorn_t1dexi_main.py # Include the entire training and evaluation process on T1DEXI dataset
├── process_yaml_files.ipynb  # Notebook for processing and updating YAML files.
├── process_the_result_txt.ipynb  # Notebook for processing result text files.
├── README_OriginalStudy.md  # Instructions for reproducing results from the original study.
├── training_evaluation_functions.py # Including the functions for modeling and training. Mainly called in the *main.py files
├── utils.py # Include utils functions.
├── Other files not mentioned are not used in this study.
```

