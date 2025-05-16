# Replicate the Model Proposed by Deng et al., 2021

The replication work followed the paper: S. -M. Lee, D. -Y. Kim and J. Woo, "Glucose Transformer: Forecasting Glucose Level and Events of Hyperglycemia and Hypoglycemia," in IEEE Journal of Biomedical and Health Informatics, vol. 27, no. 3, pp. 1600-1611, March 2023, doi: 10.1109/JBHI.2023.3236822.

## Structure

```
AccurateBG/
├── notebook/               # Reference notebook. Supplementary execution files
├── src/
│   ├── __init__.py         # Initiate function
│   ├── data.py             # Handle data preparation for training and validation
│   ├── model.py             # Include the transformer encoder model
│   ├── train.py            # Include the training process
│   ├── utils.py            # Include all the utility functions
├── diatrend_main.py        # The main training and evaluation functions on DiaTrend 
├── ohio_main.py            # The main training and evaluation functions on OhioT1DM
├── t1dexi_main.py          # The main training and evaluation functions on T1DEXI
├── requirement.txt         # Include the dependency of running this method
└── README.md               
```

## How to run this code

### OhioT1DM

```sh
chmod +x run_ohio.sh
./run_ohio.sh -d /path/to/OhioT1DM_dataset
```

### DiaTrend
```sh
chmod +x run_diatrend.sh
./run_diatrend.sh -d /path/to/your/diatrend/dataset 
```

### T1DEXI
```sh
chmod +x run_t1dexi.sh
./run_t1dexi.sh /path/to/T1DEXI/dataset
```

## Note

The fold split function is inherently inserted into the preprocessing code. So the fold number doesn't need to be specified when executing the code for this method. 