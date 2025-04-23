# Deep Learning for Blood Glucose Prediction: Reproducibility and Generalizability

This repository contains the implementation and replication code for our study:  
**"Deep Learning for Blood Glucose Prediction: Investigating Reproducibility, Bias, and Factors Affecting Differential Performance."**

## Abstract: 
(Place holder for abstract)

---

## 📂 Repository Structure
The repository is organized as follows:

- `MartinssonAndvanDoorn/` – Implementation of Martinsson et al. and vanDoorn et al.
- `AccurateBG/` – Implementation of Deng et al.
- `GlucoseTransformer/` – Implementation of Lee et al.
- `GluNet/` – Implementation of Li et al.
- `Stacked LSTM/` – Implementation of Rabby et al.
- `requirements.txt` – Python package dependencies
- `README.md` – Project overview and instructions (you are here)


---

## 📋 Requirements

```bash
# Create and activate a virtual environment (optional)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```


## 🚀 How to Run

### 1. Prepare the Dataset

Please refer to the data preparation instructions for each dataset below.  
**Note:** Due to licensing restrictions, raw datasets are not included in this repository.

- **OhioT1DM**: Download from [link or citation] and place the files under `data/ohiot1dm/`
- **DiaTrend**: Request access via [DiaTrend Dataset Access Link]
- **T1DEXI**: Apply for access through [T1DEXI Access Portal]

After accessing these three datasets, please follow these instructions for data sanitation:

- **OhioT1DM**: Please directly put the dataset folder into this directory: `../datasets/[OhioT1DM]`
- **DiaTrend**: Please following [this script](dataset_preprocessing/diatrend_preprocessing.py) for data preprocessing. And please put the generated `../datasets/diatrend_subset/fold[N]_training` and `../datasets/diatrend_subset/processed_cgm_data_Subject[N].csv`
- **T1DEXI**: Please following this [subset list](dataset_preprocessing/selected_t1dexi.txt) and notebook for data preprocessing. And please put the generated `../datasets/t1dexi_subset/fold[N]_training` and `../datasets/t1dexi_subset/[N].csv`

### 2. Run the Models

Each dataset has a dedicated main script to train and evaluate all six methods. And the parameters are all based on the version provided by the original studies. 

#### 2.1 Martinsson et al., 2019
Before you run the command, please ensure to change the data path in config yaml file 
##### OhioT1DM:

```bash
python ./MartinssonAndvanDoorn/ohio_main.py
```
##### DiaTrend:

```bash
python ./MartinssonAndvanDoorn/diatrend_main.py
```

##### T1DEXI:

```bash
python ./MartinssonAndvanDoorn/t1dexi_main.py
```


#### 2.3 van Doorn et al., 2021
##### OhioT1DM:

```bash
python ./MartinssonAndvanDoorn/vandoorn_ohio_main.py
```
##### DiaTrend:

```bash
python ./MartinssonAndvanDoorn/vandoorn_diatrend_main.py
```

##### T1DEXI:

```bash
python ./MartinssonAndvanDoorn/vandoorn_t1dexi_main.py
```


#### 2.4 Deng et al., 2021
##### OhioT1DM:

```bash
python ./AccurateBG/accurate_bg/new_ohio_main.py
```
##### DiaTrend:

```bash
python ./AccurateBG/accurate_bg/diatrend_main.py
```

##### T1DEXI:

```bash
python ../AccurateBG/accurate_bg/t1dexi_main.py
```

