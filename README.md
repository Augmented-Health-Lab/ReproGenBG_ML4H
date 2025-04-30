# Deep Learning for Blood Glucose Prediction: Reproducibility and Generalizability

This repository contains the implementation and replication code for our study:  
**"Deep Learning for Blood Glucose Prediction: Investigating Reproducibility, Bias, and Factors Affecting Differential Performance."**

## Abstract: 
(Place holder for abstract)


## Literature review summary

We reviewed 60 peer-reviewed papers published between 2018 and 2025 that introduce and evaluate deep learning models for blood glucose prediction. Drawing on prior work, we assessed each study against key factors impacting reproducibility, including: (1) availability of code and data, (2) completeness of model reporting (e.g., hyperparameters and tuning), and (3) use of standardized evaluation protocols (e.g., prediction horizon and metrics). All reviewed papers appeared in engineering or interdisciplinary journals and conferences. A summary table of our literature review is available here: [literature review table](Literature_review_summary/Submit%20version%20of%20literature%20review%20table.xlsx).

---

## 📂 Repository Structure
The repository is organized as follows:

- `AccurateBG/` – The implementation of Deng et al.
- `Baseline/` – The implementation of baseline method. 
- `dataset_preprocessing/` - Includes data preprocesssing script, subset patient id, and fold split. 
- `GlucoseTransformer/` – The first version implementation of Lee et al. Deprecated.
- `GlucoseTransformer_organized/` – The implementation of Lee et al.
- `Li_et_al_GluNet/` – The implementation of Li et al.
- `Literature_review_summary/` - Summarize from the lierature
- `Martinsson/` – The implementation of Martinsson et al. 
- `Rabby_et_al_StackedLSTM/` – The implementation of Rabby et al.
- `vanDoorn/` - The implementation of vanDoorn et al.
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

- **OhioT1DM**: Request the access to the dataset: https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/OhioT1DM-dataset.html
- **DiaTrend**: Request access thru Synapse: https://www.synapse.org/Synapse:syn38187184/wiki/619490. 
- **T1DEXI**: Apply for access Vivli: https://search.vivli.org/doiLanding/studies/PR00008428/isLanding.

After accessing these three datasets, please follow these instructions for data sanitation:

Create a folder on the same level with the pulled "ReproGenBG_ML4H" folder. Name it as "datasets"

- **OhioT1DM**: Please directly put the dataset folder into this directory: `../datasets/[OhioT1DM]`. Also copy all the xml files from folder 2018 and 2020 to another folder "both" in the same folder as 2018 and 2020. 
- **DiaTrend**: Please following [this script](dataset_preprocessing/diatrend_preprocessing.py) for data preprocessing. And please put the generated `../datasets/diatrend_subset/fold[N]_training` and `../datasets/diatrend_subset/processed_cgm_data_Subject[N].csv`. 
- **T1DEXI**: Please following this [subset list](dataset_preprocessing/selected_t1dexi.txt) and notebook for data preprocessing. And please put the generated `../datasets/t1dexi_subset/fold[N]_training` and `../datasets/t1dexi_subset/[N].csv`

Please feel free to manually drag and split the folds. [This table](dataset_preprocessing/fold_split.csv) descibes how the fold split in DiaTrend and T1DEXI.
### 2. Run the Models

Each dataset has a dedicated main script to train and evaluate all six methods. And the parameters are all based on the version provided by the original studies. 

#### 2.1 Martinsson et al., 2019 [1]
Before you run the command, please ensure to change the data path in config yaml file 
##### OhioT1DM:

```bash
python ./Martinsson/ohio_main.py
```
##### DiaTrend:

```bash
python ./Martinsson/diatrend_main.py
```

##### T1DEXI:

```bash
python ./Martinsson/t1dexi_main.py
```

#### 2.2 Li et al., 2021 [2]
##### OhioT1DM:

```bash
python ./Li_et_al_GluNet/Ohio_Processing_LSTM.py 
bash ./Li_et_al_GluNet/ohio_job.sh
```
##### DiaTrend:

```bash
bash ./Li_et_al_GluNet/diatrend_job.sh
```

##### T1DEXI:

```bash
bash ./Li_et_al_GluNet/t1dexi_job.sh
```

#### 2.3 van Doorn et al., 2021 [3]
##### OhioT1DM:

```bash
python ./vanDoorn/vandoorn_ohio_main.py
```
##### DiaTrend:

```bash
python ./vanDoorn/vandoorn_diatrend_main.py
```

##### T1DEXI:

```bash
python ./vanDoorn/vandoorn_t1dexi_main.py
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

#### 2.5 Rabby et al., 2021
##### OhioT1DM:
```bash
python ./Rabby_et_al_StackedLSTM/src/Ohio_Processing_LSTM.py 
bash ./Rabby_et_al_StackedLSTM/src/ohio_job.sh 
```
##### DiaTrend:

```bash
bash ./Rabby_et_al_StackedLSTM/src/diatrend_job.sh 
```

##### T1DEXI:

```bash
bash ../AccurateBG/accurate_bg/t1dexi_main.py
```



#### 2.6 Lee et al., 2021
##### OhioT1DM:

```bash
python ./GlucoseTransformer_organized/ohio_main.py
```
##### DiaTrend:

```bash
python ./GlucoseTransformer_organized/diatrend_main.py
```

##### T1DEXI:

```bash
python ../GlucoseTransformer_organized/t1dexi_main.py
```


## 📚 References

This work builds on prior studies in blood glucose prediction and reproducibility in machine learning. Key references include:

- [1] Martinsson, J., Schliep, A., Eliasson, B. et al. Blood Glucose Prediction with Variance Estimation Using Recurrent Neural Networks. J Healthc Inform Res 4, 1–18 (2020). https://doi.org/10.1007/s41666-019-00059-y
- [2] K. Li, C. Liu, T. Zhu, P. Herrero and P. Georgiou, "GluNet: A Deep Learning Framework for Accurate Glucose Forecasting," in IEEE Journal of Biomedical and Health Informatics, vol. 24, no. 2, pp. 414-423, Feb. 2020, doi: 10.1109/JBHI.2019.2931842.
- [3] van Doorn WPTM, Foreman YD, Schaper NC, Savelberg HHCM, Koster A, et al. (2021) Machine learning-based glucose prediction with use of continuous glucose and physical activity monitoring data: The Maastricht Study. PLOS ONE 16(6): e0253125. https://doi.org/10.1371/journal.pone.0253125
- [4] Deng, Y., Lu, L., Aponte, L. et al. Deep transfer learning and data augmentation improve glucose levels prediction in type 2 diabetes patients. npj Digit. Med. 4, 109 (2021). https://doi.org/10.1038/s41746-021-00480-x
- [5] Rabby, M.F., Tu, Y., Hossen, M.I. et al. Stacked LSTM based deep recurrent neural network with kalman smoothing for blood glucose prediction. BMC Med Inform Decis Mak 21, 101 (2021). https://doi.org/10.1186/s12911-021-01462-5
- [6] S. -M. Lee, D. -Y. Kim and J. Woo, "Glucose Transformer: Forecasting Glucose Level and Events of Hyperglycemia and Hypoglycemia," in IEEE Journal of Biomedical and Health Informatics, vol. 27, no. 3, pp. 1600-1611, March 2023, doi: 10.1109/JBHI.2023.3236822.


Reference Code:
- [1] Martinsson et al., 2019 : https://github.com/johnmartinsson/blood-glucose-prediction 
- [4] Deng et al., 2021: https://github.com/yixiangD/AccurateBG 

For implementation details and replication instructions, please see each method-specific folder and referenced repositories.
