# Deep Learning for Blood Glucose Prediction: Investigating Reproducibility

This repository contains the associated supplementary material and implementation/replication code for the paper:  
**"Deep Learning for Blood Glucose Prediction: Investigating Reproducibility and Factors Affecting Differential Performance."**

## Abstract: 
Blood glucose prediction is a fundamental part of advanced technology that promises to improve diabetes outcomes. However, a critical gap exists around understanding the reproducibility of state-of-the-art glucose prediction methods. To bridge this gap, we evaluated 60 deep learning (DL)-based glucose prediction papers published between 2018–2025. We found that code availability and use of multiple datasets are amongst the top challenges to reproducibility. Additionally, we replicated six representative models using three publicly available datasets: OhioT1DM, DiaTrend, and T1DEXI. Our results show good reproducibility of DL methods when using the same code (where available) and same evaluation dataset. However, we found poor conceptual reproducibility across datasets with significantly different diabetes management. Further analyses revealed that the accuracy of blood glucose prediction methods was significantly associated with individual diabetes management and sex/gender. All models had higher prediction errors for individuals with worse glycemic control and for female subgroups compared to males.


---

## 📂 Repository Structure
In this study, we evaluated 60 DL-based glucose prediction papers (2018–2025) and assessed them against core reproducibility criteria (see Literature review summary below). Furthermore, we reimplemented and assessed six representative deep learning models using three publicly available datasets: OhioT1DM, DiaTrend, and T1DEXI. 

As a result, this repository contains the following:
- `2019Martinsson_et_al_LSTM/` – The implementation of Martinsson et al., 2019
- `2020Li_et_al_GluNet/` – The implementation of Li et al., 2021
- `2021Deng_et_al_CNN/` – The implementation of Deng et al., 2021
- `2021Rabby_et_al_StackedLSTM/` – The implementation of Rabby et al., 2021
- `2021vanDoorn_et_al_LSTM/` - The implementation of vanDoorn et al., 2021
- `2023Lee_et_al_GlucoseTransformer/` – The implementation of Lee et al., 2023
- `Baseline/` – The implementation of baseline method. 
- `dataset_preprocessing/` - Includes data preprocesssing script, subset patient id, and fold split (For Method Martinsson et al, vanDoorn et al, and Deng et al). 
- `Literature_review_summary/` - Summarize from the lierature and code for Figure 1.
- `ResultFigures/` - Scripts for the Figure 2-5 in the paper.
- `License`
- `README.md`
- `requirement.txt`
---

## Literature review summary

We reviewed 60 peer-reviewed papers published between 2018 and 2025 that introduce and evaluate deep learning models for blood glucose prediction. Drawing on prior work, we assessed each study against key factors impacting reproducibility, including: (1) availability of code and data, (2) completeness of model reporting (e.g., hyperparameters and tuning), and (3) use of standardized evaluation protocols (e.g., prediction horizon and metrics). All reviewed papers appeared in engineering or interdisciplinary journals and conferences. A summary table of our literature review is available here: [literature review table](Literature_review_summary/Submit%20version%20of%20literature%20review%20table.xlsx).

## 📋 Requirements

```bash
# Create and activate a virtual environment (optional)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```


## 🚀 How to Run

### 1. Dataset

Please refer to the data preparation instructions for each dataset below.  
**Note:** Due to licensing restrictions and data use agreements, raw datasets are not included in this repository. However, all datasets used in this study are publicly available and can be requested by interested researchers. 

- **OhioT1DM**: Request access to OhioT1DM via this link: https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/OhioT1DM-dataset.html
- **DiaTrend**: Request access to DiaTrend via this link: [https://www.synapse.org/Synapse:syn38187184/wiki/619490](https://doi.org/10.7303/syn38187184). 
- **T1DEXI**: Request access to T1DEXI via this link: [https://search.vivli.org/doiLanding/studies/PR00008428/isLanding](https://doi.org/10.25934/PR00008428).

### 2. Data preprocessing
After accessing these three datasets, please follow these instructions for data sanitation:

Create a folder on the same level with the pulled "ReproducibilityStudy_DL_BGPrediction" folder. Name it as "datasets"

- **OhioT1DM**: Please directly put the dataset folder into this directory: `../datasets/[OhioT1DM]`. Also copy all the xml files from folder 2018 and 2020 to another folder "both" in the same folder as 2018 and 2020. 
- **DiaTrend**: Please following [README.md](./dataset_preprocessing/README,md) for data preprocessing. And please put the generated `fold[N]_training`, `fold[N]_test` and `processed_cgm_data_Subject[N].csv` under the same directory. 
- **T1DEXI**: Please following this [subset list](./dataset_preprocessing/selected_t1dexi.txt) and [README.md](./dataset_preprocessing/README.md) for data preprocessing. And please put the generated `fold[N]_training`, `fold[N]_test` and `[N].csv` under the same directory. 

Please feel free to manually drag and split the folds. [This table](dataset_preprocessing/fold_split.csv) descibes how the fold split in DiaTrend and T1DEXI.

### 3. Run the Models

Each dataset has a dedicated main script to train and evaluate all six methods. And the parameters are all based on the version provided by the original studies. 

#### Martinsson et al., 2019 [1]
Please refer this [README.md](./2019Martinsson_et_al_LSTM/README.md) for replication details. 

#### Li et al., 2020 [2]
Please refer this [README.md](./2020Li_et_al_GluNet/README.md) for replication details. 

#### van Doorn et al., 2021 [3]
Please refer this [README.md](./2021vanDoorn_et_al_LSTM/README.md) for replication details. 

#### Deng et al., 2021 [4]
Please refer this [README.md](./2021Deng_et_al_CNN/README.md) for replication details.

#### Rabby et al., 2021 [5]
Please refer this [README.md](./2021Rabby_et_al_StackedLSTM/README.md) for replication details.

#### Lee et al., 2021 [6]
Please refer this [README.md](./2023Lee_et_al_GlucoseTransformer/README.md) for replication details.

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
