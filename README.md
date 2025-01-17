# Investigating the Reproducibility and Generalizability of Deep Learning Methods for Blood Glucose Prediction

## Study introduction
Paper Abstract: The growing prevalence of mobile health monitoring devices is expediting the development of machine learning methods for blood glucose prediction. However, open questions remain about the reproducibility and generalizability of predictive models in the literature. To address this gap, our study sought to investigate the reproducibility and generalizability of six state-of-the-art deep learning models for the task of blood glucose prediction using three publicly available datasets: OhioT1DM, DiaTrend, and T1DEXI. From careful re-implementation and evaluation, we found an average difference of 4.83 mg/dL in the RMSE compared with the paper-reported results. We also found that most prediction models had worse performance (i.e., higher errors) when tested on a dataset and/or subject with more glycemic variability and higher mean blood glucose values. Finally, we found that training-related factors, such as the model complexity and sampling horizon have a limited impact on the models' performance and generalizability. Insights from this study can serve as a basis for increasing the transparency on predictive methods proposed in the literature and the development of more robust models for real-world implementation.

## Code index

All of the implementations of this study are based on JupyterNotebook. The index can help researchers to find the .ipynb files that hold the reproduction code.
1. Martinsson2019: 
    - [Reproduce on OhioT1DM](./MartinssonAndvanDoorn/verify_example_paper_method.ipynb)
    - [Reproduce on DiaTrend](./MartinssonAndvanDoorn/verify_with_DiaTrend.ipynb)
    - [Reproduce on T1DEXI](./MartinssonAndvanDoorn/verify_with_T1DEXI.ipynb)

2. vanDoorn2021:
    - [Reproduce on OhioT1DM](./MartinssonAndvanDoorn/verify_method1_vanDoorn.ipynb)
    - [Reproduce on DiaTrend](./MartinssonAndvanDoorn/verify_method1_vanDoorn.ipynb)
    - [Reproduce on T1DEXI](./MartinssonAndvanDoorn/verify_method1_vanDoorn.ipynb)

3. Deng2021:
    - [Reproduce on OhioT1DM](./AccurateBG/accurate_bg/verify_method.ipynb)
    - [Reproduce on DiaTrend](./AccurateBG/accurate_bg/verify_diatrend.ipynb)
    - [Reproduce on T1DEXI](./AccurateBG/accurate_bg/verify_t1dexi.ipynb)

4. Kim2020:
    - [Reproduce on OhioT1DM](./Blood-Glucose-Prediction-LSTM/apply_on_ohiot1dm.ipynb)
    - [Reproduce on DiaTrend](./Blood-Glucose-Prediction-LSTM/apply_on_diatrend.ipynb)
    - [Reproduce on T1DEXI](./Blood-Glucose-Prediction-LSTM/apply_on_t1dexi.ipynb)

5. Li2020:
    - [Reproduce on OhioT1DM](./pytorch-wavenet/Final_version_replicate_Glunet.ipynb)
    - [Reproduce on T1DEXI](./GluNet/t1dexi_GluNet.ipynb)

6. Rabby2021:
    - [Reproduce on OhioT1DM](./Stacked%20LSTM/final_replicate_StackedLSTM.ipynb)
    - [Reproduce on T1DEXI](./Stacked%20LSTM/verify_with_t1dexi.ipynb)

## Dataset sources:
There three datasets are all open-sourced
1. OhioT1DM: Marling C, Bunescu R. The OhioT1DM Dataset for Blood Glucose Level Prediction: Update 2020. CEUR Workshop Proc. 2020 Sep;2675:71-74. PMID: 33584164; PMCID: PMC7881904.
2. DiaTrend: Prioleau, T., Bartolome, A., Comi, R. et al. DiaTrend: A dataset from advanced diabetes technology to enable development of novel analytic solutions. Sci Data 10, 556 (2023). https://doi.org/10.1038/s41597-023-02469-5
3. T1DEXI: Riddell MC, Li Z, Gal RL, Calhoun P, Jacobs PG, Clements MA, Martin CK, Doyle Iii FJ, Patton SR, Castle JR, Gillingham MB, Beck RW, Rickels MR; T1DEXI Study Group. Examining the Acute Glycemic Effects of Different Types of Structured Exercise Sessions in Type 1 Diabetes in a Real-World Setting: The Type 1 Diabetes and Exercise Initiative (T1DEXI). Diabetes Care. 2023 Apr 1;46(4):704-713. doi: 10.2337/dc22-1721. PMID: 36795053; PMCID: PMC10090894.



## Code Reference Claim
This study investigate the reproducibility and generalizability of the state-of-the-art deep learning blood glucose prediction methods. Some code of this study partially relies on the provided code from the studies discussed in the paper. The studies include：
1. Martinsson J, Schliep A, Eliasson B, Mogren O. Blood Glucose Prediction with Variance Estimation Using Recurrent Neural Networks. J Healthc Inform Res. 2019 Dec 1;4(1):1-18. doi: 10.1007/s41666-019-00059-y. PMID: 35415439; PMCID: PMC8982803. Code:  https://github.com/johnmartinsson/blood-glucose-prediction. 

2.  Deng, Y., Lu, L., Aponte, L. et al. Deep transfer learning and data augmentation improve glucose levels prediction in type 2 diabetes patients. npj Digit. Med. 4, 109 (2021). https://doi.org/10.1038/s41746-021-00480-x. Code: https://github.com/yixiangD/AccurateBG.
3. Kim D-Y, Choi D-S, Kim J, Chun SW, Gil H-W, Cho N-J, Kang AR, Woo J. Developing an Individual Glucose Prediction Model Using Recurrent Neural Network. Sensors. 2020; 20(22):6460. https://doi.org/10.3390/s20226460. Code: https://github.com/dongsikchoi/Blood-Glucose-Prediction-LSTM

The implementation of GluNet referred the Github repo:
https://github.com/vincentherrmann/pytorch-wavenet/tree/master and https://github.com/ibab/tensorflow-wavenet/tree/master .

