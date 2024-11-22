# Enhancing-Drug-Safety-using-Explainable-AI

This repository contains the implementation of an explainable AI based method for improving drug safety.
**This project is done during AI Natives Hackathon 2024 by [BEST Erlangen](https://www.best-erlangen.de/) and [iba AG](https://www.iba-ag.com/de/) Fürth, Bavaria.**

&nbsp;
&nbsp;

## Dataset
A dataset was provided representing a large group of patients, including their demographic information, medical conditions, medications, and health outcomes after taking the drug. The goal was to identify how well the model identifies patient groups that are at risk from the drug along with the clarity in the model’s explanations for its predictions. 

&nbsp;
&nbsp;

## Proposed Solution/Methods
- **EDA**: Analysed outliers, missing values, data imbalance, statistical relationships.
- **Pre-Processing**: Data imbalance was tackled using oversampling the minority class, and data normalization was performed.
- **Model Training**: 31 models were trained and tested to find best base ML model using [Lazy Predict](https://pypi.org/project/lazypredict/)
- **Model Optimization**: Hyperparameters of the best base model was tuned using bayesian optimization.
- **Model Explanations**: Model results were explained by feature importances and shap values.
- **Visualization**: The training results and explanation plots were visualized in a front-end, which can also be used for training.

&nbsp;
&nbsp;

## Results/Output
<p align="center">
  <img src="https://github.com/arkanivasarkar/Enhancing-Drug-Safety-using-Explainable-AI/blob/main/Resources/frontend.png" alt="Image description" width="100%" height="10%">
</p>

&nbsp;

<p align="center">
  <img src="https://github.com/arkanivasarkar/Enhancing-Drug-Safety-using-Explainable-AI/blob/main/Resources/metrics.png" alt="Image description" width="100%" height="10%">
</p>

&nbsp;
&nbsp;



