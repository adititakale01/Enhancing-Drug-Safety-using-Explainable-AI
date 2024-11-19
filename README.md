# Enhancing-Drug-Safety-using-Explainable-AI

This repository contains the implementation of an explainable AI based method for improving drug safety.

&nbsp;
&nbsp;

## Dataset
A dataset was provided representing a large group of patients, including their demographic information, medical conditions, medications, and health outcomes after taking the drug. The goal was to identify how well the model identifies patient groups that are at risk from the drug along with the clarity in the model’s explanations for its predictions. 

&nbsp;
&nbsp;

## Methods
- <strong>EDA</strong>: Analysed outliers, missing values, data imbalance, statistical relationships.
- <strong>Pre-Processing</strong>: Data imbalance was tackled using oversampling the minority class, and data normalization was performed.
- <strong>Model Training</strong>: 31 models were trained and tested to find best base ML model using [Lazy Predict](https://pypi.org/project/lazypredict/)
- <strong>Model Optimization</strong>: Hyperparameters of the best base model was tuned using bayesian optimization.
- <strong>Model Explanations</strong>: Model results were explained by feature importances and shap values.

&nbsp;
&nbsp;

## Results

(https://i.ibb.co/BKr7cVF/Picture1.png) 


<h3><strong> This project is done during AI Natives Hackathon 2024 by BEST Erlangen and iba AG Fürth, Bavaria. </strong></h3>
