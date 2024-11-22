import streamlit as st
from streamlit_extras.let_it_rain import rain
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt

# Initialize Streamlit page
st.set_page_config(page_title="ExplainablePredictor", layout="wide")
rain(emoji="💊", font_size=54, falling_speed=5, animation_length=0.5)

# Sidebar for data upload
st.sidebar.title("⚕️ Explainable Drug Safety Predictor")
st.sidebar.caption("Upload Data for Training")
uploaded_file = st.sidebar.file_uploader("Upload Data", type=[".csv", ".json"])

# Helper Functions
def preprocess_data(data, targets):
    """Preprocess input data by encoding categorical variables and scaling."""
    encoder = LabelEncoder()
    scaler = StandardScaler()

    data["Ethnicity"] = encoder.fit_transform(data["Ethnicity"])
    data["Gender"] = encoder.fit_transform(data["Gender"])

    features = data.drop(columns=targets)
    return features, scaler

def train_model(features, target, params):
    """Train a RandomForest model with oversampling."""
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(features, target)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, stratify=y_resampled, test_size=0.3, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred

def display_results(tab, y_test, y_pred, model, features):
    """Display model performance metrics and feature importance."""
    with tab:
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        st.write(f"ROC AUC: {roc_auc_score(y_test, y_pred, multi_class='ovr'):.2f}")
        st.write(f"F1 Score: {f1_score(y_test, y_pred, average='macro'):.2f}")
        st.write(f"Precision: {precision_score(y_test, y_pred, average='macro'):.2f}")
        st.write(f"Recall: {recall_score(y_test, y_pred, average='macro'):.2f}")

def display_feature_importance(tab, model, feature_names):
    """Display feature importance in a bar chart."""
    with tab:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        st.write("Feature ranking:")
        for i in range(min(5, len(indices))):
            st.write(f"{i + 1}. {feature_names[indices[i]]} ({importances[indices[i]]:.4f})")

        fig, ax = plt.subplots(figsize=(8, 8))
        plt.bar(range(len(feature_names)), importances[indices], align="center")
        plt.xticks(range(len(feature_names)), indices)
        plt.xlabel("Feature Index")
        plt.ylabel("Importance Score")
        st.pyplot(fig)

# Main logic
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("DATA:")
    st.write(data)

    target_labels = ["Dizziness", "Fatigue", "Hypoglycemia", "Palpitations", 
                     "Confusion", "Fainting", "Severity"]
    features, scaler = preprocess_data(data, target_labels)

    # Parameters for RandomForestClassifier
    rf_params = [
        {"criterion": "entropy", "max_depth": 20, "n_estimators": 628},
        {"criterion": "gini", "max_depth": 21, "n_estimators": 554},
        {"criterion": "gini", "max_depth": 16, "n_estimators": 788},
        {"criterion": "gini", "max_depth": 26, "n_estimators": 589},
        {"criterion": "gini", "max_depth": 25, "n_estimators": 775},
        {"criterion": "gini", "max_depth": 54, "n_estimators": 276},
        {"criterion": "entropy", "max_depth": 21, "n_estimators": 998},
    ]

    columns = st.columns(2)
    for i, target in enumerate(target_labels):
        col = columns[i % 2]
        col.header(target)
        tab_result, tab_explain = col.tabs(["RESULTS", "EXPLANATIONS"])

        # Train model
        _, target_values = preprocess_data(data, target_labels)
        model, X_test, y_test, y_pred = train_model(features, data[target], rf_params[i])

        # Display results
        display_results(tab_result, y_test, y_pred, model, features.columns)
        display_feature_importance(tab_explain, model, features.columns)
