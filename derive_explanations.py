import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score
import lime.lime_tabular
import shap
import matplotlib.pyplot as plt


# Function to preprocess the data
def preprocess_data(data, target_column):
    label_encoder = LabelEncoder()
    scaler = StandardScaler()

    data['Ethnicity'] = label_encoder.fit_transform(data['Ethnicity'])
    data['Gender'] = label_encoder.fit_transform(data['Gender'])

    # Split features and target
    X = data.drop(columns=target_column)
    y = data[target_column]

    # Balance the dataset
    ros = RandomOverSampler(random_state=42)
    X, y = ros.fit_resample(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    # Standardize features
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X.columns


# Function to train the model
def train_model(X_train, y_train, params):
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


# Function to explain model using LIME
def explain_with_lime(model, X_train, X_test, feature_names, idx_to_explain):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        mode='classification'
    )
    explanation = explainer.explain_instance(
        data_row=X_test[idx_to_explain],
        predict_fn=model.predict_proba
    )
    explanation.show_in_notebook(show_table=True)
    return explanation


# Function to explain model using SHAP
def explain_with_shap(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)
    return shap_values


# Main function
def main():
    # Load the dataset
    data = pd.read_csv('task1-data.csv')
    print("Dataset Loaded:")
    print(data.head())

    # Select target and label columns
    target_column = "Severity"

    # Preprocess the data
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(data, target_column)

    # Train the model
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }
    model = train_model(X_train, y_train, params)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    # LIME Explanation
    idx_to_explain = int(input(f"Enter the index (0-{len(X_test) - 1}) of the instance to explain with LIME: "))
    print("Generating LIME explanation...")
    lime_explanation = explain_with_lime(model, X_train, X_test, feature_names, idx_to_explain)
    lime_explanation.as_pyplot_figure()
    plt.show()

    # SHAP Explanation
    print("Generating SHAP explanation...")
    explain_with_shap(model, X_test)
    plt.show()


if __name__ == "__main__":
    main()
