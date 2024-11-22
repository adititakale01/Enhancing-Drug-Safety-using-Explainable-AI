import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
import optuna
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
)



# Defining objective function for hyperparameter tuning
def objective(trial):
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    max_depth = trial.suggest_int("max_depth", 2, 100, log=True)
    n_estimators = trial.suggest_int("n_estimators", 1,1000)
    min_samples_split = trial.suggest_int("min_samples_split",2,10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf",1,5)


    model = RandomForestClassifier(criterion =criterion,
            max_depth=max_depth,
            n_estimators=n_estimators,
            min_samples_split = min_samples_split,
            min_samples_leaf = min_samples_leaf

        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # metric  to optimize
    score = roc_auc_score(y_test, y_pred, multi_class='ovr')
    return score



# Read data
data = pd.read_csv('task1-data.csv')


# Label encoding to 'Ethnicity' and 'Gender'
label_encoder = LabelEncoder()
scaler = StandardScaler()
data['Ethnicity'] = label_encoder.fit_transform(data['Ethnicity'])
data['Gender'] = label_encoder.fit_transform(data['Gender'])
#display(train)


# Target columns
targets = ["Dizziness", "Fatigue", "Hypoglycemia", "Palpitations", "Confusion", "Fainting", 'Severity']


# Loop through each target and find hyperparameters for the models
for i,lbl in enumerate(targets):
  print(f'For {lbl}:')
  print("_________________________________________________________________")

  X = data.drop(columns=targets)
  y = data[lbl]  # Target column
  ros = RandomOverSampler(random_state=42)
  X, y = ros.fit_resample(X, y)
  X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = 0.3, random_state = 42)
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)
  study = optuna.create_study(direction="maximize")
  study.optimize(objective, n_trials=200)
  trial = study.best_trial
  print(f"{i}________###################################################________")
  print('roc: {}'.format(trial.value))
  print("Best hyperparameters: {}".format(trial.params))

