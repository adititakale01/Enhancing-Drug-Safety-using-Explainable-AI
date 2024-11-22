import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from lazypredict.Supervised import LazyClassifier



# Function to test many ML algorithms at once
def test_lazyClassifier(X_train, y_train, X_test, y_test):
  clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
  models, _ = clf.fit(X_train, X_test, y_train, y_test)
  print(models)


# Read data
data = pd.read_csv('task1-data.csv')


# Label encoding to 'Ethnicity' and 'Gender'
label_encoder = LabelEncoder()
scaler = StandardScaler()
data['Ethnicity'] = label_encoder.fit_transform(data['Ethnicity'])
data['Gender'] = label_encoder.fit_transform(data['Gender'])
#display(train)


# target columns
targets = ["Dizziness", "Fatigue", "Hypoglycemia", "Palpitations", "Confusion", "Fainting", 'Severity']


# loop through and find best algorithms for each target
for lbl in targets:
  print(f'For {lbl}:')
  print("_________________________________________________________________")

  X = data.drop(columns=targets)
  y = data[lbl]  # Target column
  ros = RandomOverSampler(random_state=42)
  X, y = ros.fit_resample(X, y)
  X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = 0.3, random_state = 42)
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  test_lazyClassifier(X_train, y_train, X_test, y_test)

