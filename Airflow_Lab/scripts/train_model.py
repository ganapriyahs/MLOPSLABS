import pandas as pd
import sys
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

input_path = sys.argv[1]
model_path = sys.argv[2]
best_rf_path = sys.argv[3]

df = pd.read_csv(input_path)
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Baseline logistic regression
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_sc, y_train)
with open(model_path, 'wb') as f:
    pickle.dump({'model':'logreg','clf':logreg,'scaler':scaler}, f)

# Fine-tuned random forest
rf = RandomForestClassifier(random_state=42)
param_dist = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 8],
    'min_samples_leaf': [1, 2, 4]
}
rf_search = RandomizedSearchCV(rf, param_dist, n_iter=10, cv=3, scoring='accuracy', random_state=42, n_jobs=-1)
rf_search.fit(X_train_sc, y_train)
best_rf = rf_search.best_estimator_
with open(best_rf_path, 'wb') as f:
    pickle.dump({'model': 'rf', 'clf': best_rf, 'scaler': scaler}, f)
