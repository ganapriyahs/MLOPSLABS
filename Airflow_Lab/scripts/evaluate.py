import pandas as pd
import sys
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

input_path = sys.argv[1]
logreg_path = sys.argv[2]
rf_path = sys.argv[3]
report_path = sys.argv[4]

df = pd.read_csv(input_path)
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']

with open(logreg_path, 'rb') as f:
    logreg_bundle = pickle.load(f)
with open(rf_path, 'rb') as f:
    rf_bundle = pickle.load(f)

X_scaled = logreg_bundle['scaler'].transform(X)

logreg_preds = logreg_bundle['clf'].predict(X_scaled)
rf_preds = rf_bundle['clf'].predict(X_scaled)

logreg_acc = accuracy_score(y, logreg_preds)
rf_acc = accuracy_score(y, rf_preds)

with open(report_path, 'w') as f:
    f.write("Baseline Logistic Regression Accuracy: {:.4f}\n".format(logreg_acc))
    f.write("Confusion Matrix:\n{}\n".format(confusion_matrix(y, logreg_preds)))
    f.write("Classification Report:\n{}\n".format(classification_report(y, logreg_preds)))
    f.write("\n")
    f.write("Fine-tuned Random Forest Accuracy: {:.4f}\n".format(rf_acc))
    f.write("Confusion Matrix:\n{}\n".format(confusion_matrix(y, rf_preds)))
    f.write("Classification Report:\n{}\n".format(classification_report(y, rf_preds)))
