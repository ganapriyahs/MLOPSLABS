import os
from sklearn.tree import DecisionTreeRegressor
import joblib
from data import load_data, split_data

def fit_model(X_train, y_train):
    """
    Train a Decision Tree Regressor and save the model to a file.
    """
    dt_regressor = DecisionTreeRegressor(max_depth=5, random_state=12)
    dt_regressor.fit(X_train, y_train)

    model_dir = os.path.join(os.path.dirname(__file__), "..", "model")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "housing_model.pkl")
    joblib.dump(dt_regressor, model_path)
    print(f"✅ Model saved at {model_path}")

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    fit_model(X_train, y_train)
