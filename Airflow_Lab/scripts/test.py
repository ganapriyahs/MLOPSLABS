import pandas as pd
import pickle
import sys

def feature_engineering(df):
    # AgeGroup
    df['AgeGroup'] = pd.cut(df['Age'], bins=[20, 30, 40, 50, 60, 100],
                            labels=[0, 1, 2, 3, 4], right=False).astype(int)
    # BMICategory
    bmi_mapping = {1: (0, 18.5), 2: (18.5, 24.9), 3: (25, 29.9), 4: (30, 39.9), 5: (40, 100)}
    for cat, (low, high) in bmi_mapping.items():
        df.loc[df['BMI'].between(low, high, inclusive='left'), 'BMICategory'] = cat
    df['BMICategory'] = df['BMICategory'].fillna(2).astype(int)  # fill missing BMI category as normal weight (2)

    # Glucose_Insulin_Ratio (avoid division by zero)
    df['Glucose_Insulin_Ratio'] = df.apply(lambda r: r.Glucose / r.Insulin if r.Insulin > 0 else 0, axis=1)

    # HighPregnancyCount
    df['HighPregnancyCount'] = (df['Pregnancies'] > 3).astype(int)

    return df

def write_log(log_file, lines):
    with open(log_file, 'a') as f:
        for line in lines:
            f.write(line + '\n')
        f.write('\n')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 test.py <model_path> <log_file>")
        sys.exit(1)

    model_path = sys.argv[1]
    log_file = sys.argv[2]

    with open(log_file, 'w') as f:
        f.write("Test output log:\n\n")

    with open(model_path, 'rb') as f:
        model_bundle = pickle.load(f)

    clf = model_bundle['clf']
    model_type = model_bundle['model']
    scaler = model_bundle.get('scaler', None)

    # Raw test data: basic features only - you can adjust these values
    test_data = {
        "Pregnancies": 2,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 85,
        "BMI": 28.0,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 30
    }

    df_test = pd.DataFrame([test_data])

    # Add feature engineering to test data
    df_test = feature_engineering(df_test)

    # Scale test data if scaler exists
    if scaler:
        df_scaled = scaler.transform(df_test)
    else:
        df_scaled = df_test

    prediction = clf.predict(df_scaled)[0]
    result = 'Diabetic' if prediction == 1 else 'Non-Diabetic'

    lines = [
        f"Model trained with {model_type}.",
        "Input test data (after feature engineering):"
    ] + [f"  {k}: {v}" for k, v in df_test.iloc[0].items()] + [f"Prediction: {result}"]

    write_log(log_file, lines)
