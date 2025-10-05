import pandas as pd
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]

df = pd.read_csv(input_path)
# Replace 0s with median for selected columns
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    df[col] = df[col].replace(0, df[col].median())

# Feature engineering
df['AgeGroup'] = pd.cut(df['Age'], bins=[20, 30, 40, 50, 60, 100], labels=['20-30', '30-40', '40-50', '50-60', '60+'])
df['BMICategory'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
df['Glucose_Insulin_Ratio'] = df['Glucose'] / (df['Insulin'] + 1)
df['HighPregnancyCount'] = (df['Pregnancies'] > 3).astype(int)
df['AgeGroup'] = df['AgeGroup'].cat.codes
df['BMICategory'] = df['BMICategory'].cat.codes

df.to_csv(output_path, index=False)
