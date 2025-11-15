import pandas as pd
import numpy as np

def preprocess(p):
    columns = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'bmi_per_age',
               'glucose_HbA1c_ratio', 'smoking_history', 'gender', 'hypertension',
               'heart_disease', 'is_elderly', 'is_obese']
    
    df = pd.DataFrame([p])
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    df['smoking_history'] = df['smoking_history'].replace('No Info', 'unknown')
    df['bmi_per_age'] = df['bmi'] / (df['age'] if p['age'] != 0 else np.nan)
    df['glucose_HbA1c_ratio'] = df['blood_glucose_level'] / (df['HbA1c_level'] if p['HbA1c_level'] != 0 else np.nan)
    df['is_elderly'] = (df['age'] > 60).astype(int)
    df['is_obese'] = (df['bmi'] >= 30).astype(int)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df[columns]
