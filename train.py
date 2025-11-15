import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight

from utils import preprocess

import pickle

file_path = 'data/diabetes_prediction_dataset.csv'
numeric_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'bmi_per_age', 'glucose_HbA1c_ratio']
categorical_features = ['smoking_history', 'gender', 'hypertension', 'heart_disease', 'is_elderly', 'is_obese']

# Load Data
def load_data(file_path):
    df = pd.read_csv(file_path)


    df = df[df['gender'] != 'Other'].copy()
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

    # Replace 'No Info' with 'unknown' in smoking_history
    df['smoking_history'] = df['smoking_history'].replace('No Info', 'unknown')

    # Derived features (these are okay -- they are deterministic transformations of existing features)
    df['bmi_per_age'] = df['bmi'] / (df['age'].replace(0, np.nan))
    df['glucose_HbA1c_ratio'] = df['blood_glucose_level'] / (df['HbA1c_level'].replace(0, np.nan))
    df['is_elderly'] = (df['age'] > 60).astype(int)
    df['is_obese'] = (df['bmi'] >= 30).astype(int)

    # Fill any infinite / NaN results from divisions
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df


def load_and_preprocess_data(file_path):
    df = load_data(file_path)
    

    X = df[numeric_features + categorical_features].copy()
    y = df['diabetes'].astype(int).values

    # Stratified split to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )

    # Then proceed with your current SMOTE approach
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


def train_model(file_path):
    X_train_processed, X_test_processed, y_train, y_test, preprocessor = load_and_preprocess_data(file_path)


    gb_balanced = GradientBoostingClassifier(random_state=42)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    sample_weights = class_weights[y_train]
    gb_balanced.fit(X_train_processed, y_train, sample_weight=sample_weights)

    return gb_balanced, preprocessor


def save_model(file_name, pipeline):
    with open(file_name, 'wb') as f_out:
        pickle.dump(pipeline, f_out)


gb_balanced, preprocessor = train_model(file_path)
fe_transformer = FunctionTransformer(preprocess, validate=False)
model_file = 'model_final.bin'

pipeline = Pipeline([
    ('fe', fe_transformer),
    ('pre', preprocessor),
    ('model', gb_balanced) 
])

save_model(model_file, pipeline)



