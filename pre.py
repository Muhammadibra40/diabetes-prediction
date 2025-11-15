import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import FunctionTransformer
from fastapi import FastAPI
import uvicorn


model_file = 'model.bin'

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


fe_transformer = FunctionTransformer(preprocess, validate=False)

with open(model_file, 'rb') as f_in:
    pipeline = pickle.load(f_in)




patient= {  "gender":"Male",
            "age":70,
            "hypertension":1,
            "heart_disease":1,
            "smoking_history":"former",
            "bmi":40,
            "HbA1c_level":5.5,
            "blood_glucose_level":100}


prediction = pipeline.predict_proba(patient)[0, 1]
print(prediction)






app = FastAPI(title="patient-diabetes-prediction")

with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_single(customer):
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)


@app.post("/predict")
def predict(customer):
    prob = predict_single(customer)

    return {
        "churn_probability": prob,
        "churn": bool(prob >= 0.5)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)