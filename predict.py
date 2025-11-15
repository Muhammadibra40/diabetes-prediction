import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import FunctionTransformer
from fastapi import FastAPI
import uvicorn
from typing import Dict, Any
from pydantic import BaseModel, Field, conint, confloat
from enum import Enum
from utils import preprocess

model_file = 'model_final.bin'


fe_transformer = FunctionTransformer(preprocess, validate=False)

with open(model_file, 'rb') as f_in:
    pipeline = pickle.load(f_in)




app = FastAPI(title="patient-diabetes-prediction")

def predict_single(patient):
    result = pipeline.predict_proba(patient)[0, 1]
    return float(result)


@app.post("/predict")
def predict(patient: Dict[str, Any]):
    prob = predict_single(patient)

    return {
        "diabetes_probability": prob,
        "diabetes": bool(prob >= 0.5)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)