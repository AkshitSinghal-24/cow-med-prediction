from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd

# Load artifacts
model = joblib.load("cow_medicine_model2.pkl")
label_encoder = joblib.load("label_encoder2.pkl")
feature_columns = joblib.load("feature_columns2.pkl")
unique_diagnoses = joblib.load("unique_diagnoses2.pkl")

app = FastAPI(title="Cow Medicine Predictor API")

# Input model allowing multiple diagnoses
class CowData(BaseModel):
    diagnosis: List[str]
    breed: str
    num_calvings: float
    age: float
    months_pregnant: float
    months_since_calving: float
    avg_lpd: float

@app.post("/predict")
def predict_medicine(data: CowData):
    input_base = data.dict()
    diagnosis_list = input_base.pop("diagnosis")

    # Validate all diagnoses
    invalid_diagnoses = [d for d in diagnosis_list if d not in unique_diagnoses]
    if invalid_diagnoses:
        raise HTTPException(status_code=400, detail=f"Unknown diagnosis(es): {', '.join(invalid_diagnoses)}")

    all_inputs = []
    for diag in diagnosis_list:
        row = input_base.copy()
        row["diagnosis"] = diag
        all_inputs.append(row)

    df = pd.DataFrame(all_inputs)
    df_encoded = pd.get_dummies(df)

    # Ensure all expected features are present
    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[feature_columns]

    # Predict for each diagnosis entry
    all_predictions = []
    probs = model.predict_proba(df_encoded)

    for i, diag in enumerate(diagnosis_list):
        prob_row = probs[i]
        top_indices = prob_row.argsort()[-3:][::-1]

        recommendations = []
        for idx in top_indices:
            med = label_encoder.inverse_transform([idx])[0]
            confidence = round(prob_row[idx] * 100, 2)
            recommendations.append({
                "medicine": med,
                "confidence": f"{confidence}%"
            })

        all_predictions.append({
            'recommendations': recommendations
        })

    return {"results": all_predictions}
