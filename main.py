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

    # Validate diagnoses
    invalid_diagnoses = [d for d in diagnosis_list if d not in unique_diagnoses]
    if invalid_diagnoses:
        raise HTTPException(status_code=400, detail=f"Unknown diagnosis(es): {', '.join(invalid_diagnoses)}")


    # Create input rows for each diagnosis
    rows = []
    for diag in diagnosis_list:
        row = input_base.copy()
        row["diagnosis"] = diag
        rows.append(row)

    user_df = pd.DataFrame(rows)

    # Convert numeric fields
    numeric_fields = ['num_calvings', 'age', 'months_pregnant', 'months_since_calving', 'avg_lpd']
    user_df[numeric_fields] = user_df[numeric_fields].apply(pd.to_numeric, errors='coerce')

    # One-hot encode
    encoded_df = pd.get_dummies(user_df)

    # Add missing columns
    for col in feature_columns:
        if col not in encoded_df.columns:
            encoded_df[col] = 0
    encoded_df = encoded_df[feature_columns]

    # Predict probabilities
    probs = model.predict_proba(encoded_df)

    # Collect and sort top predictions globally
    all_recommendations = []
    for i, diag in enumerate(diagnosis_list):
        for idx, prob in enumerate(probs[i]):
            all_recommendations.append({
                "diagnosis": diag,
                "med_idx": idx,
                "confidence": prob
            })
    print(f"Total recommendations before filtering: {all_recommendations}")
    # Sort and filter top unique medicines
    all_recommendations.sort(key=lambda x: x['confidence'], reverse=True)
    seen_meds = set()
    final_results = []

    for item in all_recommendations:
        med = label_encoder.inverse_transform([item["med_idx"]])[0]
        if med in seen_meds:
            continue
        seen_meds.add(med)

        final_results.append({
            "medicine": med,
            "confidence": f"{round(item['confidence'] * 100, 2)}%"
        })

        if len(final_results) >= 4:  # Limit to top 4 globally
            break

    return {"recommendations": final_results}