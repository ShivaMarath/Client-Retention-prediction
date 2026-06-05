from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Customer Churn Prediction API")

# Mount the static directory to serve the frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Load the model and scaler
try:
    model = joblib.load(os.path.join(os.path.dirname(__file__), "../models/xgboost_churn_model.pkl"))
    scaler = joblib.load(os.path.join(os.path.dirname(__file__), "../models/scaler.pkl"))
except Exception as e:
    print(f"Error loading model/scaler. Did you run train_model.py first? Error: {e}")
    model, scaler = None, None

class CustomerData(BaseModel):
    tenure: int
    monthly_charges: float
    total_charges: float
    contract_type: int  # 0: Month-to-month, 1: One year, 2: Two year

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.post("/predict")
async def predict_churn(data: CustomerData):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model is not loaded. Train the model first.")
        
    # Prepare data for prediction
    features = np.array([[
        data.tenure,
        data.monthly_charges,
        data.total_charges,
        data.contract_type
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    
    return {
        "churn_prediction": bool(prediction),
        "churn_probability": float(probability)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
