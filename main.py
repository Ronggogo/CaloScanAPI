from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model yang sudah dilatih
model = joblib.load("rfr_model.joblib")

# Endpoint testing
@app.get("/")
def root():
    return {"message": "Server berjalan"}

# Schema input (format data yang harus dikirim)
class PredictionRequest(BaseModel):
    class_id: int
    ratio: float

Calorie_per_100gram = {
    0: 167,  # AyamBakar
    1: 260,  # AyamBumbu
    2: 410,  # Dendeng
    3: 105,  # LeleGoreng
    4: 129,  # Nasi
    5: 128,  # NilaGoreng
    6: 271,  # TahuGoreng
    7: 154,  # TelurRebus
    8: 225,  # TempeGoreng
    9: 0,  # piring
    10:0   # sendok
}

# Endpoint predict ()
@app.post("/predict")
def predict(data: PredictionRequest):
    
    features = np.array([[
        data.class_id,
        data.ratio
    ]])

    prediction = model.predict(features)
    estimated_weight = float(prediction[0])

    return {
        "estimated_weight": estimated_weight,
        "estimated_calories": float(estimated_weight*(Calorie_per_100gram[data.class_id]/100))
    }