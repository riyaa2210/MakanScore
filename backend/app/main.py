from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import tensorflow as tf
import json
import numpy as np
import pandas as pd

# --------------------------
# App Initialization
# --------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# Load model and preprocessors
# --------------------------
model = tf.keras.models.load_model("app/models/house_model.keras")
preprocessor = joblib.load("app/models/preprocessor.joblib")

with open("app/models/feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# --------------------------
# Request Schema
# --------------------------
class HouseFeatures(BaseModel):
    features: dict

# --------------------------
# Routes
# --------------------------
@app.get("/")
def root():
    return {"message": "Indian House Price Prediction API. Use POST /predict with JSON {'features': {...}}"}

@app.post("/predict")
def predict(data: HouseFeatures):
    try:
        # Convert input dict to DataFrame
        df = pd.DataFrame([data.features])

        # Arrange columns in correct order (missing columns filled with 0)
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_columns]

        # Preprocess
        X_processed = preprocessor.transform(df)

        # Predict
        prediction = model.predict(X_processed)
        predicted_price = np.round(prediction[0][0], 2)

        return {"predicted_price": float(predicted_price)}

    except Exception as e:
        return {"error": str(e)}
