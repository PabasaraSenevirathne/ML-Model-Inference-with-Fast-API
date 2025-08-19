from fastapi import FastAPI, HTTPException # type: ignore
from pydantic import BaseModel # type: ignore
import joblib # type: ignore
import numpy as np # type: ignore

# Load the trained model
try:
    model = joblib.load("model.pkl")
except Exception as e:
    raise RuntimeError("Model loading failed:", e)

app = FastAPI(title="House Price Predictor", description="Predict house prices based on input features")

# Define input schema
class PredictionInput(BaseModel):
    square_footage: float
    bedrooms: int
    bathrooms: float
    house_age: int

class PredictionOutput(BaseModel):
    prediction: float

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "House Price Prediction API is running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        features = np.array([[input_data.square_footage,
                              input_data.bedrooms,
                              input_data.bathrooms,
                              input_data.house_age]])
        prediction = model.predict(features)[0]
        return PredictionOutput(prediction=round(prediction, 2))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info")
def model_info():
    return {
        "model_type": "LinearRegression",
        "problem_type": "Regression",
        "features": ["square_footage", "bedrooms", "bathrooms", "house_age"]
    }
