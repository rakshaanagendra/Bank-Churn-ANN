import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow

# ----------------------------------------------------
# Logging setup
# ----------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api.main")

# ----------------------------------------------------
# FastAPI app
# ----------------------------------------------------
app = FastAPI(title="Churn Prediction API", version="1.0")

# ----------------------------------------------------
# Request/Response Schemas
# ----------------------------------------------------
class CustomerFeatures(BaseModel):
    CreditScore: float
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

class PredictionResponse(BaseModel):
    prediction: int

# ----------------------------------------------------
# Model Loading Function
# ----------------------------------------------------
def load_model_and_meta():
    ci_mode = os.getenv("CI_MODE", "false").lower() == "true"

    if ci_mode:
        logger.info("✅ CI_MODE enabled → using DummyModel instead of MLflow model")

        class DummyModel:
            def predict(self, X):
                # Always predict 0 for all samples
                return [0] * len(X)

        return DummyModel(), None

    # Normal mode → load from MLflow registry
    MODEL_URI = "models:/Churn_ANN@prod"
    try:
        logger.info(f"Loading MLflow model from {MODEL_URI}")
        model = mlflow.pyfunc.load_model(MODEL_URI)
        return model, None
    except Exception as e:
        logger.error("❌ Failed to load MLflow model.")
        raise RuntimeError(f"Could not load MLflow model: {e}")

# ----------------------------------------------------
# Global model at startup
# ----------------------------------------------------
model, _ = load_model_and_meta()

# ----------------------------------------------------
# Routes
# ----------------------------------------------------
@app.get("/")
def root():
    return {"message": "Churn Prediction API is running!"}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    try:
        # Convert request to 2D list for model
        input_data = [[
            features.CreditScore,
            features.Geography,
            features.Gender,
            features.Age,
            features.Tenure,
            features.Balance,
            features.NumOfProducts,
            features.HasCrCard,
            features.IsActiveMember,
            features.EstimatedSalary
        ]]

        preds = model.predict(input_data)
        prediction = int(preds[0])
        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
