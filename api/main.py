from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import mlflow.pyfunc
import os
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Detect CI mode
# ----------------------------
### NEW FOR CI MODE
CI_MODE = os.environ.get("CI", "false").lower() == "true"

# ----------------------------
# Input schema (raw customer JSON)
# ----------------------------
class CustomerRaw(BaseModel):
    RowNumber: int | None = None
    CustomerId: int | None = None
    Surname: str | None = None
    CreditScore: float
    Geography: str
    Gender: str
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

class CustomersBatch(BaseModel):
    customers: List[CustomerRaw]

# ----------------------------
# Config / globals
# ----------------------------
MODEL_URI = os.environ.get("MODEL_URI", "models:/Churn_ANN@prod")
TRAIN_CSV = "data/processed/train.csv"  # used to read feature names/order
model = None
scaler = None
expected_model_features = None   # canonical columns order used by the model
scaler_features = None          # columns scaler was fitted on (if available)

app = FastAPI(title="Churn Prediction API", version="1.0")

# ----------------------------
# Startup: load model, scaler, and expected feature list
# ----------------------------
@app.on_event("startup")
def load_model_and_meta():
    global model, scaler, expected_model_features, scaler_features

    ### NEW FOR CI MODE
    if CI_MODE:
        # ---- Dummy model ----
        class DummyModel:
            def predict(self, X):
                import numpy as np
                return np.array([[0.5]] * len(X))  # Always 0.5 probability

        model = DummyModel()

        # ---- Dummy scaler ----
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        # Fit on fake data just to satisfy .transform()
        scaler.fit(np.array([[0, 0], [1, 1]]))

        expected_model_features = ["CreditScore", "Age"]
        scaler_features = ["CreditScore", "Age"]

        logger.info("✅ Running in CI mode: Dummy model + scaler loaded")
        return

    # ---------------- Real model load ----------------
    try:
        logger.info(f"Loading MLflow model from {MODEL_URI}")
        model = mlflow.pyfunc.load_model(MODEL_URI)
        logger.info("Model loaded.")
    except Exception as e:
        logger.exception("Failed to load MLflow model.")
        raise RuntimeError(f"Could not load MLflow model: {e}")

    try:
        scaler = joblib.load("data/processed/scaler.joblib")
        logger.info("Scaler loaded from data/processed/scaler.joblib")
    except Exception as e:
        logger.exception("Failed to load scaler.")
        raise RuntimeError(f"Could not load scaler: {e}")

    if not os.path.exists(TRAIN_CSV):
        logger.warning(f"{TRAIN_CSV} not found. Make sure processed train.csv exists.")
        expected_model_features = None
    else:
        header_df = pd.read_csv(TRAIN_CSV, nrows=0)
        cols = list(header_df.columns)
        if "Exited" in cols:
            cols.remove("Exited")
        expected_model_features = cols
        logger.info(f"Expected model features (count={len(expected_model_features)}): {expected_model_features}")

    scaler_features = getattr(scaler, "feature_names_in_", None)
    if scaler_features is not None:
        scaler_features = list(scaler_features)
        logger.info(f"Scaler fitted on columns (count={len(scaler_features)}): {scaler_features}")
    else:
        scaler_features = expected_model_features
        logger.info("scaler.feature_names_in_ not available; using expected_model_features as scaler_features fallback.")

# ----------------------------
# Preprocessing helpers
# ----------------------------
def preprocess_and_align(customer_dict: dict) -> pd.DataFrame:
    global scaler, expected_model_features, scaler_features

    df = pd.DataFrame([customer_dict])

    for c in ("RowNumber", "CustomerId", "Surname"):
        if c in df.columns:
            df = df.drop(columns=c)

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})

    if "Geography" in df.columns:
        df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

    if scaler_features is not None:
        for col in scaler_features:
            if col not in df.columns:
                df[col] = 0

    if expected_model_features is not None:
        for col in expected_model_features:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_model_features]

    if scaler is not None and scaler_features is not None:
        scaled_arr = scaler.transform(df[scaler_features])
        df_scaled = pd.DataFrame(scaled_arr, columns=scaler_features, index=df.index)
        final = pd.DataFrame(index=df.index)
        for col in expected_model_features:
            if col in df_scaled.columns:
                final[col] = df_scaled[col]
            elif col in df.columns:
                final[col] = df[col]
            else:
                final[col] = 0
        df = final

    return df


def preprocess_batch(df: pd.DataFrame) -> pd.DataFrame:
    global scaler, expected_model_features, scaler_features

    df = df.copy()
    for c in ("RowNumber", "CustomerId", "Surname"):
        if c in df.columns:
            df = df.drop(columns=c)

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})

    if "Geography" in df.columns:
        df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

    if scaler_features is not None:
        for col in scaler_features:
            if col not in df.columns:
                df[col] = 0

    if expected_model_features is not None:
        for col in expected_model_features:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_model_features]

    if scaler is not None and scaler_features is not None:
        scaled_arr = scaler.transform(df[scaler_features])
        df_scaled = pd.DataFrame(scaled_arr, columns=scaler_features, index=df.index)
        final = pd.DataFrame(index=df.index)
        for col in expected_model_features:
            if col in df_scaled.columns:
                final[col] = df_scaled[col]
            elif col in df.columns:
                final[col] = df[col]
            else:
                final[col] = 0
        df = final

    return df

# ----------------------------
# Prediction endpoints
# ----------------------------
@app.post("/predict")
def predict(customer: CustomerRaw):
    global model
    try:
        df = preprocess_and_align(customer.model_dump())  # ✅ model_dump for Pydantic v2
        preds = model.predict(df)

        if isinstance(preds, np.ndarray):
            if preds.ndim == 2 and preds.shape[1] == 1:
                proba = float(preds[0, 0])
            elif preds.ndim == 2 and preds.shape[1] > 1:
                proba = float(preds[0, 1])
            else:
                proba = float(preds[0])
        else:
            proba = float(preds[0])

        prediction = int(proba >= 0.5)
        return {"prediction": prediction, "probability": proba}
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_batch")
def predict_batch(customers: List[CustomerRaw], return_features: Optional[bool] = False):
    global model
    try:
        df_raw = pd.DataFrame([c.model_dump() for c in customers])  # ✅ model_dump
        df_prepped = preprocess_batch(df_raw)
        preds = model.predict(df_prepped)

        preds_arr = np.array(preds)
        if preds_arr.ndim == 2 and preds_arr.shape[1] == 1:
            proba_arr = preds_arr[:, 0]
        elif preds_arr.ndim == 2 and preds_arr.shape[1] > 1:
            proba_arr = preds_arr[:, 1]
        else:
            proba_arr = preds_arr.ravel()

        result_list = []
        for idx, p in enumerate(proba_arr):
            pred_label = int(p >= 0.5)
            entry = {"prediction": pred_label, "probability": float(p)}
            if return_features:
                entry["features_vector"] = df_prepped.iloc[idx].tolist()
            result_list.append(entry)

        return {"count": len(result_list), "predictions": result_list}
    except Exception as e:
        logger.exception("Batch prediction error")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
