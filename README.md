# 📊 Bank Churn Prediction API

A **FastAPI** service that predicts customer churn (whether a customer is likely to leave the bank) using a tuned **Artificial Neural Network (ANN)** model.

The project demonstrates an **end-to-end MLOps pipeline**:
- Model training & experiment tracking with **MLflow**
- Model versioning with **MLflow Model Registry**
- Reproducibility with **Git + DVC**
- Testing & CI/CD with **GitHub Actions**
- Containerization with **Docker**
- Deployment on **Render (via GitHub Container Registry - GHCR)**


## 🚀 Features

- **REST API Endpoints**
  - `POST /predict` → Single customer prediction  
  - `POST /predict_batch` → Batch predictions  
  - `GET /health` → Service health check  
  - `GET /docs` → Interactive API docs (Swagger UI)

- **Pipeline**
  - Preprocessing: drop irrelevant IDs, encode categorical features, scale numerical ones
  - ANN model tuned with **KerasTuner**
  - Metrics, hyperparameters, and artifacts logged with **MLflow**
  - Best model registered as **`Churn_ANN`** in MLflow Model Registry

- **DevOps**
  - Automated testing with **pytest**
  - **CI/CD** pipeline with GitHub Actions
  - Docker image published to **GHCR**
  - Deployed as a live API with **Render**


## ⚡ Quickstart (Local)

### 1. **Clone the repo**
```bash
   git clone https://github.com/rakshaanagndra/Bank-Churn-ANN.git
   cd Bank-Churn-ANN
```

### 2. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the API
```bash
uvicorn api.main:app --reload --port 8000
```

## Example request for single prediction
POST /predict
```json
{
  "CreditScore": 619,
  "Geography": "France",
  "Gender": "Female",
  "Age": 42,
  "Tenure": 2,
  "Balance": 0.0,
  "NumOfProducts": 1,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 101348.88
}
```
  <br /><br />

## 🐳 Run with Docker
1. **Build image**
   ```bash
   docker build -t churn-api .
   ```

2. **Run container**
   ```bash
   docker run -p 8000:8000 churn-api
   ```
  <br /><br />
   
## 🌐 Live Demo (Render)
Deployed at
👉 https://churn-api.onrender.com/docs
- Free instance → may take ~30s to spin up on first request
- Currently running with a DummyModel in CI_MODE=true for demo
  <br /><br />

## 📊 Results

### Training Curves
![Training vs Validation Accuracy](artifacts/tuning_training_curves.png)

### Precision–Recall Tradeoff
![Precision–Recall Curve](artifacts/tuning_precision_recall_curve.png)
  <br /><br />

## 🤖 Model Training

- Model is tuned with KerasTuner (tune_ann.py).
- Metrics, hyperparameters, and artifacts are logged with MLflow.
- Best model is registered in the MLflow Model Registry as Churn_ANN.
  <br /><br />

## 🔧 CI/CD Workflow
- Github actions
  - Run tests with pytest
  - Build and push docker image to GHCR
    ```bash
    ghcr.io/rakshaanagndra/bank-churn-ann/churn-api:latest
    ```
  <br /><br />
  
## 📜 License
MIT License © 2025

