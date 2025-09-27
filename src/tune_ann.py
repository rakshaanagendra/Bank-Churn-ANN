# src/tune_ann.py
"""
Hyperparameter tuning for ANN on Churn dataset using KerasTuner + MLflow.
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, precision_recall_curve, f1_score
import keras_tuner as kt
import mlflow
import mlflow.keras

# ---------------------------
# 1) Load preprocessed data
# ---------------------------
train_path = Path("data/processed/train.csv")
test_path = Path("data/processed/test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train = train_df.drop(columns=["Exited"]).astype("float32").values
y_train = train_df["Exited"].astype("int32").values

X_test = test_df.drop(columns=["Exited"]).astype("float32").values
y_test = test_df["Exited"].astype("int32").values

# ---------------------------
# 2) Compute class weights
# ---------------------------
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
print("Class Weights:", class_weight_dict)

# ---------------------------
# 3) Build model function
# ---------------------------
def build_model(hp):
    model = Sequential()

    # Input layer
    model.add(Dense(
        units=hp.Int("units_input", min_value=32, max_value=128, step=32),
        activation="relu",
        input_shape=(X_train.shape[1],)
    ))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float("dropout_input", 0.1, 0.5, step=0.1)))

    # Hidden layers
    for i in range(hp.Int("num_hidden_layers", 1, 3)):
        model.add(Dense(
            units=hp.Int(f"units_hidden_{i}", min_value=16, max_value=64, step=16),
            activation="relu"
        ))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float(f"dropout_hidden_{i}", 0.1, 0.5, step=0.1)))

    # Output
    model.add(Dense(1, activation="sigmoid"))

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice("lr", values=[1e-2, 1e-3, 1e-4])
        ),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    # Register batch_size as hyperparameter
    hp.Choice("batch_size", values=[16, 32, 64])

    return model

# ---------------------------
# 4) Tuner setup
# ---------------------------
tuner = kt.RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=10,
    executions_per_trial=1,
    directory="artifacts",
    project_name="churn_tuning"
)

# ---------------------------
# 5) Run tuner
# ---------------------------
tuner.search(
    X_train, y_train,
    validation_split=0.2,
    epochs=20,
    class_weight=class_weight_dict,
    verbose=1
)

# ---------------------------
# 6) Get best hyperparameters
# ---------------------------
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\nBest Hyperparameters:")
print(best_hps.values)

# ---------------------------
# 7) Train best model
# ---------------------------
best_model = tuner.hypermodel.build(best_hps)

history = best_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=best_hps.get("batch_size"),
    class_weight=class_weight_dict,
    verbose=1
)

# ---------------------------
# 8) Evaluate
# ---------------------------
test_loss, test_acc, test_auc = best_model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test AUC: {test_auc:.4f}")

y_pred_prob = best_model.predict(X_test)

# Best threshold by F1
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
best_threshold, best_f1 = 0.5, 0
for t in thresholds:
    f1 = f1_score(y_test, (y_pred_prob > t).astype(int))
    if f1 > best_f1:
        best_f1, best_threshold = f1, t

print(f"\nBest threshold by F1: {best_threshold:.2f} (F1={best_f1:.4f})")

y_pred_best = (y_pred_prob > best_threshold).astype(int)
print("\nClassification Report at best threshold:")
print(classification_report(y_test, y_pred_best, digits=4))

# ---------------------------
# 9) Save artifacts
# ---------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

best_model.save("models/best_tuned_ann.keras")

# Save training history
with open("artifacts/tuning_history.json", "w") as f:
    json.dump(history.history, f)

# Save test metrics
metrics = {"test_loss": float(test_loss),
           "test_accuracy": float(test_acc),
           "test_auc": float(test_auc),
           "best_threshold": float(best_threshold),
           "best_f1": float(best_f1)}
with open("artifacts/tuning_metrics.json", "w") as f:
    json.dump(metrics, f)

# Training curves
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("artifacts/tuning_training_curves.png")
plt.close()

# Precision–Recall tradeoff
plt.figure(figsize=(8,6))
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.xlabel("Threshold"); plt.ylabel("Score")
plt.title("Precision–Recall Tradeoff")
plt.legend(); plt.grid()
plt.savefig("artifacts/tuning_precision_recall_curve.png")
plt.close()

# ---------------------------
# 10) Log to MLflow
# ---------------------------
mlflow.set_experiment("Churn_ANN_Tuning")

with mlflow.start_run():
    # Log hyperparameters
    for param, value in best_hps.values.items():
        mlflow.log_param(param, value)

    # Log metrics
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_auc", test_auc)
    mlflow.log_metric("best_threshold", best_threshold)
    mlflow.log_metric("best_f1", best_f1)

    # Log model
    mlflow.keras.log_model(best_model, "model")

    # Log artifacts
    mlflow.log_artifact("artifacts/tuning_training_curves.png")
    mlflow.log_artifact("artifacts/tuning_precision_recall_curve.png")
    mlflow.log_artifact("artifacts/tuning_history.json")
    mlflow.log_artifact("artifacts/tuning_metrics.json")

print("\n✅ Tuning complete. Best model, metrics, and plots logged to MLflow + artifacts saved.")


from mlflow.tracking import MlflowClient

mlflow.set_experiment("Churn_ANN_Tuning")

with mlflow.start_run() as run:
    run_id = run.info.run_id   # ✅ capture run_id while run is active

    # Log hyperparameters
    for param, value in best_hps.values.items():
        mlflow.log_param(param, value)

    # Log metrics
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_auc", test_auc)
    mlflow.log_metric("best_threshold", best_threshold)
    mlflow.log_metric("best_f1", best_f1)

    # Log model
    mlflow.keras.log_model(best_model, "model")

    # Log artifacts
    mlflow.log_artifact("artifacts/tuning_training_curves.png")
    mlflow.log_artifact("artifacts/tuning_precision_recall_curve.png")
    mlflow.log_artifact("artifacts/tuning_history.json")
    mlflow.log_artifact("artifacts/tuning_metrics.json")

    # ✅ Register model here
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    model_name = "Churn_ANN"
    model_details = mlflow.register_model(model_uri, model_name)

    print(f"\n✅ Model registered as '{model_name}', version {model_details.version}")

