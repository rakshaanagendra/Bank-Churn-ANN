# src/train_ann.py
"""
Train an Artificial Neural Network (ANN) on the Churn dataset.
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, precision_recall_curve

# ---------------------------
# 1) Load preprocessed data
# ---------------------------
train_path = Path("data/processed/train.csv")
test_path = Path("data/processed/test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Ensure numeric arrays
X_train = train_df.drop(columns=["Exited"]).astype("float32").values
y_train = train_df["Exited"].astype("int32").values

X_test = test_df.drop(columns=["Exited"]).astype("float32").values
y_test = test_df["Exited"].astype("int32").values

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

# ---------------------------
# 2) Compute class weights
# ---------------------------
classes = np.unique(y_train)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
print("Class Weights:", class_weight_dict)


from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# ---------------------------
# 3) Define ANN architecture (improved)
# ---------------------------
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.2),

    Dense(32, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),

    Dense(16, activation="relu"),
    BatchNormalization(),

    Dense(1, activation="sigmoid")  # Output layer
])


# ---------------------------
# 4) Compile the model
# ---------------------------
model.compile(
    optimizer="adam",  # Adaptive optimizer
    loss="binary_crossentropy",  # For binary classification
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

# ---------------------------
# 5) Callbacks
# ---------------------------
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        filepath="models/best_ann.keras",
        save_best_only=True
    )
]

# ---------------------------
# 6) Train the model
# ---------------------------
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# ---------------------------
# 7) Evaluate on test set
# ---------------------------
test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# Classification report at default threshold=0.5
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\nClassification Report (threshold=0.5):")
print(classification_report(y_test, y_pred, digits=4))

# ---------------------------
# 8) Precision–Recall tradeoff curve
# ---------------------------
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

plt.figure(figsize=(8,6))
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision–Recall Tradeoff")
plt.legend()
plt.grid()
os.makedirs("artifacts", exist_ok=True)
plt.savefig("artifacts/precision_recall_curve.png")
plt.show()

# Example: Evaluate at custom threshold = 0.4
custom_threshold = 0.4
y_pred_custom = (y_pred_prob > custom_threshold).astype(int)
print(f"\nClassification Report (threshold={custom_threshold}):")
print(classification_report(y_test, y_pred_custom, digits=4))

from sklearn.metrics import f1_score

# ---------------------------
# 9) Find the best threshold automatically
# ---------------------------
best_threshold = 0.5
best_f1 = 0

for t in thresholds:
    y_pred_t = (y_pred_prob > t).astype(int)
    f1 = f1_score(y_test, y_pred_t)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"\nBest threshold by F1-score: {best_threshold:.2f} (F1={best_f1:.4f})")

# Evaluate at this best threshold
y_pred_best = (y_pred_prob > best_threshold).astype(int)
print(f"\nClassification Report (threshold={best_threshold:.2f}):")
print(classification_report(y_test, y_pred_best, digits=4))


# ---------------------------
# 10) Save model & artifacts
# ---------------------------
os.makedirs("models", exist_ok=True)

# Save final model
model.save("models/final_ann.keras")

# Save training history
history_dict = history.history
with open("artifacts/history.json", "w") as f:
    json.dump(history_dict, f)

# Save test metrics
metrics = {
    "test_loss": float(test_loss),
    "test_accuracy": float(test_acc),
    "test_auc": float(test_auc)
}
with open("artifacts/test_metrics.json", "w") as f:
    json.dump(metrics, f)

# ---------------------------
# 11) Plot training curves
# ---------------------------
plt.figure(figsize=(12,5))

# Loss curves
plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

# Accuracy curves
plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("artifacts/training_curves.png")
plt.show()

print("✅ All artifacts saved in 'models/' and 'artifacts/' folders.")
