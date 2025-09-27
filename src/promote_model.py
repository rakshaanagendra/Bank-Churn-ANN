# src/promote_model.py
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = "Churn_ANN"
version = 1   # your version from earlier

# Assign alias "prod" to version 1
client.set_registered_model_alias(
    name=model_name,
    alias="prod",
    version=version
)

print(f"✅ Assigned alias 'prod' to {model_name} v{version}")