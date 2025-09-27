# src/list_models.py
from mlflow.tracking import MlflowClient

client = MlflowClient()

for rm in client.search_registered_models():
    print("Registered Model:", rm.name)
    for v in rm.latest_versions:
        print(f"  version: {v.version}, stage: {v.current_stage}, run_id: {v.run_id}")
