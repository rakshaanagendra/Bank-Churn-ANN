import requests

url = "http://127.0.0.1:8000/predict_batch"
payload = {
    "customers": [
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
        },
        {
            "CreditScore": 500,
            "Geography": "Spain",
            "Gender": "Male",
            "Age": 35,
            "Tenure": 5,
            "Balance": 15000.0,
            "NumOfProducts": 2,
            "HasCrCard": 1,
            "IsActiveMember": 0,
            "EstimatedSalary": 50000.0
        }
    ]
}
r = requests.post(url, json=payload)
print(r.status_code)
print(r.json())
