import pandas as pd
from sklearn.preprocessing import StandardScaler
from src import data_preprocessing as dp

def test_encode_categorical_with_scaler():
    df = pd.DataFrame({
        "CreditScore": [600],
        "Gender": ["Male"],
        "Geography": ["France"],
        "Age": [40],
        "Tenure": [5],
        "Balance": [1000.0],
        "NumOfProducts": [2],
        "HasCrCard": [1],
        "IsActiveMember": [1],
        "EstimatedSalary": [50000.0]
    })

    # One-hot encode manually first, so scaler sees final columns
    df_encoded = pd.get_dummies(df.copy(), columns=["Geography"], drop_first=True)
    df_encoded["Gender"] = df_encoded["Gender"].map({"Female": 0, "Male": 1})

    # Fit scaler on ALL numeric columns after encoding
    scaler = StandardScaler()
    scaler.fit(df_encoded)

    # Inject scaler into module
    dp.scaler = scaler

    encoded = dp.encode_categorical(df)
    assert isinstance(encoded, pd.DataFrame)
    assert set(encoded.columns) == set(scaler.feature_names_in_)
