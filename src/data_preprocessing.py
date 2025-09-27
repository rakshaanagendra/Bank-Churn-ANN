# src/data_preprocessing.py
import pandas as pd

model = None
scaler = None

def drop_irrelevant(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are identifiers and don't help prediction.
    - RowNumber, CustomerId, Surname are IDs/names, not useful features.
    """
    return df.drop(columns=["RowNumber", "CustomerId", "Surname"], errors="ignore")


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables into numeric form.
    - Gender: map Female->0, Male->1
    - Geography: one-hot encode into multiple binary columns
    """
    df = df.copy()

    # Encode Gender
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})

    # One-hot encode Geography (drop_first avoids redundant column)
    if "Geography" in df.columns:
        # Encode categorical columns exactly like training
        df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

        # Apply scaler
        df_scaled = scaler.transform(df)

        # Back to DataFrame
        df = pd.DataFrame(df_scaled, columns=scaler.feature_names_in_)


    return df


from sklearn.preprocessing import StandardScaler

def scale_numeric(df: pd.DataFrame, scaler: StandardScaler = None, training: bool = True):
    """
    Standardize numeric columns (mean=0, std=1).
    - If training=True, fit a new scaler.
    - If training=False, use the given scaler to transform (no refit).
    Returns: (df_scaled, scaler)
    """
    df = df.copy()

    # Separate target column if present
    target = None
    if "Exited" in df.columns:
        target = df["Exited"]
        df = df.drop(columns=["Exited"])

    # Identify numeric columns (int/float types)
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if training:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])

    # Add back target if it was present
    if target is not None:
        df["Exited"] = target

    return df, scaler


from sklearn.model_selection import train_test_split

def split_train_test(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split into train and test sets.
    - test_size: fraction of dataset for test (default 20%)
    - random_state: seed for reproducibility
    - stratify: ensures same churn proportion (Exited=0/1) in both sets
    Returns: train_df, test_df
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["Exited"] if "Exited" in df.columns else None
    )
    return train_df, test_df


import joblib
import os

def save_artifacts(train_df, test_df, scaler, processed_dir="data/processed"):
    """
    Save processed train/test CSVs and the fitted scaler.
    """
    os.makedirs(processed_dir, exist_ok=True)

    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)
    joblib.dump(scaler, os.path.join(processed_dir, "scaler.joblib"))

    print(f"Artifacts saved in {processed_dir}/")
