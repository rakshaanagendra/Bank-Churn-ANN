from data_preprocessing import (
    drop_irrelevant, encode_categorical, scale_numeric,
    split_train_test, save_artifacts
)
import pandas as pd
from pathlib import Path

# Load raw data
df = pd.read_csv(Path("data/raw/Churn_Modelling.csv"))

# Preprocessing
df = drop_irrelevant(df)
df = encode_categorical(df)
df_scaled, scaler = scale_numeric(df, training=True)

# Split
train_df, test_df = split_train_test(df_scaled)

# Save artifacts
save_artifacts(train_df, test_df, scaler)
