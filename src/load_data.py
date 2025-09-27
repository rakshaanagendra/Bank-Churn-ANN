# src/load_data.py
"""
Load the churn CSV and print a few sanity checks.
Run from project root with:
    python src\load_data.py
"""

from pathlib import Path         # friendly path handling across OSes
import pandas as pd              

# ---- 1) define path to CSV data ----
DATA_PATH = Path("data/raw/Churn_Modelling.csv")

def main():
    # ---- 2) load CSV into a pandas DataFrame ----
    df = pd.read_csv(DATA_PATH)

    # ---- 3) Quick sanity prints ----
    print("1) Shape (rows, columns):", df.shape)
    print("\n2) Column names:\n", df.columns.tolist())
    print("\n3) First 5 rows:\n", df.head().to_string(index=False))
    print("\n4) Missing values per column:\n", df.isnull().sum())
    if 'Exited' in df.columns:
        print("\n5) 'Exited' distribution (fractions):")
        print(df['Exited'].value_counts(normalize=True))
    else:
        print("\n5) NOTE: 'Exited' column not found.")

if __name__ == "__main__":
    main()
