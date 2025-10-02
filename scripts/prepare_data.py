import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from src.feature_engineering import add_engineered_features

# Load paths from .env
load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")
CLEANED_DATA_PATH = os.getenv("CLEANED_DATA_PATH")

def load_and_prepare_data(test_size=0.2, random_state=42):
    print(f"\n Loading raw data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Raw data shape: {df.shape}")

    # ---------------------------
    # 1. Feature engineering
    # ---------------------------
    print(" Applying feature engineering...")
    df = add_engineered_features(df)

    # ---------------------------
    # 2. Drop duplicates and handle missing values
    # ---------------------------
    before = df.shape
    df = df.drop_duplicates()
    after = df.shape
    print(f" Dropped {before[0]-after[0]} duplicate rows")

    missing_before = df.isna().sum().sum()
    df = df.dropna(thresh=int(0.95 * df.shape[1]))  # drop rows with >5% missing
    df = df.fillna(df.median(numeric_only=True))    # fill numeric NaN

    # Fix: safely handle categorical/object NaNs
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    df[cat_cols] = df[cat_cols].astype(str).fillna("Unknown")

    missing_after = df.isna().sum().sum()
    print(f"Missing values before: {missing_before}, after cleaning: {missing_after}")

    # ---------------------------
    # 3. Drop identifiers (not useful for ML)
    # ---------------------------
    drop_cols = ["video_id", "channel_id"]  
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # ---------------------------
    # 4. Save cleaned data
    # ---------------------------
    os.makedirs(os.path.dirname(CLEANED_DATA_PATH), exist_ok=True)
    df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f" Cleaned data saved to {CLEANED_DATA_PATH}")
    print(f" Cleaned data shape: {df.shape}")

    # ---------------------------
    # 5. Split features & target
    # ---------------------------
    target = "ad_revenue_usd"
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f" Train size: {X_train.shape}, Test size: {X_test.shape}")

    # ---------------------------
    # 6. Identify numeric & categorical columns
    # ---------------------------
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    print(f" Numeric cols: {len(num_cols)},   Categorical cols: {len(cat_cols)}")

    return X_train, X_test, y_train, y_test, num_cols, cat_cols
