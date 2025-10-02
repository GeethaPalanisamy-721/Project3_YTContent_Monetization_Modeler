import pandas as pd
import numpy as np

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Engagement
    df["engagement_rate"] = (df["likes"] + df["comments"]) / (df["views"] + 1)

    # Subscribers influence
    if "subscribers" in df.columns:
        df["subs_per_view"] = df["subscribers"] / (df["views"] + 1)
        df["views_per_sub"] = df["views"] / (df["subscribers"] + 1)

    # Video duration
    if "video_length_minutes" in df.columns:
        df["log_video_length"] = np.log1p(df["video_length_minutes"])
        df["length_category"] = pd.cut(
            df["video_length_minutes"],
            bins=[0, 5, 15, 60, np.inf],
            labels=["short", "medium", "long", "very_long"]
        )

    # Temporal
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["dayofweek"] = df["date"].dt.dayofweek
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # Country grouping
    if "country" in df.columns:
        top_countries = df["country"].value_counts().nlargest(10).index
        df["country_grouped"] = df["country"].where(df["country"].isin(top_countries), "Other")

    return df
