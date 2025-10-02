import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ---------------------------
# Load trained model + data
# ---------------------------
MODEL_PATH = "C:/Users/Siva Sankar/Desktop/Python Workspace/MDTM46B/YT_content_Monetization/artifacts/best_model.pkl"
DATA_PATH = "C:/Users/Siva Sankar/Desktop/Python Workspace/MDTM46B/YT_content_Monetization/data/cleaned/youtube_ad_revenue_CleanData.csv"

# Load model
model = joblib.load(MODEL_PATH)

# Load cleaned data
df = pd.read_csv(DATA_PATH)

# Expected columns (features only)
expected_cols = [col for col in df.columns if col != "ad_revenue_usd"]

# ---------------------------
# Streamlit Config
# ---------------------------
st.set_page_config(page_title="YouTube Revenue Prediction", layout="wide")
st.title("YouTube Ad Revenue Modeler")

menu = ["EDA", "Manual Prediction", "Random Sample Test"]
choice = st.sidebar.radio(" Select Page", menu)

# ---------------------------
# 1. EDA Page
# ---------------------------
if choice == "EDA":
    st.header("Exploratory Data Analysis (EDA)")

    st.subheader("Dataset Summary")
    st.write(df.describe(include="all"))

    st.subheader(" Missing Values")
    st.write(df.isna().sum())

    # Target distribution
    st.subheader(" Target Distribution: Ad Revenue")
    fig, ax = plt.subplots()
    df["ad_revenue_usd"].hist(bins=50, ax=ax)
    ax.set_xlabel("Ad Revenue (USD)")
    ax.set_ylabel("Frequency")
    ax.set_title("Ad Revenue Distribution")
    st.pyplot(fig)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    numeric_df = df.select_dtypes(include=["number"])
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Avg Revenue by Category
    st.subheader("Average Revenue(USD) by Category")
    avg_rev_cat = df.groupby("category")["ad_revenue_usd"].mean().sort_values()
    st.bar_chart(avg_rev_cat)

    st.subheader("Average Revenue(USD) by Country Grouped")
    avg_rev_country = df.groupby("country_grouped")["ad_revenue_usd"].mean().sort_values()
    st.bar_chart(avg_rev_country)

    st.subheader("Average Revenue(USD) by Device")
    avg_rev_device = df.groupby("device")["ad_revenue_usd"].mean().sort_values()
    st.bar_chart(avg_rev_device)


# ---------------------------
# 2. Manual Prediction + What-if
# ---------------------------
elif choice == "Manual Prediction":
    st.header("Predict Revenue from Video Features")

    col1, col2 = st.columns(2)

    with col1:
        likes = st.number_input("Likes", min_value=0, step=1)
        comments = st.number_input("Comments", min_value=0, step=1)
        watch_time = st.number_input("Watch Time (minutes)", min_value=0, step=1)
        video_length = st.number_input("Video Length (minutes)", min_value=1, step=1)

    with col2:
        subscribers = st.number_input("Subscribers", min_value=0, step=1)
        category = st.selectbox("Category", df["category"].unique())
        device = st.selectbox("Device", df["device"].unique())
        country = st.selectbox("Country", df["country_grouped"].unique())
        date_input = st.date_input("Upload Date", datetime.today())

    # Feature engineering for date
    year = date_input.year
    month = date_input.month
    dayofweek = date_input.weekday()
    is_weekend = 1 if dayofweek in [5, 6] else 0

    # Input dataframe
    input_df = pd.DataFrame([{
        "likes": likes,
        "comments": comments,
        "watch_time_minutes": watch_time,
        "video_length_minutes": video_length,
        "subscribers": subscribers,
        "category": category,
        "device": device,
        "country_grouped": country,
        "year": year,
        "month": month,
        "dayofweek": dayofweek,
        "is_weekend": is_weekend
    }])

    # Align columns
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_cols]

    # Prediction
    if st.button("Predict Revenue"):
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Ad Revenue: **${prediction:,.2f}**")

        # ========================
        # What-if Analysis
        # ========================
        st.subheader("What-if Analysis")
        st.caption("Experiment with feature changes to see impact on revenue.")

        # Increase likes by 20%
        scenario_df1 = input_df.copy()
        scenario_df1["likes"] *= 1.2
        pred1 = model.predict(scenario_df1)[0]
        st.write(f"If Likes increase by 20% → **${pred1:,.2f}**")

        # Double watch time
        scenario_df2 = input_df.copy()
        scenario_df2["watch_time_minutes"] *= 2
        pred2 = model.predict(scenario_df2)[0]
        st.write(f"If Watch Time doubles → **${pred2:,.2f}**")

        # Increase subscribers by 50%
        scenario_df3 = input_df.copy()
        scenario_df3["subscribers"] *= 1.5
        pred3 = model.predict(scenario_df3)[0]
        st.write(f"If Subscribers increase by 50% → **${pred3:,.2f}**")


# ---------------------------
# 3. Random Sample Test
# ---------------------------
elif choice == "Random Sample Test":
    st.header("Random Sample Test")

    n_samples = st.slider("Number of samples", min_value=1, max_value=50, value=5)

    if st.button("Pick Random Sample"):
        sample = df.sample(n_samples, random_state=None).copy()
        X_sample = sample.drop(columns=["ad_revenue_usd"])
        y_true = sample["ad_revenue_usd"]

        # Align with model
        for col in expected_cols:
            if col not in X_sample.columns:
                X_sample[col] = 0
        X_sample = X_sample[expected_cols]

        y_pred = model.predict(X_sample)

        results = sample.copy()
        results["Predicted_Revenue"] = y_pred
        st.write(results.head(20))

        # Scatter Plot
        fig, ax = plt.subplots()
        ax.scatter(y_true, y_pred, color="blue", alpha=0.6, label="Predictions")
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                "r--", label="Perfect Prediction")
        ax.set_xlabel("Actual Revenue (USD)")
        ax.set_ylabel("Predicted Revenue (USD)")
        ax.set_title("Actual vs Predicted Revenue")
        ax.legend()
        st.pyplot(fig)
