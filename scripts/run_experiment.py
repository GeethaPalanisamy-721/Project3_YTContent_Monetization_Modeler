import time
import os
import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from scripts.prepare_data import load_and_prepare_data
from src.preprocessing import build_preprocessor
from src.Evaluate import evaluate_model  # central evaluation

# Define models
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "DecisionTree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "RandomForest": RandomForestRegressor(
        n_estimators=50, max_depth=15, n_jobs=-1, random_state=42
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=50, random_state=42
    ),
}

def main():
    start_time = time.time()

    # Load and prepare data
    X_train, X_test, y_train, y_test, num_cols, cat_cols = load_and_prepare_data()

    results = {}

    # Train and evaluate
    for name, model in models.items():
        print(f"\n Training {name}...")

        # choose preprocessing strategy
        if name in ["DecisionTree", "RandomForest", "GradientBoosting"]:
            model_type = "tree"
        else:
            model_type = "linear"

        pipe = Pipeline([
            ("preprocessor", build_preprocessor(num_cols, cat_cols, model_type=model_type)),
            ("model", model),
        ])

        t0 = time.time()
        pipe.fit(X_train, y_train)
        print(f" Finished {name} in {time.time() - t0:.2f} seconds")

        # evaluate
        metrics = evaluate_model(pipe, X_test, y_test)
        results[name] = metrics
        print(f" {name} -> {metrics}")

    # Save results
    results_df = pd.DataFrame(results).T
    os.makedirs("artifacts", exist_ok=True)
    results_path = os.path.join("artifacts", "model_results.csv")
    results_df.to_csv(results_path)
    print(f"\n Results saved to {results_path}")

    # Pick best model (highest R2)
    best_model_name = results_df["R2"].idxmax()
    best_model = models[best_model_name]
    print(f"\n Best model: {best_model_name}")

    # Refit on full training data
    best_pipe = Pipeline([
        ("preprocessor", build_preprocessor(num_cols, cat_cols, model_type="tree" if best_model_name in ["DecisionTree", "RandomForest", "GradientBoosting"] else "linear")),
        ("model", best_model),
    ])
    best_pipe.fit(X_train, y_train)

    # Save best model
    model_path = os.path.join("artifacts", "best_model.pkl")
    joblib.dump(best_pipe, model_path)
    print(f" Best model saved to {model_path}")

    print(f"\n Total runtime: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
