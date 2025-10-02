from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def get_models():
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "RandomForest": RandomForestRegressor(
            n_estimators=50, max_depth=10, n_jobs=-1, random_state=42
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=50, max_depth=5, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=50, max_depth=5, n_jobs=-1, verbosity=0, random_state=42
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=50, n_jobs=-1, random_state=42
        ),
    }
