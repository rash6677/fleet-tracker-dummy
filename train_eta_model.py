# train_eta_models.py

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

DATA_FILE = "data/raw/trips.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

if __name__ == "__main__":
    # 1. Load data
    df = pd.read_csv(DATA_FILE)

    # 2. Select features and target
    feature_cols = ["distance_km", "avg_speed"]
    X = df[feature_cols].values
    y = df["duration_min"].values

    # 3. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. PCA (dimensionality reduction)
    n_components = min(2, X.shape[1])  # here it's 2 anyway
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # 5. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42
    )

    # 6. Define models
    models = {
        "rf_eta.pkl": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        ),
        "svr_eta.pkl": SVR(
            kernel="rbf",
            C=10.0,
            epsilon=1.0,
        ),
        "gbr_eta.pkl": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
        "xgb_eta.pkl": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42,
        ),
    }

    results = []

    # 7. Train & evaluate each model
    for filename, model in models.items():
        print(f"\nTraining {filename} ...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        results.append({
            "model_file": filename,
            "model_name": model.__class__.__name__,
            "R2": r2,
            "MAE_min": mae,
        })

        # Save each model separately
        joblib.dump(model, os.path.join(MODELS_DIR, filename))
        print(f"{filename} → R² = {r2:.4f}, MAE = {mae:.2f} min")

    # 8. Save shared artifacts (scaler, pca, feature columns)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(pca, os.path.join(MODELS_DIR, "pca.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.pkl"))

    print("\n=== Summary ===")
    for r in results:
        print(
            f"{r['model_name']:>25} | file: {r['model_file']:11} | "
            f"R² = {r['R2']:.4f} | MAE = {r['MAE_min']:.2f} min"
        )
