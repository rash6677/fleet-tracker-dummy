# backend/ml_model.py

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# xgboost is optional; we fall back if it's not installed
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


class EtaModelManager:
    """
    Manages PCA + multiple regression models for ETA prediction.
    Models: RandomForest, SVR, GradientBoosting, (optional) XGBoost.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)

        X, y = self._generate_fake_data()
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)

        self.models = {
            "rf": RandomForestRegressor(
                n_estimators=120, random_state=42
            ).fit(X_pca, y),
            "svr": SVR(kernel="rbf", C=100, gamma="scale").fit(X_pca, y),
            "gbr": GradientBoostingRegressor(
                random_state=42
            ).fit(X_pca, y),
        }

        if HAS_XGB:
            self.models["xgb"] = xgb.XGBRegressor(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=42,
            ).fit(X_pca, y)

        print("✅ ETAModelManager initialized with models:", list(self.models.keys()))

    def _generate_fake_data(self, n: int = 3000):
        np.random.seed(42)
        distance = np.random.uniform(10, 800, n)
        speed = np.random.uniform(20, 90, n)

        true_time = (distance / speed) * 60  # minutes
        noise = np.random.normal(0, 5, n)
        y = true_time + noise

        X = np.vstack([distance, speed]).T
        return X, y

    def predict_eta(self, features: dict, model_key: str) -> float:
        """
        features: {"distance_km": float, "avg_speed": float}
        model_key: "rf", "svr", "gbr", or "xgb"
        """
        distance = float(features["distance_km"])
        speed = max(float(features["avg_speed"]), 5.0)

        X = np.array([[distance, speed]])
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)

        if model_key not in self.models:
            # fall back gracefully
            model_key = "xgb" if ("xgb" in self.models) else "rf"

        model = self.models[model_key]
        eta = float(model.predict(X_pca)[0])

        return max(1.0, eta)  # ensure positive
