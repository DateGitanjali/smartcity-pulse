"""
Preprocessor — SmartCity Pulse
Cleans raw data and engineers features for ML models.
"""
import pandas as pd                              # manipulate DataFrames
import numpy as np                               # math operations
from sklearn.preprocessing import StandardScaler # scale features to same range
import logging                                   # professional logging

logger = logging.getLogger(__name__)             # logger for this file


class WeatherPreprocessor:
    """
    Cleans and prepares weather data for ML models.

    Usage:
        preprocessor = WeatherPreprocessor()
        X, y = preprocessor.prepare(df)
    """

    def __init__(self):
        self.scaler = StandardScaler() # one scaler reused across calls
        self.is_fitted = False         # tracks if scaler has been fitted yet

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates new features from raw weather DataFrame.

        Args:
            df: Raw weather DataFrame from data_fetcher.py

        Returns:
            DataFrame with new engineered features added.
        """
        df = df.copy()  # never modify original data — always work on a copy

        # Extract time-based features from datetime column
        df["hour"]        = df["dt"].dt.hour          # 0-23
        df["day_of_week"] = df["dt"].dt.dayofweek     # 0=Monday, 6=Sunday
        df["month"]       = df["dt"].dt.month         # 1-12

        # Binary features — 1 means yes, 0 means no
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)  # Sat=5, Sun=6
        df["is_night"]   = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)

        # Target variable — what we want to predict
        df["will_rain"] = (df["rain_1h"] > 0).astype(int)  # 1=rain, 0=no rain

        # Temperature category — 0=Cold, 1=Mild, 2=Warm, 3=Hot
        df["temp_category"] = pd.cut(
            df["temp"],
            bins=[0, 15, 25, 35, 50],
            labels=[0, 1, 2, 3]
        ).astype(int)

        logger.info(f"Engineered features. Shape: {df.shape}")
        return df

    def prepare(self, df: pd.DataFrame, target: str = "will_rain"):
        """
        Full preparation pipeline — engineer, clean, scale, split.

        Args:
            df: Raw weather DataFrame
            target: Column name to predict

        Returns:
            X: Scaled feature matrix ready for ML models
            y: Target variable
        """
        # Step 1 — create new features
        df = self.engineer_features(df)

        # Step 2 — fill any missing values with 0
        df = df.fillna(0)

        # Step 3 — define which columns are features
        feature_cols = [
            "hour", "day_of_week", "month",
            "is_weekend", "is_night",
            "humidity", "wind_speed",
            "feels_like", "temp_category"
        ]

        X = df[feature_cols]  # features — what model learns FROM
        y = df[target]        # target — what model learns TO predict

        # Step 4 — scale features
        # fit_transform on first call — learns mean and std from this data
        # transform only on later calls — uses already learned mean and std
        if not self.is_fitted:
            X_scaled = self.scaler.fit_transform(X)  # learn + scale
            self.is_fitted = True
        else:
            X_scaled = self.scaler.transform(X)      # only scale, don't relearn

        # Convert back to DataFrame so column names are preserved
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

        logger.info(f"Prepared data. X shape: {X_scaled.shape}")
        return X_scaled, y


# This block only runs when you run this file directly
# It does NOT run when another file imports this file
if __name__ == "__main__":
    import sys
    sys.path.append("..")                        # so Python finds src/
    from data_fetcher import CityDataFetcher

    # Fetch live data
    fetcher = CityDataFetcher()
    df = fetcher.fetch_weather("Mumbai")

    # Preprocess it
    preprocessor = WeatherPreprocessor()
    X, y = preprocessor.prepare(df)

    print("\n--- FEATURES ---")
    print(X.head())                              # first 5 rows
    print(f"\nX shape: {X.shape}")               # rows x columns
    print(f"y shape: {y.shape}")                 # number of targets

    print(f"\nTarget distribution:")
    print(y.value_counts())                      # how many 0s and 1s