"""
Preprocessor — SmartCity Pulse
Cleans raw data and engineers features for ML models.
"""

# import pandas for data manipulation
import pandas as pd

# import numpy for math operations
import numpy as np

# import StandardScaler to scale features to same range
from sklearn.preprocessing import StandardScaler

# import logging for professional log messages
import logging

# create logger for this file
logger = logging.getLogger(__name__)


class WeatherPreprocessor:
    """
    Cleans and prepares weather data for ML models.

    Usage:
        preprocessor = WeatherPreprocessor()
        X, y = preprocessor.prepare(df)
    """

    def __init__(self):
        # create one scaler — reused across all calls
        self.scaler = StandardScaler()

        # tracks whether scaler has been fitted yet
        self.is_fitted = False

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates new features from raw weather DataFrame.

        Args:
            df: Raw weather DataFrame from data_fetcher.py

        Returns:
            DataFrame with new engineered features added.
        """
        # never modify original data — always work on a copy
        df = df.copy()

        # extract hour from datetime — 0 to 23
        df["hour"] = df["dt"].dt.hour

        # extract day of week — 0=Monday, 6=Sunday
        df["day_of_week"] = df["dt"].dt.dayofweek

        # extract month — 1 to 12
        df["month"] = df["dt"].dt.month

        # 1 if Saturday or Sunday, 0 otherwise
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # 1 if night time (10pm to 6am), 0 otherwise
        df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)

        # target for rain classification — 1=rain, 0=no rain
        df["will_rain"] = (df["rain_1h"] > 0).astype(int)

        # temperature category — 0=Cold, 1=Mild, 2=Warm, 3=Hot
        df["temp_category"] = pd.cut(
            df["temp"],
            bins=[0, 15, 25, 35, 50],
            labels=[0, 1, 2, 3]
        ).astype(int)

        # hour category — 0=Night, 1=Morning, 2=Afternoon, 3=Evening
        df["hour_category"] = pd.cut(
            df["hour"],
            bins=[-1, 6, 12, 18, 24],
            labels=[0, 1, 2, 3]
        ).astype(int)

        # log shape after engineering
        logger.info(f"Engineered features. Shape: {df.shape}")
        return df

    def prepare(self, df: pd.DataFrame, target: str = "hour_category"):
        """
        Full preparation pipeline — engineer, clean, scale, split.

        Args:
            df: Raw weather DataFrame
            target: Column name to predict (default: hour_category)

        Returns:
            X: Scaled feature matrix ready for ML models
            y: Target variable
        """
        # step 1 — create new features
        df = self.engineer_features(df)

        # step 2 — fill any missing values with 0
        df = df.fillna(0)

        # step 3 — define which columns are features
        feature_cols = [
            "hour", "day_of_week", "month",
            "is_weekend", "is_night",
            "humidity", "wind_speed",
            "feels_like", "temp_category"
        ]

        # X = features — what model learns FROM
        X = df[feature_cols]

        # y = target — what model learns TO predict
        y = df[target]

        # step 4 — scale features
        # fit_transform on first call — learns mean and std
        # transform only on later calls — uses already learned values
        if not self.is_fitted:
            X_scaled = self.scaler.fit_transform(X)
            self.is_fitted = True
        else:
            # only scale — do not relearn from new data
            X_scaled = self.scaler.transform(X)

        # convert back to DataFrame to preserve column names
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

        # log final shape
        logger.info(f"Prepared data. X shape: {X_scaled.shape}")
        return X_scaled, y


# runs only when you run this file directly
if __name__ == "__main__":
    import sys

    # add pipeline folder so Python finds data_fetcher
    sys.path.append("src/pipeline")
    from data_fetcher import CityDataFetcher

    # fetch live weather data
    fetcher = CityDataFetcher()
    df = fetcher.fetch_weather("Mumbai")

    # preprocess data
    preprocessor = WeatherPreprocessor()
    X, y = preprocessor.prepare(df)

    # print results
    print("\n--- FEATURES ---")
    print(X.head())
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"\nTarget distribution:")
    print(y.value_counts())