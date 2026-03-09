"""
Boosting Models — SmartCity Pulse
Trains XGBoost and LightGBM with Optuna hyperparameter tuning.
"""

# import pickle to save trained models to disk
import pickle

# import numpy for math operations
import numpy as np

# import pandas for data manipulation
import pandas as pd

# import mlflow for experiment tracking
import mlflow

# import mlflow sklearn integration
import mlflow.sklearn

# import xgboost classifier
from xgboost import XGBClassifier

# import lightgbm classifier
from lightgbm import LGBMClassifier

# import optuna for hyperparameter tuning
import optuna

# silence optuna logs — too noisy otherwise
optuna.logging.set_verbosity(optuna.logging.WARNING)

# import train test split
from sklearn.model_selection import train_test_split, cross_val_score

# import accuracy score
from sklearn.metrics import accuracy_score, classification_report

# import os to create folders
import os

# import sys to add file paths
import sys

# import logging
import logging

# create logger for this file
logger = logging.getLogger(__name__)

def train_xgboost(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Trains XGBoost classifier with Optuna hyperparameter tuning.

    Args:
        X: Feature matrix from preprocessor.py
        y: Target variable

    Returns:
        Dictionary with best params and metrics
    """

    # split data — 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # log split sizes
    logger.info(f"XGBoost — Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Step 1: Define Optuna objective function ─────────
    # Optuna calls this function 50 times with different params
    # Each call returns a score — Optuna tries to maximize it
    def objective(trial):

        # Optuna suggests values — it learns which ranges work best
        params = {
            # number of boosting rounds — more = slower but better
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),

            # max depth of each tree — controls complexity
            "max_depth": trial.suggest_int("max_depth", 3, 8),

            # learning rate — how much each tree contributes
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),

            # fraction of features used per tree — prevents overfitting
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),

            # fraction of rows used per tree — prevents overfitting
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),

            # fixes random seed for reproducibility
            "random_state": 42,

            # suppresses XGBoost output during tuning
            "verbosity": 0,

            # handles multi-class classification automatically
            "use_label_encoder": False,
            
            # evaluation metric used internally
            "eval_metric": "mlogloss"
        }

        # create model with suggested params
        model = XGBClassifier(**params)

        # cross validation — 5 fold — more reliable than single split
        # cv=5 means split into 5 parts, train 5 times, average score
        scores = cross_val_score(
            model, X_train, y_train,
            cv=3,               # 3 fold because our dataset is small
            scoring="accuracy"
        )

        # return average accuracy across 3 folds
        return scores.mean()

    # ── Step 2: Run Optuna study ─────────────────────────
    # direction=maximize means find params that give highest accuracy
    study = optuna.create_study(direction="maximize")

    # optimize runs the objective function 50 times
    # each time with different params suggested by Optuna
    logger.info("Running Optuna for XGBoost — 50 trials...")
    study.optimize(objective, n_trials=50)

    # get the best params found across all 50 trials
    best_params = study.best_params
    best_score  = study.best_value

    logger.info(f"Best XGBoost params: {best_params}")
    logger.info(f"Best CV score: {best_score:.4f}")

    # ── Step 3: Train final model with best params ───────
    # now train on full training data with the best params
    mlflow.set_experiment("smartcity-boosting")

    with mlflow.start_run(run_name="xgboost_optuna"):

        # create final model with best params from Optuna
        final_model = XGBClassifier(
            **best_params,
            random_state=42,
            verbosity=0,
            eval_metric="mlogloss"
        )

        # train on full training set
        final_model.fit(X_train, y_train)

        # predict on test set
        predictions = final_model.predict(X_test)

        # calculate final accuracy
        accuracy = accuracy_score(y_test, predictions)

        # get detailed report
        report = classification_report(y_test, predictions)

        # log best params to MLflow
        for param, value in best_params.items():
            mlflow.log_param(param, value)

        # log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("best_cv_score", best_score)

        # save model to MLflow
        mlflow.sklearn.log_model(final_model, "xgboost")

        # save model to disk
        with open("models/xgboost.pkl", "wb") as f:
            pickle.dump(final_model, f)

        # print results
        print(f"\n✅ XGBoost + Optuna")
        print(f"   Best CV Score : {best_score:.4f}")
        print(f"   Test Accuracy : {accuracy:.4f}")
        print(f"   Best Params   :")
        for param, value in best_params.items():
            print(f"      {param} : {value}")
        print(f"   Report        :\n{report}")

    # return results
    return {
        "accuracy"   : accuracy,
        "best_params": best_params,
        "cv_score"   : best_score
    }
    
    
def train_lightgbm(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Trains LightGBM classifier with Optuna hyperparameter tuning.

    Args:
        X: Feature matrix from preprocessor.py
        y: Target variable

    Returns:
        Dictionary with best params and metrics
    """

    # split data — 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # log split sizes
    logger.info(f"LightGBM — Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Step 1: Define Optuna objective function ─────────
    def objective(trial):

        # Optuna suggests hyperparameter values
        params = {
            # number of boosting rounds
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),

            # max depth of each tree
            "max_depth": trial.suggest_int("max_depth", 3, 8),

            # learning rate — smaller = more careful learning
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),

            # number of leaves — controls tree complexity
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),

            # fraction of features used per tree
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),

            # fraction of rows used per tree
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),

            # fixes random seed
            "random_state": 42,

            # silence LightGBM output during tuning
            "verbosity": -1
        }

        # create model with suggested params
        model = LGBMClassifier(**params)

        # cross validation — 3 fold
        scores = cross_val_score(
            model, X_train, y_train,
            cv=3,
            scoring="accuracy"
        )

        # return average accuracy
        return scores.mean()

    # ── Step 2: Run Optuna study ─────────────────────────
    study = optuna.create_study(direction="maximize")

    # run 50 trials
    logger.info("Running Optuna for LightGBM — 50 trials...")
    study.optimize(objective, n_trials=50)

    # get best params and score
    best_params = study.best_params
    best_score  = study.best_value

    logger.info(f"Best LightGBM params: {best_params}")
    logger.info(f"Best CV score: {best_score:.4f}")

    # ── Step 3: Train final model with best params ───────
    mlflow.set_experiment("smartcity-boosting")

    with mlflow.start_run(run_name="lightgbm_optuna"):

        # create final model with best params
        final_model = LGBMClassifier(
            **best_params,
            random_state=42,
            verbosity=-1
        )

        # train on full training set
        final_model.fit(X_train, y_train)

        # predict on test set
        predictions = final_model.predict(X_test)

        # calculate final accuracy
        accuracy = accuracy_score(y_test, predictions)

        # get detailed report
        report = classification_report(y_test, predictions)

        # log best params to MLflow
        for param, value in best_params.items():
            mlflow.log_param(param, value)

        # log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("best_cv_score", best_score)

        # save model to MLflow
        mlflow.sklearn.log_model(final_model, "lightgbm")

        # save model to disk
        with open("models/lightgbm.pkl", "wb") as f:
            pickle.dump(final_model, f)

        # print results
        print(f"\n✅ LightGBM + Optuna")
        print(f"   Best CV Score : {best_score:.4f}")
        print(f"   Test Accuracy : {accuracy:.4f}")
        print(f"   Best Params   :")
        for param, value in best_params.items():
            print(f"      {param} : {value}")
        print(f"   Report        :\n{report}")

    # return results
    return {
        "accuracy"   : accuracy,
        "best_params": best_params,
        "cv_score"   : best_score
    }
# runs only when you run this file directly
if __name__ == "__main__":

    # add pipeline folder to path
    sys.path.append("src/pipeline")

    # import data fetcher and preprocessor
    from data_fetcher import CityDataFetcher
    from preprocessor import WeatherPreprocessor

    # fetch live weather data
    print("Fetching live weather data...")
    fetcher = CityDataFetcher()
    df = fetcher.fetch_weather("Mumbai")

    # preprocess data
    print("Preprocessing data...")
    preprocessor = WeatherPreprocessor()

    # use hour_category as target — has 4 classes
    X, y = preprocessor.prepare(df, target="hour_category")

    print(f"\nData ready:")
    print(f"X shape : {X.shape}")
    print(f"y shape : {y.shape}")
    print(f"Classes : {y.unique()}")

    # create models folder
    os.makedirs("models", exist_ok=True)

    # train XGBoost with Optuna
    print("\n" + "="*50)
    print("TRAINING XGBOOST WITH OPTUNA")
    print("="*50)
    xgb_results = train_xgboost(X, y)

    # train LightGBM with Optuna
    print("\n" + "="*50)
    print("TRAINING LIGHTGBM WITH OPTUNA")
    print("="*50)
    lgbm_results = train_lightgbm(X, y)

    # final comparison
    print("\n" + "="*50)
    print("FINAL COMPARISON")
    print("="*50)
    print(f"\nXGBoost  accuracy : {xgb_results['accuracy']:.4f}")
    print(f"LightGBM accuracy : {lgbm_results['accuracy']:.4f}")

    # which one won
    if xgb_results['accuracy'] >= lgbm_results['accuracy']:
        print("\n🏆 XGBoost wins today")
    else:
        print("\n🏆 LightGBM wins today")