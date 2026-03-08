"""
Classical ML Models — SmartCity Pulse
Trains 6 sklearn models on live weather data.
"""

# import pickle to save trained models to disk as .pkl files
import pickle

# import numpy for math operations like sqrt
import numpy as np

# import pandas for data manipulation
import pandas as pd

# import mlflow for experiment tracking
import mlflow

# import mlflow sklearn integration to save sklearn models
import mlflow.sklearn

# import train_test_split to split data into train and test sets
from sklearn.model_selection import train_test_split

# import LinearRegression for predicting continuous values like temperature
from sklearn.linear_model import LinearRegression

# import LogisticRegression for binary classification like rain yes/no
from sklearn.linear_model import LogisticRegression

# import DecisionTreeClassifier for flowchart style decisions
from sklearn.tree import DecisionTreeClassifier

# import RandomForestClassifier for 100 trees voting together
from sklearn.ensemble import RandomForestClassifier

# import SVC for finding best boundary between classes
from sklearn.svm import SVC

# import KNeighborsClassifier for finding similar past readings
from sklearn.neighbors import KNeighborsClassifier

# import metrics to evaluate model performance
from sklearn.metrics import (
    mean_squared_error,      # measures average error for regression
    r2_score,                # measures how well model fits data
    accuracy_score,          # measures percentage of correct predictions
    classification_report    # detailed breakdown of precision recall f1
)

# import logging for professional log messages
import logging

# import os to create folders
import os

# import sys to add file paths
import sys

# create logger for this file
logger = logging.getLogger(__name__)


def train_all_models(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Trains 6 classical ML models on weather data.

    Args:
        X: Feature matrix from preprocessor.py
        y: Target variable (will_rain)

    Returns:
        Dictionary with model names and their metrics
    """

    # split data — 80% for training, 20% for testing
    # random_state=42 means same split every time you run
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # log how many rows in train and test
    logger.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # create models folder if it does not exist
    os.makedirs("models", exist_ok=True)

    # set MLflow experiment name — all runs go here
    mlflow.set_experiment("smartcity-classical-ml")

    # empty dictionary to store results of all models
    results = {}

    # ════════════════════════════════════════════════════
    # MODEL 1 — LINEAR REGRESSION
    # predicts exact temperature value (regression)
    # ════════════════════════════════════════════════════

    # log message that training is starting
    logger.info("Training Linear Regression...")

    # start MLflow run — everything inside gets logged automatically
    with mlflow.start_run(run_name="linear_regression"):

        # create linear regression model
        lr = LinearRegression()

        # train model on training data — model learns here
        lr.fit(X_train, y_train)

        # predict on test data — data model has never seen
        predictions = lr.predict(X_test)

        # calculate RMSE — lower is better
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        # calculate R2 score — higher is better, max is 1.0
        r2 = r2_score(y_test, predictions)

        # log rmse metric to MLflow dashboard
        mlflow.log_metric("rmse", rmse)

        # log r2 metric to MLflow dashboard
        mlflow.log_metric("r2", r2)

        # save model to MLflow
        mlflow.sklearn.log_model(lr, "linear_regression")

        # save model to disk as pkl file
        with open("models/linear_regression.pkl", "wb") as f:
            pickle.dump(lr, f)

        # print results to terminal
        print(f"\n✅ Linear Regression")
        print(f"   RMSE : {rmse:.4f}")
        print(f"   R2   : {r2:.4f}")

        # store results in dictionary
        results["linear_regression"] = {
            "rmse": rmse,
            "r2"  : r2
        }

    # ════════════════════════════════════════════════════
    # MODEL 2 — LOGISTIC REGRESSION
    # predicts will it rain — YES=1 or NO=0
    # ════════════════════════════════════════════════════

    # log message that training is starting
    logger.info("Training Logistic Regression...")

    # start MLflow run for logistic regression
    with mlflow.start_run(run_name="logistic_regression"):

        # create logistic regression model
        # max_iter=1000 gives enough iterations to converge
        log_reg = LogisticRegression(max_iter=1000)

        # train model on training data
        log_reg.fit(X_train, y_train)

        # predict on test data
        predictions = log_reg.predict(X_test)

        # calculate accuracy — percentage of correct predictions
        accuracy = accuracy_score(y_test, predictions)

        # get detailed report with precision recall f1
        report = classification_report(y_test, predictions)

        # log accuracy to MLflow
        mlflow.log_metric("accuracy", accuracy)

        # save model to MLflow
        mlflow.sklearn.log_model(log_reg, "logistic_regression")

        # save model to disk
        with open("models/logistic_regression.pkl", "wb") as f:
            pickle.dump(log_reg, f)

        # print results
        print(f"\n✅ Logistic Regression")
        print(f"   Accuracy : {accuracy:.4f}")
        print(f"   Report   :\n{report}")

        # store results
        results["logistic_regression"] = {
            "accuracy": accuracy
        }

    # ════════════════════════════════════════════════════
    # MODEL 3 — DECISION TREE
    # makes decisions like a flowchart
    # ════════════════════════════════════════════════════

    # log message that training is starting
    logger.info("Training Decision Tree...")

    # start MLflow run for decision tree
    with mlflow.start_run(run_name="decision_tree"):

        # create decision tree model
        # max_depth=5 limits tree size to prevent overfitting
        # random_state=42 gives same tree every run
        dt = DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        )

        # train model on training data
        dt.fit(X_train, y_train)

        # predict on test data
        predictions = dt.predict(X_test)

        # calculate accuracy
        accuracy = accuracy_score(y_test, predictions)

        # get detailed classification report
        report = classification_report(y_test, predictions)

        # log accuracy metric to MLflow
        mlflow.log_metric("accuracy", accuracy)

        # log max_depth parameter to MLflow
        mlflow.log_param("max_depth", 5)

        # save model to MLflow
        mlflow.sklearn.log_model(dt, "decision_tree")

        # save model to disk
        with open("models/decision_tree.pkl", "wb") as f:
            pickle.dump(dt, f)

        # print results
        print(f"\n✅ Decision Tree")
        print(f"   Accuracy  : {accuracy:.4f}")
        print(f"   Max Depth : 5")
        print(f"   Report    :\n{report}")

        # store results
        results["decision_tree"] = {
            "accuracy" : accuracy,
            "max_depth": 5
        }

    # ════════════════════════════════════════════════════
    # MODEL 4 — RANDOM FOREST
    # 100 decision trees voting together
    # ════════════════════════════════════════════════════

    # log message that training is starting
    logger.info("Training Random Forest...")

    # start MLflow run for random forest
    with mlflow.start_run(run_name="random_forest"):

        # create random forest model
        # n_estimators=100 means build 100 trees
        # random_state=42 gives same forest every run
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        # train model on training data
        rf.fit(X_train, y_train)

        # predict on test data
        predictions = rf.predict(X_test)

        # calculate accuracy
        accuracy = accuracy_score(y_test, predictions)

        # get detailed classification report
        report = classification_report(y_test, predictions)

        # get feature importance — which features matter most
        feature_importance = dict(zip(
            X.columns,
            rf.feature_importances_
        ))

        # sort features by importance — highest first
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # log accuracy to MLflow
        mlflow.log_metric("accuracy", accuracy)

        # log n_estimators parameter to MLflow
        mlflow.log_param("n_estimators", 100)

        # save model to MLflow
        mlflow.sklearn.log_model(rf, "random_forest")

        # save model to disk
        with open("models/random_forest.pkl", "wb") as f:
            pickle.dump(rf, f)

        # print results
        print(f"\n✅ Random Forest")
        print(f"   Accuracy     : {accuracy:.4f}")
        print(f"   Top Features :")

        # print top 3 most important features
        for feature, importance in sorted_features[:3]:
            print(f"      {feature} : {importance:.4f}")

        print(f"   Report       :\n{report}")

        # store results
        results["random_forest"] = {
            "accuracy"          : accuracy,
            "feature_importance": feature_importance
        }

    # ════════════════════════════════════════════════════
    # MODEL 5 — SVM
    # finds best boundary between rain and no rain
    # ════════════════════════════════════════════════════

    # log message that training is starting
    logger.info("Training SVM...")

    # start MLflow run for SVM
    with mlflow.start_run(run_name="svm"):

        # create SVM model
        # kernel=rbf handles non linear boundaries
        # C=1.0 controls strictness of boundary
        # random_state=42 for reproducibility
        svm = SVC(
            kernel="rbf",
            C=1.0,
            random_state=42
        )

        # train model on training data
        svm.fit(X_train, y_train)

        # predict on test data
        predictions = svm.predict(X_test)

        # calculate accuracy
        accuracy = accuracy_score(y_test, predictions)

        # get detailed classification report
        report = classification_report(y_test, predictions)

        # log accuracy to MLflow
        mlflow.log_metric("accuracy", accuracy)

        # log kernel parameter to MLflow
        mlflow.log_param("kernel", "rbf")

        # log C parameter to MLflow
        mlflow.log_param("C", 1.0)

        # save model to MLflow
        mlflow.sklearn.log_model(svm, "svm")

        # save model to disk
        with open("models/svm.pkl", "wb") as f:
            pickle.dump(svm, f)

        # print results
        print(f"\n✅ SVM")
        print(f"   Accuracy : {accuracy:.4f}")
        print(f"   Kernel   : rbf")
        print(f"   C        : 1.0")
        print(f"   Report   :\n{report}")

        # store results
        results["svm"] = {
            "accuracy": accuracy,
            "kernel"  : "rbf",
            "C"       : 1.0
        }

    # ════════════════════════════════════════════════════
    # MODEL 6 — KNN
    # finds 5 most similar past readings and votes
    # ════════════════════════════════════════════════════

    # log message that training is starting
    logger.info("Training KNN...")

    # start MLflow run for KNN
    with mlflow.start_run(run_name="knn"):

        # create KNN model
        # n_neighbors=5 means look at 5 closest data points
        knn = KNeighborsClassifier(n_neighbors=5)

        # train model — just memorizes all training data
        knn.fit(X_train, y_train)

        # predict on test data
        predictions = knn.predict(X_test)

        # calculate accuracy
        accuracy = accuracy_score(y_test, predictions)

        # get detailed classification report
        report = classification_report(y_test, predictions)

        # log accuracy to MLflow
        mlflow.log_metric("accuracy", accuracy)

        # log n_neighbors parameter to MLflow
        mlflow.log_param("n_neighbors", 5)

        # save model to MLflow
        mlflow.sklearn.log_model(knn, "knn")

        # save model to disk
        with open("models/knn.pkl", "wb") as f:
            pickle.dump(knn, f)

        # print results
        print(f"\n✅ KNN")
        print(f"   Accuracy    : {accuracy:.4f}")
        print(f"   n_neighbors : 5")
        print(f"   Report      :\n{report}")

        # store results
        results["knn"] = {
            "accuracy"   : accuracy,
            "n_neighbors": 5
        }

    # log that all models finished training
    logger.info("All 6 models trained successfully")

    # return all results dictionary
    return results


# ════════════════════════════════════════════════════════
# this block runs only when you run this file directly
# it does NOT run when another file imports this file
# ════════════════════════════════════════════════════════
if __name__ == "__main__":

    # add pipeline folder to path so Python finds data_fetcher
    sys.path.append("src/pipeline")

    # import data fetcher to get live weather data
    from data_fetcher import CityDataFetcher

    # import preprocessor to clean and engineer features
    from preprocessor import WeatherPreprocessor

    # fetch live weather data from API
    print("Fetching live weather data...")
    fetcher = CityDataFetcher()
    df = fetcher.fetch_weather("Mumbai")

    # preprocess data — engineer features and scale
    print("Preprocessing data...")
    preprocessor = WeatherPreprocessor()
    X, y = preprocessor.prepare(df, target="hour_category")

    # print data shapes so we know what we are working with
    print(f"\nData ready:")
    print(f"X shape : {X.shape}")
    print(f"y shape : {y.shape}")

    # train all 6 models and get results
    print("\nTraining all 6 models...")
    results = train_all_models(X, y)

    # print final summary of all models
    print("\n" + "=" * 50)
    print("FINAL RESULTS SUMMARY")
    print("=" * 50)

    # loop through each model and print its metrics
    for model_name, metrics in results.items():
        print(f"\n{model_name}")
        for metric, value in metrics.items():
            # only print float values — skip dictionaries
            if isinstance(value, float):
                print(f"   {metric} : {value:.4f}")