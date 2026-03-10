"""
Predictions Router — SmartCity Pulse API
Handles all ML prediction endpoints.
Weather data, ML predictions, anomaly detection, NLP analysis.
"""

# import APIRouter — creates a group of related endpoints
from fastapi import APIRouter, HTTPException

# import sys for path manipulation
import sys
import os

# add pipeline and ml folders to path
sys.path.append("src/pipeline")
sys.path.append("src/ml")

# import our schemas
sys.path.append("src/api")
from schemas import (
    PredictionRequest,
    PredictionResponse,
    WeatherResponse,
    SentimentResponse,
    EntityResponse,
    TopicResponse,
    AnomalyRequest,
    AnomalyResponse
)

# import our pipeline files
from data_fetcher import CityDataFetcher
from preprocessor import WeatherPreprocessor

# import pandas
import pandas as pd
import numpy as np

# import pickle to load saved models
import pickle

# import logging
import logging

# create logger
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════
# CREATE ROUTER
# prefix and tags appear in API documentation
# ════════════════════════════════════════════════════════

router = APIRouter(
    prefix="/predictions",
    tags=["predictions"]
)

# ════════════════════════════════════════════════════════
# LOAD ML MODELS AT STARTUP
# We load models once when server starts
# Not every time a request comes in
# This makes responses much faster
# ════════════════════════════════════════════════════════

# dictionary to store all loaded models
models = {}

def load_models():
    """
    Loads all trained ML models from models/ folder.
    Called once when server starts.
    """
    # list of models to load
    model_files = {
        "random_forest"    : "models/random_forest.pkl",
        "xgboost"          : "models/xgboost.pkl",
        "decision_tree"    : "models/decision_tree.pkl",
        "svm"              : "models/svm.pkl",
        "isolation_forest" : "models/isolation_forest.pkl",
    }

    # loop through each model and load it
    for name, path in model_files.items():
        try:
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
            logger.info(f"Loaded model: {name}")
        except Exception as e:
            logger.warning(f"Could not load {name}: {e}")

    logger.info(f"Total models loaded: {len(models)}")


# load models when this file is imported
load_models()

# create single instances of fetcher and preprocessor
fetcher      = CityDataFetcher()
preprocessor = WeatherPreprocessor()

# label mapping for hour_category
LABEL_MAP = {
    0: "Night",
    1: "Morning",
    2: "Afternoon",
    3: "Evening"
}


# ════════════════════════════════════════════════════════
# ENDPOINT 1 — GET WEATHER
# GET /api/predictions/weather?city=Mumbai
# Returns current weather data for a city
# ════════════════════════════════════════════════════════

@router.get("/weather")
def get_weather(city: str = "Mumbai"):
    """
    Fetches live weather data for a city.
    Returns current conditions and forecast.
    """
    try:
        # fetch weather data
        logger.info(f"Fetching weather for {city}")
        df = fetcher.fetch_weather(city)

        # get current weather from first row
        current = df.iloc[0]

        # return weather response
        return {
            "city"         : city,
            "temperature"  : round(float(current["temp"]), 1),
            "feels_like"   : round(float(current["feels_like"]), 1),
            "humidity"     : int(current["humidity"]),
            "wind_speed"   : round(float(current["wind_speed"]), 1),
            "description"  : str(current["description"]),
            "rain_1h"      : round(float(current["rain_1h"]), 2),
            "total_records": len(df),
            # include forecast — next 4 readings
            "forecast"     : [
                {
                    "time"       : row["dt"].strftime("%H:%M"),
                    "temperature": round(float(row["temp"]), 1),
                    "description": str(row["description"])
                }
                for _, row in df.head(5).iloc[1:].iterrows()
            ]
        }

    except Exception as e:
        logger.error(f"Weather fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════
# ENDPOINT 2 — ML PREDICTION
# POST /api/predictions/predict
# Runs ML model on live weather data
# Returns prediction with confidence
# ════════════════════════════════════════════════════════

@router.post("/predict")
def predict(request: PredictionRequest):
    """
    Runs ML model prediction on live weather data.
    Fetches current weather, preprocesses it,
    runs the selected model, returns prediction.
    """
    try:
        # check if requested model exists
        if request.model_name not in models:
            available = list(models.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model_name}' not found. Available: {available}"
            )

        # fetch live weather data
        logger.info(f"Running prediction for {request.city}")
        df = fetcher.fetch_weather(request.city)

        # preprocess the data
        X, y = preprocessor.prepare(df, target="hour_category")

        # get the model
        model = models[request.model_name]

        # make prediction on first row only
        X_current = X[0:1]

        # get prediction
        prediction_num = model.predict(X_current)[0]

        # get confidence if model supports probability
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_current)[0]
            confidence = round(float(max(proba)), 4)

        # convert number to label
        prediction_label = LABEL_MAP.get(int(prediction_num), str(prediction_num))

        # get current weather for context
        current = df.iloc[0]

        # return prediction response
        return {
            "city"        : request.city,
            "model_used"  : request.model_name,
            "prediction"  : prediction_label,
            "confidence"  : confidence,
            "temperature" : round(float(current["temp"]), 1),
            "humidity"    : int(current["humidity"]),
            "description" : str(current["description"])
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════
# ENDPOINT 3 — ANOMALY DETECTION
# GET /api/predictions/anomaly?city=Mumbai
# Runs Isolation Forest on live weather data
# Returns list of anomalous weather readings
# ════════════════════════════════════════════════════════

@router.get("/anomaly")
def detect_anomaly(city: str = "Mumbai"):
    """
    Detects anomalous weather patterns using Isolation Forest.
    Returns which readings are unusual.
    """
    try:
        # check if isolation forest model is loaded
        if "isolation_forest" not in models:
            raise HTTPException(
                status_code=500,
                detail="Isolation Forest model not loaded"
            )

        # fetch live weather data
        logger.info(f"Running anomaly detection for {city}")
        df = fetcher.fetch_weather(city)

        # preprocess data
        X, y = preprocessor.prepare(df, target="hour_category")

        # run isolation forest
        model = models["isolation_forest"]
        predictions = model.predict(X)

        # isolation forest returns:
        # 1  = normal
        # -1 = anomaly
        anomaly_indices = [i for i, p in enumerate(predictions) if p == -1]

        # build anomaly details
        anomalies = []
        for idx in anomaly_indices:
            row = df.iloc[idx]
            anomalies.append({
                "index"      : idx,
                "time"       : str(row["dt"]),
                "temperature": round(float(row["temp"]), 1),
                "humidity"   : int(row["humidity"]),
                "wind_speed" : round(float(row["wind_speed"]), 1),
                "description": str(row["description"])
            })

        return {
            "city"         : city,
            "total_records": len(df),
            "anomaly_count": len(anomalies),
            "anomalies"    : anomalies
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════
# ENDPOINT 4 — SENTIMENT ANALYSIS
# GET /api/predictions/sentiment?city=Mumbai
# Runs HuggingFace sentiment on live news
# ════════════════════════════════════════════════════════

@router.get("/sentiment")
def get_sentiment(city: str = "Mumbai"):
    """
    Runs sentiment analysis on live news headlines.
    Returns positive/negative breakdown.
    """
    try:
        # import nlp pipeline
        from nlp_pipeline import analyze_sentiment

        # fetch live news
        logger.info(f"Running sentiment analysis for {city}")
        news_df = fetcher.fetch_news(f"{city} city")

        # run sentiment analysis
        result_df = analyze_sentiment(news_df)

        # count sentiments
        pos_count = int((result_df["sentiment"] == "POSITIVE").sum())
        neg_count = int((result_df["sentiment"] == "NEGATIVE").sum())
        total     = len(result_df)

        # build headlines list
        headlines = []
        for _, row in result_df.iterrows():
            headlines.append({
                "title"    : str(row["title"]),
                "sentiment": str(row["sentiment"]),
                "score"    : float(row["sentiment_score"])
            })

        return {
            "city"           : city,
            "total_headlines": total,
            "positive_count" : pos_count,
            "negative_count" : neg_count,
            "positive_pct"   : round(pos_count / total * 100, 1) if total > 0 else 0,
            "headlines"      : headlines
        }

    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════
# ENDPOINT 5 — NAMED ENTITY RECOGNITION
# GET /api/predictions/entities?city=Mumbai
# ════════════════════════════════════════════════════════

@router.get("/entities")
def get_entities(city: str = "Mumbai"):
    """
    Extracts named entities from live news headlines.
    Returns places, organizations, people mentioned.
    """
    try:
        # import nlp pipeline
        from nlp_pipeline import extract_entities

        # fetch live news
        logger.info(f"Running NER for {city}")
        news_df = fetcher.fetch_news(f"{city} city")

        # run NER
        entities = extract_entities(news_df)

        # count total entities
        total = sum(len(v) for v in entities.values())

        return {
            "city"          : city,
            "total_entities": total,
            "entities"      : entities
        }

    except Exception as e:
        logger.error(f"NER failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════
# ENDPOINT 6 — TOPIC MODELING
# GET /api/predictions/topics?city=Mumbai
# ════════════════════════════════════════════════════════

@router.get("/topics")
def get_topics(city: str = "Mumbai"):
    """
    Discovers topics in live news headlines using LDA.
    Returns top words for each topic.
    """
    try:
        # import nlp pipeline
        from nlp_pipeline import model_topics

        # fetch live news
        logger.info(f"Running topic modeling for {city}")
        news_df = fetcher.fetch_news(f"{city} city")

        # run topic modeling
        topics = model_topics(news_df, n_topics=3)

        return {
            "city"      : city,
            "num_topics": len(topics),
            "topics"    : topics
        }

    except Exception as e:
        logger.error(f"Topic modeling failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))