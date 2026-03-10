"""
Schemas — SmartCity Pulse API
Pydantic models for request and response validation.
Pydantic automatically validates all data coming in and going out.
If wrong data is sent, FastAPI returns a clear error message.
"""

# import pydantic for data validation
from pydantic import BaseModel

# import typing for optional fields
from typing import Optional, List

# ════════════════════════════════════════════════════════
# WHAT IS A SCHEMA?
#
# A schema defines the SHAPE of data.
# Like a form that must be filled correctly.
#
# Example:
# PredictionRequest says:
#   - city must be a string
#   - model_name must be a string
#
# If React sends city=123 (a number) FastAPI
# automatically rejects it and returns an error.
# ════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════
# REQUEST SCHEMAS
# These define what data the CLIENT sends TO the server
# ════════════════════════════════════════════════════════

class PredictionRequest(BaseModel):
    """
    Request body for ML prediction endpoint.
    React sends this when asking for a prediction.
    """
    # city name to fetch weather for
    city: str = "Mumbai"

    # which model to use for prediction
    # default is random_forest — our best classical model
    model_name: str = "random_forest"


class ChatRequest(BaseModel):
    """
    Request body for Gemini chatbot endpoint.
    React sends this when user types a question.
    """
    # the question the user typed
    question: str

    # city context for the chatbot
    city: str = "Mumbai"


class AnomalyRequest(BaseModel):
    """
    Request body for anomaly detection endpoint.
    """
    # city to check for anomalies
    city: str = "Mumbai"


# ════════════════════════════════════════════════════════
# RESPONSE SCHEMAS
# These define what data the SERVER sends BACK to client
# ════════════════════════════════════════════════════════

class WeatherResponse(BaseModel):
    """
    Response from weather endpoint.
    Sent back to React after fetching weather.
    """
    # city name
    city           : str

    # current temperature in celsius
    temperature    : float

    # feels like temperature
    feels_like     : float

    # humidity percentage
    humidity       : int

    # wind speed in m/s
    wind_speed     : float

    # weather description
    description    : str

    # rain in last hour
    rain_1h        : float

    # total records fetched
    total_records  : int


class PredictionResponse(BaseModel):
    """
    Response from ML prediction endpoint.
    Contains the model prediction and confidence.
    """
    # city that was predicted for
    city           : str

    # model that was used
    model_used     : str

    # predicted class — Night Morning Afternoon Evening
    prediction     : str

    # confidence score 0 to 1
    confidence     : float

    # current temperature
    temperature    : float

    # current humidity
    humidity       : int


class SentimentResponse(BaseModel):
    """
    Response from sentiment analysis endpoint.
    """
    # city analyzed
    city           : str

    # total headlines analyzed
    total_headlines: int

    # count of positive headlines
    positive_count : int

    # count of negative headlines
    negative_count : int

    # percentage positive
    positive_pct   : float

    # list of headline results
    headlines      : List[dict]


class EntityResponse(BaseModel):
    """
    Response from NER endpoint.
    """
    # city analyzed
    city           : str

    # total entities found
    total_entities : int

    # entities grouped by type
    entities       : dict


class TopicResponse(BaseModel):
    """
    Response from topic modeling endpoint.
    """
    # city analyzed
    city           : str

    # number of topics found
    num_topics     : int

    # list of topics with their words
    topics         : List[dict]


class ChatResponse(BaseModel):
    """
    Response from Gemini chatbot endpoint.
    """
    # the original question
    question       : str

    # Gemini answer
    answer         : str

    # city context used
    city           : str


class AnomalyResponse(BaseModel):
    """
    Response from anomaly detection endpoint.
    """
    # city analyzed
    city           : str

    # total records checked
    total_records  : int

    # number of anomalies found
    anomaly_count  : int

    # list of anomaly details
    anomalies      : List[dict]


class ErrorResponse(BaseModel):
    """
    Standard error response.
    Returned when something goes wrong.
    """
    # error status
    status         : str = "error"

    # error message
    message        : str

    # optional details
    detail         : Optional[str] = None