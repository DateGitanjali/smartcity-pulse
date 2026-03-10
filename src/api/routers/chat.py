"""
Chat Router — SmartCity Pulse API
Handles the Gemini chatbot endpoint.
User sends a question, we send back an AI answer
with live city context.
"""

# import APIRouter and HTTPException
from fastapi import APIRouter, HTTPException

# import sys for path
import sys

# add folders to path
sys.path.append("src/pipeline")
sys.path.append("src/api")

# import our schemas
from schemas import ChatRequest, ChatResponse

# import logging
import logging

# create logger
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════
# CREATE ROUTER
# ════════════════════════════════════════════════════════

router = APIRouter(
    prefix="/chat",
    tags=["chat"]
)


# ════════════════════════════════════════════════════════
# ENDPOINT — CHAT WITH GEMINI
# POST /api/chat
# User sends question → we add live city data → Gemini answers
# ════════════════════════════════════════════════════════

@router.post("/")
def chat(request: ChatRequest):
    """
    Sends user question to Gemini with live city context.
    Fetches live weather and news before calling Gemini.
    Returns AI generated answer.
    """
    try:
        # import our pipeline files
        from data_fetcher import CityDataFetcher
        from nlp_pipeline import chat_with_gemini

        # log the incoming question
        logger.info(f"Chat request: {request.question[:50]}...")

        # create fetcher
        fetcher = CityDataFetcher()

        # fetch live weather data
        logger.info(f"Fetching live data for {request.city}")
        weather_df = fetcher.fetch_weather(request.city)

        # fetch live news data
        news_df = fetcher.fetch_news(f"{request.city} city")

        # call gemini with live context
        logger.info("Calling Gemini...")
        answer = chat_with_gemini(
            question   = request.question,
            weather_df = weather_df,
            news_df    = news_df,
            city       = request.city
        )

        # return response
        return {
            "question" : request.question,
            "answer"   : answer,
            "city"     : request.city
        }

    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════
# ENDPOINT — GET CITY CONTEXT
# GET /api/chat/context?city=Mumbai
# Returns the live context without asking Gemini
# Useful for debugging what context Gemini receives
# ════════════════════════════════════════════════════════

@router.get("/context")
def get_context(city: str = "Mumbai"):
    """
    Returns the live city context string.
    Shows exactly what data is sent to Gemini.
    """
    try:
        # import pipeline
        from data_fetcher import CityDataFetcher
        from nlp_pipeline import create_city_context

        # create fetcher
        fetcher = CityDataFetcher()

        # fetch live data
        weather_df = fetcher.fetch_weather(city)
        news_df    = fetcher.fetch_news(f"{city} city")

        # create context
        context = create_city_context(weather_df, news_df, city)

        return {
            "city"   : city,
            "context": context
        }

    except Exception as e:
        logger.error(f"Context fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))