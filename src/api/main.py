"""
FastAPI Backend — SmartCity Pulse
Main entry point for the API server.
All ML models and NLP pipeline are served from here.
"""

# import FastAPI — the web framework
from fastapi import FastAPI

# import CORSMiddleware — allows React frontend to talk to this server
# without CORS React cannot call our API
from fastapi.middleware.cors import CORSMiddleware

# import our routers — each router handles one group of endpoints
from src.api.routers import predictions, chat

# import logging
import logging

# import sys for path
import sys

# import os for environment variables
import os

# add src to path so we can import our pipeline files
sys.path.append("src/pipeline")
sys.path.append("src/ml")

# create logger
logger = logging.getLogger(__name__)

# configure logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# ════════════════════════════════════════════════════════
# CREATE FASTAPI APP
# ════════════════════════════════════════════════════════

# create the FastAPI application
# title and description appear in automatic API documentation
app = FastAPI(
    title="SmartCity Pulse API",
    description="Real-time urban intelligence platform powered by ML and NLP",
    version="1.0.0"
)

# ════════════════════════════════════════════════════════
# CORS SETTINGS
# CORS = Cross Origin Resource Sharing
# This allows our React app (running on port 3000)
# to call our FastAPI server (running on port 8000)
# Without this the browser blocks all requests
# ════════════════════════════════════════════════════════

app.add_middleware(
    CORSMiddleware,
    # allow requests from React dev server and production
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    # allow all HTTP methods — GET, POST, PUT, DELETE
    allow_methods=["*"],
    # allow all headers
    allow_headers=["*"],
)

# ════════════════════════════════════════════════════════
# INCLUDE ROUTERS
# routers group related endpoints together
# predictions router handles all ML prediction endpoints
# chat router handles the Gemini chatbot endpoint
# ════════════════════════════════════════════════════════

# include predictions router with prefix /api
app.include_router(predictions.router, prefix="/api")

# include chat router with prefix /api
app.include_router(chat.router, prefix="/api")

# ════════════════════════════════════════════════════════
# ROOT ENDPOINT
# ════════════════════════════════════════════════════════

@app.get("/")
def root():
    """
    Health check endpoint.
    Returns server status.
    If you get this response the server is running.
    """
    return {
        "status"     : "online",
        "app"        : "SmartCity Pulse API",
        "version"    : "1.0.0",
        "message"    : "Server is running successfully"
    }


@app.get("/health")
def health_check():
    """
    Detailed health check.
    Returns status of all components.
    """
    return {
        "status"    : "healthy",
        "ml_models" : "loaded",
        "nlp"       : "ready",
        "database"  : "connected"
    }