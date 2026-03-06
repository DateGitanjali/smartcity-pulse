"""
Data Fetcher — SmartCity Pulse
Fetches real-time city data from external APIs.
Author: ER.Gitanjali
Date: Today
"""
import os
import pandas as pd
import logging
import requests
import datetime 
from typing import Optional
from dotenv import load_dotenv

# Load your .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)#s automatically the file's name

class CityDataFetcher:
    """
    Fetches real-time city data from weather and news APIs.

    Usage:
        fetcher = CityDataFetcher()
        df = fetcher.fetch_weather("Mumbai")
    """
    WEATHER_URL = "https://api.openweathermap.org/data/2.5/forecast"
    NEWS_URL    = "https://newsapi.org/v2/everything"
    
    def __init__(self):
        self.weather_key = os.getenv("OPENWEATHER_API_KEY")
        self.news_key = os.getenv("NEWS_API_KEY")
        
        if not self.weather_key:
            raise ValueError(
                "OPENWEATHER_API_KEY NOT FOUND."
                "check your .env file and ensure the key is set."
            )
      
   
    def fetch_weather(self, city: str) -> Optional[pd.DataFrame]:
        """
        Fetch 5-day weather forecast for a city.

        Args:
            city: City name e.g. "Mumbai", "Delhi", "London"

        Returns:
            DataFrame with weather data or None if failed.
        """
        try:
            params = {
                "q":      city,
                "appid":  self.weather_key,
                "units":  "metric",
                "cnt":    40
            }
            response = requests.get(self.WEATHER_URL,
                                    params=params,
                                    timeout=10)
            response.raise_for_status()
            
            raw = response.json()
            records = []
            
            for item in raw['list']:
                records.append(
                    {
                    "dt" : pd.to_datetime(item["dt"] , unit="s"),
                    "temp" : item["main"]["temp"],
                    "feels_like" : item["main"]["feels_like"],
                    "humidity" : item["main"]["humidity"],
                    "wind_speed" : item["wind"]["speed"],
                    "rain_1h" : item.get("rain" , {}).get("1h" ,0),
                    "description" : item["weather"][0]["description"]
                       
                    })
            df = pd.DataFrame(records)
            logger.info(f"Fetched {len(df)} weather records for {city}")
            return df

        except requests.exceptions.Timeout:
            logger.error(f"Weather API timed out for {city}")
            return None

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            return None
        
        except Exception as e:
            logger.error(f"Unexcepted error : {e}")
            return None
    def fetch_news(self,query:str) -> Optional[pd.DataFrame]:
        try:
            params = {
                "q" : query,
                "apikey" : self.news_key,
                'pageSize' : 20,
                "language" : "en"
            }
            
            response = requests.get(
                self.NEWS_URL,
                params = params,
                timeout=10
            )
            response.raise_for_status()
            raw = response.json()
            records = []
            
            for article in raw["articles"]:
                records.append({
                    "title": article["title"],
                    "description" : article["description"],
                    'published':article["publishedAt"],
                    "source":article["source"]["name"]
                })
                
            df = pd.DataFrame(records)
            logger.info(f"Fetched {len(df) }news articles for '{query}'")
            return df
        
        except Exception as e :
            logger.error(f"News API error: {e}")
            return None
if __name__ == "__main__":
    fetcher = CityDataFetcher()
    
    print("\n -- WEATHER DATA -- ")
    weather_df = fetcher.fetch_weather("Mumbai")
    if weather_df is not None :
        print(weather_df.head())
        print(f"shape:{weather_df.shape}")
        print(f"Columns:{list(weather_df.columns)}")
        
    print("\n--NEWS DATA --")
    news_df = fetcher.fetch_news("Mumbai city")
    if news_df is not None:
        print(news_df.head())
        print(f"Shape:{news_df.shape}")
        
    