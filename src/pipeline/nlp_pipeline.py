"""
NLP Pipeline — SmartCity Pulse
Sentiment analysis, Named Entity Recognition, and Topic Modeling
on live news headlines about the city.
"""

# import pandas for data manipulation
import pandas as pd

# import numpy for math operations
import numpy as np

# import transformers pipeline for sentiment analysis
from transformers import pipeline

# import spacy for named entity recognition
import spacy

# import gensim for LDA topic modeling
from gensim import corpora
from gensim.models import LdaModel

# import nltk for text preprocessing
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# import logging
import logging

# import sys
import sys

# create logger for this file
logger = logging.getLogger(__name__)

# download required nltk data quietly
nltk.download("punkt",     quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)


# ════════════════════════════════════════════════════════
# PART 1 — SENTIMENT ANALYSIS
# uses HuggingFace pre-trained model
# classifies each headline as POSITIVE or NEGATIVE
# ════════════════════════════════════════════════════════

def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes sentiment of news headlines using HuggingFace.

    Args:
        df: News DataFrame from data_fetcher.py
            must have 'title' column

    Returns:
        DataFrame with sentiment and score columns added
    """

    # log that sentiment analysis is starting
    logger.info("Running sentiment analysis...")

    # load pre-trained sentiment model
    # distilbert is a small fast version of BERT
    # runs on CPU without GPU
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,      # truncate long headlines to max length
        max_length=512        # maximum tokens per headline
    )

    # lists to store results
    sentiments = []
    scores     = []

    # loop through each headline
    for title in df["title"]:

        # skip if title is missing
        if pd.isna(title) or title == "":
            sentiments.append("UNKNOWN")
            scores.append(0.0)
            continue

        # run sentiment model on headline
        result = sentiment_model(str(title))[0]

        # extract label — POSITIVE or NEGATIVE
        sentiments.append(result["label"])

        # extract confidence score — 0 to 1
        scores.append(round(result["score"], 4))

    # add results to DataFrame
    df = df.copy()
    df["sentiment"] = sentiments
    df["sentiment_score"] = scores

    # count positive and negative
    pos_count = sentiments.count("POSITIVE")
    neg_count = sentiments.count("NEGATIVE")

    # log summary
    logger.info(f"Sentiment — Positive: {pos_count} | Negative: {neg_count}")

    # print results
    print(f"\n✅ Sentiment Analysis")
    print(f"   Total headlines : {len(df)}")
    print(f"   Positive        : {pos_count}")
    print(f"   Negative        : {neg_count}")
    print(f"\n   Sample results:")

    # print first 5 results
    for _, row in df.head(5).iterrows():
        print(f"   [{row['sentiment']:8s} {row['sentiment_score']:.2f}] {row['title'][:60]}...")

    return df


# ════════════════════════════════════════════════════════
# PART 2 — NAMED ENTITY RECOGNITION
# uses spaCy to find places and organizations in news
# ════════════════════════════════════════════════════════

def extract_entities(df: pd.DataFrame) -> dict:
    """
    Extracts named entities from news headlines using spaCy.
    Finds people, places, organizations, dates mentioned in news.

    Args:
        df: News DataFrame with 'title' column

    Returns:
        Dictionary with entity types and their counts
    """

    # log that NER is starting
    logger.info("Running Named Entity Recognition...")

    # load spacy english model
    # en_core_web_sm is the small english model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # if model not found download it
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")

    # dictionary to store all entities found
    all_entities = {
        "GPE"  : [],    # countries, cities, states
        "ORG"  : [],    # organizations, companies
        "PERSON": [],   # people names
        "LOC"  : [],    # locations, mountains, rivers
        "DATE" : []     # dates and time expressions
    }

    # loop through each headline
    for title in df["title"]:

        # skip missing titles
        if pd.isna(title):
            continue

        # process text with spacy
        doc = nlp(str(title))

        # loop through entities found
        for ent in doc.ents:

            # only keep entity types we care about
            if ent.label_ in all_entities:
                all_entities[ent.label_].append(ent.text)

    # print results
    print(f"\n✅ Named Entity Recognition")
    for entity_type, entities in all_entities.items():

        # skip empty entity types
        if not entities:
            continue

        # count occurrences of each entity
        from collections import Counter
        counts = Counter(entities).most_common(5)

        print(f"\n   {entity_type}:")
        for entity, count in counts:
            print(f"      {entity} ({count}x)")

    # log summary
    total_entities = sum(len(v) for v in all_entities.values())
    logger.info(f"NER found {total_entities} entities total")

    return all_entities


# ════════════════════════════════════════════════════════
# PART 3 — TOPIC MODELING
# uses LDA to find hidden topics in news headlines
# ════════════════════════════════════════════════════════

def model_topics(df: pd.DataFrame, n_topics: int = 3) -> list:
    """
    Discovers hidden topics in news headlines using LDA.

    Args:
        df: News DataFrame with 'title' column
        n_topics: number of topics to find

    Returns:
        List of topics with their top words
    """

    # log that topic modeling is starting
    logger.info(f"Running LDA Topic Modeling with {n_topics} topics...")

    # get english stopwords — words to remove
    stop_words = set(stopwords.words("english"))

    # add custom stopwords — common news words that add no meaning
    custom_stops = {
        "says", "said", "new", "one", "two", "three",
        "year", "years", "day", "days", "time", "also",
        "could", "would", "may", "will", "get", "got"
    }
    stop_words.update(custom_stops)

    # preprocess each headline
    processed_docs = []

    for title in df["title"]:

        # skip missing titles
        if pd.isna(title):
            continue

        # tokenize — split into individual words
        tokens = word_tokenize(str(title).lower())

        # filter tokens — keep only meaningful words
        filtered = [
            word for word in tokens
            if word.isalpha()           # only alphabetic words
            and word not in stop_words  # remove stopwords
            and len(word) > 2           # remove very short words
        ]

        # only add if we have enough words
        if len(filtered) > 1:
            processed_docs.append(filtered)

    # check if we have enough documents
    if len(processed_docs) < n_topics:
        print(f"\n⚠️  Not enough documents for {n_topics} topics")
        return []

    # create dictionary — maps each word to an ID
    dictionary = corpora.Dictionary(processed_docs)

    # create bag of words corpus
    # each document becomes a list of (word_id, word_count) tuples
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    # train LDA model
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,   # number of topics to find
        passes=15,             # number of training passes
        random_state=42        # reproducibility
    )

    # extract topics
    topics = []

    print(f"\n✅ LDA Topic Modeling")
    print(f"   Topics found: {n_topics}")

    # loop through each topic
    for topic_id in range(n_topics):

        # get top 5 words for this topic
        top_words = lda_model.show_topic(topic_id, topn=5)

        # extract just the words without weights
        words = [word for word, weight in top_words]

        # store topic
        topics.append({
            "topic_id": topic_id,
            "words"   : words
        })

        # print topic
        print(f"\n   Topic {topic_id + 1}: {' | '.join(words)}")

    # log summary
    logger.info(f"LDA found {n_topics} topics successfully")

    return topics


# ════════════════════════════════════════════════════════
# PART 4 — GEMINI CHATBOT
# uses Google Gemini with live city context
# answers questions about the city intelligently
# ════════════════════════════════════════════════════════

def create_city_context(weather_df: pd.DataFrame,
                        news_df: pd.DataFrame,
                        city: str = "Mumbai") -> str:
    """
    Creates a context string combining live weather and news data.
    This context is sent to Gemini so it can answer city questions.

    Args:
        weather_df: Weather DataFrame from data_fetcher.py
        news_df: News DataFrame from data_fetcher.py
        city: City name

    Returns:
        Formatted context string for Gemini
    """

    # get current weather from first row
    current = weather_df.iloc[0]

    # get next 3 weather readings for forecast
    forecast = weather_df.head(4).iloc[1:]

    # format current weather
    weather_context = f"""
CURRENT WEATHER IN {city.upper()}:
- Temperature  : {current['temp']:.1f}°C (feels like {current['feels_like']:.1f}°C)
- Humidity     : {current['humidity']}%
- Wind Speed   : {current['wind_speed']} m/s
- Condition    : {current['description']}
- Rain (1h)    : {current['rain_1h']} mm

WEATHER FORECAST (next 9 hours):"""

    # add forecast readings
    for _, row in forecast.iterrows():
        weather_context += f"""
- {row['dt'].strftime('%H:%M')} : {row['temp']:.1f}°C, {row['description']}"""

    # format top 5 news headlines
    news_context = f"\n\nLATEST NEWS ABOUT {city.upper()}:"
    for _, row in news_df.head(5).iterrows():
        news_context += f"\n- {row['title']}"

    # combine everything
    full_context = weather_context + news_context

    return full_context


def chat_with_gemini(question: str,
                     weather_df: pd.DataFrame,
                     news_df: pd.DataFrame,
                     city: str = "Mumbai") -> str:
    """
    Answers city-related questions using Gemini with live data.
    """

    # import new google genai package
    try:
        from google import genai
    except ImportError:
        return "Please install google-genai: pip install google-genai"

    # import os and dotenv
    import os
    from dotenv import load_dotenv

    # load environment variables
    load_dotenv()

    # get gemini api key
    api_key = os.getenv("GEMINI_API_KEY")

    # check if key exists
    if not api_key:
        return "GEMINI_API_KEY not found in .env file"

    # create client with api key
    client = genai.Client(api_key=api_key)

    # create live city context
    context = create_city_context(weather_df, news_df, city)

    # create full prompt
    prompt = f"""You are SmartCity Pulse — an intelligent urban assistant.
You have access to real-time data about {city}.

{context}

Based on this real-time data, please answer the following question:
{question}

Give a helpful, concise, and accurate answer based on the data provided.
"""

    # send to gemini and get response
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    # return response text
    return response.text
# ════════════════════════════════════════════════════════
# TEST BLOCK
# runs only when you run this file directly
# ════════════════════════════════════════════════════════

if __name__ == "__main__":

    # add pipeline folder to path
    sys.path.append("src/pipeline")

    # import data fetcher
    from data_fetcher import CityDataFetcher

    # fetch live data
    print("Fetching live data...")
    fetcher = CityDataFetcher()

    # fetch weather data
    weather_df = fetcher.fetch_weather("Mumbai")

    # fetch news data
    news_df = fetcher.fetch_news("Mumbai city")

    print(f"Weather records : {len(weather_df)}")
    print(f"News articles   : {len(news_df)}")

    # run sentiment analysis on news
    print("\n" + "="*50)
    print("SENTIMENT ANALYSIS")
    print("="*50)
    news_with_sentiment = analyze_sentiment(news_df)

    # run named entity recognition
    print("\n" + "="*50)
    print("NAMED ENTITY RECOGNITION")
    print("="*50)
    entities = extract_entities(news_df)

    # run topic modeling
    print("\n" + "="*50)
    print("TOPIC MODELING")
    print("="*50)
    topics = model_topics(news_df, n_topics=3)

    # test chatbot
    print("\n" + "="*50)
    print("GEMINI CHATBOT TEST")
    print("="*50)

    # test questions
    questions = [
        "What is the current weather in Mumbai?",
        "Should I carry an umbrella today?",
        "What are the latest news headlines about Mumbai?"
    ]

    # ask each question
    for question in questions:
        print(f"\n❓ Question: {question}")
        answer = chat_with_gemini(question, weather_df, news_df)
        print(f"🤖 Answer: {answer}")
        print("-" * 50)