import os
import requests
import logging
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_recent_news(ticker: str, limit: int = 5) -> List[str]:
    url = f"https://data.alpaca.markets/v1beta1/news?symbols={ticker}&limit={limit}"
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        news_data = response.json()
        return [article.get("headline", "") for article in news_data.get("news", [])]
    except Exception as e:
        logger.error(f"[News] Error fetching news for {ticker}: {e}")
        return []

def analyze_sentiment(text: str) -> float:
    url = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

    try:
        response = requests.post(url, headers=headers, json={"inputs": text}, timeout=10)
        response.raise_for_status()
        result = response.json()
        if not result or not isinstance(result[0], list):
            return 0.0
        label, score = result[0][0]["label"], result[0][0]["score"]
        return score if label == "positive" else -score if label == "negative" else 0.0
    except Exception as e:
        logger.warning(f"[Sentiment] Failed for: '{text[:50]}...': {e}")
        return 0.0

def get_sentiment_score(ticker: str) -> float:
    headlines = fetch_recent_news(ticker)
    if not headlines:
        logger.info(f"No headlines found for {ticker}")
        return 0.0

    print(f"\n📰 Headlines for {ticker}:")
    for h in headlines:
        print(f"  • {h}")

    scores = [analyze_sentiment(h) for h in headlines]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    sentiment = round(avg_score, 3)

    print(f"📊 Sentiment Score for {ticker}: {sentiment}")
    return sentiment

# --- 🔧 Testing standalone ---
if __name__ == "__main__":
    user_input = input("Enter a single ticker symbol (e.g. AAPL): ").strip().upper()
    if user_input:
        score = get_sentiment_score(user_input)
        print(f"\n✅ Final Sentiment Score: {score}")
