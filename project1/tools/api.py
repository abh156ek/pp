import os
import pandas as pd
import requests
import yfinance as yf
from alpaca_trade_api.rest import REST, TimeFrame
from dotenv import load_dotenv
load_dotenv()

from data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
)

# Alpaca API setup
alpaca_api_key = os.environ.get("ALPACA_API_KEY")
alpaca_api_secret = os.environ.get("ALPACA_API_SECRET")
alpaca_client = REST(alpaca_api_key, alpaca_api_secret, base_url="https://paper-api.alpaca.markets")

def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch price data from Yahoo Finance or Alpaca."""
    try:
        # First, try Yahoo Finance
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if not stock_data.empty:
            prices = [
                Price(time=str(date), open=row['Open'], high=row['High'], low=row['Low'], close=row['Close'], volume=row['Volume'])
                for date, row in stock_data.iterrows()
            ]
            return prices
    except Exception as e:
        print(f"Error fetching data from Yahoo Finance: {e}")

    # If Yahoo Finance fails, try Alpaca
    try:
        alpaca_data = alpaca_client.get_barset(ticker, TimeFrame.Day, start=start_date, end=end_date).get(ticker, [])
        if alpaca_data:
            prices = [
                Price(time=str(bar.t), open=bar.o, high=bar.h, low=bar.l, close=bar.c, volume=bar.v)
                for bar in alpaca_data
            ]
            return prices
    except Exception as e:
        print(f"Error fetching data from Alpaca: {e}")
    
    return []  # Return empty list if both fail

def get_financial_metrics(ticker, end_date, period="ttm", limit=10):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return [FinancialMetrics(
            ticker=ticker,
            report_period=end_date,
            period=period,
            market_cap=info.get("marketCap"),
            price_to_earnings_ratio=info.get("trailingPE"),
            earnings_per_share=info.get("trailingEps"),
        )]
    except Exception as e:
        print(f"Error fetching metrics for {ticker}: {e}")
        return []
        
def search_line_items(ticker: str, line_items: list[str], end_date: str, period: str = "ttm", limit: int = 10) -> list[LineItem]:
    """Fetch line items from API (you can implement this similarly for Yahoo/Alpaca)."""
    return []

def get_insider_trades(ticker: str, end_date: str, start_date: str | None = None, limit: int = 1000) -> list[InsiderTrade]:
    """Fetch insider trades (this might not be available via Yahoo or Alpaca, so consider using another service)."""
    return []

def get_company_news(ticker: str, end_date: str, start_date: str | None = None, limit: int = 1000) -> list[CompanyNews]:
    """Fetch company news from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        # Filter news by date if necessary
        filtered_news = [
            CompanyNews(date=news_item['providerPublishTime'], headline=news_item['title'], url=news_item['link'])
            for news_item in news if news_item['providerPublishTime'] <= end_date
        ]
        return filtered_news[:limit]
    except Exception as e:
        print(f"Error fetching news from Yahoo Finance: {e}")
    return []

def get_market_cap(ticker: str, end_date: str) -> float | None:
    """Fetch market cap from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        market_cap = stock.info.get("marketCap", None)
        return market_cap
    except Exception as e:
        print(f"Error fetching market cap from Yahoo Finance: {e}")
    return None

def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)
