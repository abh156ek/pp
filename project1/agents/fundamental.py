import os
import json
from datetime import datetime, timedelta
import yfinance as yf
import alpaca_trade_api as tradeapi

# ✅ Load API keys from environment
API_KEY = "PK9DLADGXDHTZY5IOAFJ"
API_SECRET = "RY3LrwWYWbDFW1ZfDnvfmgXdeHrFUxPrWIcbgo8o"
BASE_URL = "https://paper-api.alpaca.markets"

# ✅ Initialize Alpaca client early so it's available everywhere
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

def get_financial_ratios(ticker):
    """Get key financial ratios for a ticker using yfinance."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "price_to_earnings_ratio": info.get("trailingPE"),
            "price_to_book_ratio": info.get("priceToBook"),
            "earnings_per_share": info.get("epsTrailingTwelveMonths"),
            "debt_to_equity": info.get("debtToEquity"),
        }
    except Exception as e:
        print(f"[Error] Failed to get financial ratios for {ticker}: {e}")
        return None

def get_historical_data(ticker, start_date, end_date):
    """Get historical price data and calculate average price change."""
    try:
        bars = api.get_bars(
            ticker,
            tradeapi.rest.TimeFrame.Day,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d')
        ).df

        if bars.empty:
            return None

        price_changes = [
            (bars.iloc[i].close - bars.iloc[i - 1].close) / bars.iloc[i - 1].close
            for i in range(1, len(bars))
        ]
        avg_price_change = sum(price_changes) / len(price_changes) if price_changes else 0
        return {"price_change_avg": avg_price_change}
    except Exception as e:
        print(f"[Error] Failed to get historical data for {ticker}: {e}")
        return None

def get_fundamentals(ticker):
    """
    Get fundamental analysis for a single ticker.
    This function is used by the main module.
    
    Returns:
        dict: Fundamental data including earnings, PE ratio, PB ratio, and signal
    """
    # Get financial ratios
    financial_ratios = get_financial_ratios(ticker)
    
    # Get historical data for signal calculation
    today = datetime.now()
    start_date = today.replace(day=1)
    end_date = today - timedelta(days=7)
    hist = get_historical_data(ticker, start_date, end_date)
    
    # Set defaults if data retrieval failed
    if not financial_ratios:
        return {"earnings": 0.0, "pe_ratio": 0.0, "pb_ratio": 0.0, "signal": "neutral"}
    
    # Extract values with fallbacks
    earnings = financial_ratios.get("earnings_per_share", 0.0) or 0.0
    pe_ratio = financial_ratios.get("price_to_earnings_ratio", 0.0) or 0.0
    pb_ratio = financial_ratios.get("price_to_book_ratio", 0.0) or 0.0
    
    # Generate signal if historical data exists
    signal = "neutral"
    if hist and "price_change_avg" in hist:
        change = hist["price_change_avg"]
        signal = (
            "bullish" if change > 0 else
            "bearish" if change < 0 else
            "neutral"
        )
    
    # Return data in format expected by main module
    return {
        "earnings": earnings,
        "pe_ratio": pe_ratio,
        "pb_ratio": pb_ratio,
        "signal": signal
    }

def fundamentals_agent(state):
    """
    Legacy function for agent-based architecture.
    Processes a batch of tickers and adds results to state.
    """
    data = state["data"]
    tickers = data.get("tickers", [])
    today = datetime.now()
    start_date = today.replace(day=1)
    end_date = today - timedelta(days=7)

    analysis = {}

    for ticker in tickers:
        hist = get_historical_data(ticker, start_date, end_date)
        ratios = get_financial_ratios(ticker)

        if not hist or not ratios:
            continue

        change = hist["price_change_avg"]
        pe = ratios["price_to_earnings_ratio"]

        signal = (
            "bullish" if change > 0 else
            "bearish" if change < 0 else
            "neutral"
        )

        analysis[ticker] = {
            "signal": signal,
            "confidence": 100,
            "reasoning": {
                "profitability_signal": {
                    "signal": signal,
                    "details": f"Avg Price Change: {change:.2%}, P/E: {pe or 'N/A'}"
                }
            }
        }

    state["data"].setdefault("analyst_signals", {})["fundamentals_agent"] = analysis
    return {
        "messages": state.get("messages", []) + [{
            "role": "function",
            "name": "fundamentals_agent",
            "content": json.dumps(analysis, indent=2)
        }],
        "data": state["data"]
    }

# === TESTING ===

if __name__ == "__main__":
    print(">> \nTesting get_fundamentals...")
    print(json.dumps(get_fundamentals("AAPL"), indent=2))

    print("\nTesting get_financial_ratios...")
    print(json.dumps(get_financial_ratios("AAPL"), indent=2))

    print("\nTesting get_historical_data...")
    today = datetime.now()
    start = today.replace(day=1)
    end = today - timedelta(days=7)
    print(json.dumps(get_historical_data("AAPL", start, end), indent=2))

    print("\nTesting fundamentals_agent with sample state...")
    sample_state = {
        "data": {"tickers": ["AAPL", "MSFT"]},
        "messages": []
    }
    result = fundamentals_agent(sample_state)
    print("\nFinal Output from fundamentals_agent:")
    print(json.dumps(result["data"]["analyst_signals"]["fundamentals_agent"], indent=2))