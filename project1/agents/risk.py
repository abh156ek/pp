import os
import requests
from typing import Dict, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

from pydantic import BaseModel

# --- 🔐 Load API Keys ---
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
BASE_URL = "https://data.alpaca.markets/v2/stocks"

class RiskAnalysis(BaseModel):
    current_price: float
    portfolio_value: float
    current_position: float
    position_limit: float
    remaining_limit: float
    available_cash: float

class RiskScoreInput(BaseModel):
    ticker: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    cost_basis: Optional[float] = None
    cash: Optional[float] = None

# --- 🔧 Get Historical Prices ---
def get_prices(ticker: str, start_date: str, end_date: str) -> list:
    """Fetch historical price data from Alpaca API"""
    url = f"{BASE_URL}/{ticker}/bars?start={start_date}&end={end_date}&timeframe=1Day"
    headers = {
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": API_SECRET,
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        return [{"close": bar["c"]} for bar in data.get("bars", [])]
    except requests.exceptions.RequestException as e:
        print(f"[API Error] {ticker}: {str(e)}")
        return []
    except KeyError as e:
        print(f"[Data Format Error] {ticker}: {str(e)}")
        return []

# --- 🧠 Risk Analysis Core ---
def analyze_position_risk(ticker: str, 
                        cost_basis: float,
                        cash: float,
                        start_date: str,
                        end_date: str) -> Optional[RiskAnalysis]:
    """Core risk analysis for a single position"""
    prices = get_prices(ticker, start_date, end_date)
    if not prices:
        return None

    current_price = prices[-1]["close"]
    total_value = cash + cost_basis
    position_limit_pct = 0.20
    position_limit = total_value * position_limit_pct
    remaining_limit = position_limit - cost_basis

    return RiskAnalysis(
        current_price=current_price,
        portfolio_value=total_value,
        current_position=cost_basis,
        position_limit=position_limit,
        remaining_limit=remaining_limit,
        available_cash=cash
    )

# --- 📉 Risk Score Calculation ---
def get_risk_score(input_data: RiskScoreInput) -> float:
    """Calculate risk score based on position size and limits
    
    Args:
        input_data: RiskScoreInput object containing ticker and optional parameters
        
    Returns:
        float: Risk score between 0.0 and 1.0
    """
    ticker = input_data.ticker
    
    # Default values
    cost_basis = input_data.cost_basis or 10000.0
    cash = input_data.cash or 50000.0
    
    # Set default dates if not provided
    if not input_data.start_date or not input_data.end_date:
        end_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=93)).strftime("%Y-%m-%d")
    else:
        start_date = input_data.start_date
        end_date = input_data.end_date

    try:
        # Perform risk analysis
        analysis = analyze_position_risk(
            ticker=ticker,
            cost_basis=cost_basis,
            cash=cash,
            start_date=start_date,
            end_date=end_date
        )
        
        if not analysis:
            print(f"[Warning] Could not get risk data for {ticker}")
            return 0.5
            
        # Calculate risk score based on position compared to limit
        if analysis.position_limit <= 0:
            return 0.5

        risk_ratio = min(analysis.current_position / analysis.position_limit, 1.0)
        score = round(risk_ratio, 2)
        
        print(f"\n✅ Risk Score for {ticker}: {score}")
        return score

    except Exception as e:
        print(f"[Risk Calculation Error] {ticker}: {str(e)}")
        return 0.5

# --- 🧪 Standalone Testing ---
if __name__ == "__main__":
    # Example usage when run directly
    test_input = RiskScoreInput(ticker="AAPL")
    score = get_risk_score(test_input)
    print(f"\n🧠 Risk Score for {test_input.ticker}: {score}")