import math
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tools.api import get_prices, prices_to_df
from utils.state import AgentState
from utils.debug import show_agent_reasoning
from tools.api import get_prices, prices_to_df
from utils.progress import progress
from typing import Optional, Dict, Any



# Implement proper ADX calculation (replace placeholder)
def calculate_adx(df, window=14):
    high, low, close = df['high'], df['low'], df['close']
    tr1 = abs(high - low)
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    plus_di = 100 * (plus_dm.ewm(span=window).mean() / atr)
    minus_di = 100 * (abs(minus_dm).ewm(span=window).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(span=window).mean()
    
    return pd.DataFrame({"adx": adx})





def calculate_trend_signals(df):
    if df.empty or len(df) < 60:
        return {"signal": "neutral", "confidence": 0.5, "metrics": {}}

    ema_8, ema_21, ema_55 = calculate_ema(df, 8), calculate_ema(df, 21), calculate_ema(df, 55)
    adx = calculate_adx(df)["adx"]
    short, medium = ema_8 > ema_21, ema_21 > ema_55
    trend_strength = adx.iloc[-1] / 100.0
    if short.iloc[-1] and medium.iloc[-1]: s, c = "bullish", trend_strength
    elif not short.iloc[-1] and not medium.iloc[-1]: s, c = "bearish", trend_strength
    else: s, c = "neutral", 0.5
    return {"signal": s, "confidence": c, "metrics": {"adx": float(adx.iloc[-1]), "trend_strength": float(trend_strength)}}

def calculate_mean_reversion_signals(df):
    if df.empty or len(df) < 60:
        return {"signal": "neutral", "confidence": 0.5, "metrics": {}}

    ma, std = df["close"].rolling(50).mean(), df["close"].rolling(50).std()
    z = (df["close"] - ma) / std
    bb_upper, bb_lower = calculate_bollinger_bands(df)
    rsi_14, rsi_28 = calculate_rsi(df, 14), calculate_rsi(df, 28)
    price_vs_bb = (df["close"].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
    if z.iloc[-1] < -2 and price_vs_bb < 0.2: s, c = "bullish", min(abs(z.iloc[-1])/4, 1.0)
    elif z.iloc[-1] > 2 and price_vs_bb > 0.8: s, c = "bearish", min(abs(z.iloc[-1])/4, 1.0)
    else: s, c = "neutral", 0.5
    return {"signal": s, "confidence": c, "metrics": {"z_score": float(z.iloc[-1]), "price_vs_bb": float(price_vs_bb), "rsi_14": float(rsi_14.iloc[-1]), "rsi_28": float(rsi_28.iloc[-1])}}

def calculate_momentum_signals(df):
    if df.empty or len(df) < 60:
        return {"signal": "neutral", "confidence": 0.5, "metrics": {}}

    returns = df["close"].pct_change()
    mom_1m, mom_3m, mom_6m = returns.rolling(21).sum(), returns.rolling(63).sum(), returns.rolling(126).sum()
    volume_ma = df["volume"].rolling(21).mean()
    vol_mom = df["volume"] / volume_ma
    score = (0.4 * mom_1m + 0.3 * mom_3m + 0.3 * mom_6m).iloc[-1]
    volume_confirm = vol_mom.iloc[-1] > 1.0
    if score > 0.05 and volume_confirm: s, c = "bullish", min(abs(score)*5, 1.0)
    elif score < -0.05 and volume_confirm: s, c = "bearish", min(abs(score)*5, 1.0)
    else: s, c = "neutral", 0.5
    return {"signal": s, "confidence": c, "metrics": {"momentum_1m": float(mom_1m.iloc[-1]), "momentum_3m": float(mom_3m.iloc[-1]), "momentum_6m": float(mom_6m.iloc[-1]), "volume_momentum": float(vol_mom.iloc[-1])}}

def calculate_volatility_signals(df):
    if df.empty or len(df) < 60:
        return {"signal": "neutral", "confidence": 0.5, "metrics": {}}

    returns = df["close"].pct_change()
    hist_vol = returns.rolling(21).std() * math.sqrt(252)
    vol_ma = hist_vol.rolling(63).mean()
    vol_regime = hist_vol / vol_ma
    vol_z = (hist_vol - vol_ma) / hist_vol.rolling(63).std()
    atr = calculate_atr(df)
    atr_ratio = atr / df["close"]
    vr, vz = vol_regime.iloc[-1], vol_z.iloc[-1]
    if vr < 0.8 and vz < -1: s, c = "bullish", min(abs(vz)/3, 1.0)
    elif vr > 1.2 and vz > 1: s, c = "bearish", min(abs(vz)/3, 1.0)
    else: s, c = "neutral", 0.5
    return {"signal": s, "confidence": c, "metrics": {"volatility_regime": float(vr), "volatility_z_score": float(vz), "atr_ratio": float(atr_ratio.iloc[-1])}}

def calculate_stat_arb_signals(df):
    if df.empty or len(df) < 60:
        return {"signal": "neutral", "confidence": 0.5, "metrics": {}}

    returns = df["close"].pct_change()
    skew, kurt = returns.rolling(63).skew(), returns.rolling(63).kurt()
    hurst = calculate_hurst_exponent(df["close"])
    if hurst < 0.4 and skew.iloc[-1] > 1: s, c = "bullish", (0.5 - hurst)*2
    elif hurst < 0.4 and skew.iloc[-1] < -1: s, c = "bearish", (0.5 - hurst)*2
    else: s, c = "neutral", 0.5
    return {"signal": s, "confidence": c, "metrics": {"hurst_exponent": float(hurst), "skewness": float(skew.iloc[-1]), "kurtosis": float(kurt.iloc[-1])}}

def weighted_signal_combination(signals, weights):
    map_val = {"bullish": 1, "neutral": 0, "bearish": -1}
    ws, tc = 0, 0
    for strat, signal in signals.items():
        if not isinstance(signal, dict) or "signal" not in signal or "confidence" not in signal:
            continue  # skip faulty strategy signal
        sig_val = map_val.get(signal["signal"], 0)
        conf = signal["confidence"]
        weight = weights.get(strat, 0)
        ws += sig_val * weight * conf
        tc += weight * conf
    score = ws / tc if tc else 0
    signal = "bullish" if score > 0.2 else "bearish" if score < -0.2 else "neutral"
    return {"signal": signal, "confidence": abs(score)}

def minimal_signal(signal):
    return {
        "signal": signal["signal"],
        "confidence": round(signal["confidence"] * 100),
        "metrics": normalize(signal["metrics"]),
    }

def normalize(obj):
    if isinstance(obj, pd.Series): return obj.tolist()
    if isinstance(obj, pd.DataFrame): return obj.to_dict("records")
    if isinstance(obj, dict): return {k: normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [normalize(v) for v in obj]
    return obj

def calculate_ema(df, window): return df["close"].ewm(span=window, adjust=False).mean()

def calculate_rsi(df, period):
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_bollinger_bands(df):
    ma = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    return ma + 2 * std, ma - 2 * std

def calculate_atr(df):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(14).mean()


def calculate_hurst_exponent(series):
    # Placeholder for Hurst exponent logic
    return 0.5

from datetime import datetime, timedelta
from tools.api import get_prices, prices_to_df

def get_technical_signal(
    ticker: str,
    state: Optional[Dict[str, Any]] = None,
    days_shift: int = 3  # Shift both start & end dates back by this many days
) -> str:
    """
    Returns technical signal (neutral/bullish/bearish) for a given ticker.
    - Uses original logic but shifts the date window back by `days_shift`.
    - Default: 90-day window, shifted back by 3 days (now covering -93 to -3 days ago).
    """
    # Default date range (90-day window shifted back by `days_shift`)
    end_date = datetime.now() - timedelta(days=days_shift)
    start_date = end_date - timedelta(days=90)  # Original 90-day lookback

    # If state is provided, use its dates (but still shift them back)
    if state:
        original_start = state.get("data", {}).get("start_date", datetime.now() - timedelta(days=90))
        original_end = state.get("data", {}).get("end_date", datetime.now())
        start_date = original_start - timedelta(days=days_shift)
        end_date = original_end - timedelta(days=days_shift)

    # Fetch and prepare data
    prices = get_prices(ticker=ticker, start_date=start_date.strftime("%Y-%m-%d"), end_date=end_date.strftime("%Y-%m-%d"))
    df = prices_to_df(prices)

    # Fallback to neutral if insufficient data
    if df.empty or len(df) < 10:
        return "neutral"

    # Calculate all technical signals (original logic)
    ts = calculate_trend_signals(df)
    mr = calculate_mean_reversion_signals(df)
    mo = calculate_momentum_signals(df)
    vo = calculate_volatility_signals(df)
    sa = calculate_stat_arb_signals(df)

    # Combine signals with weights
    weights = {"trend": 0.25, "mean_reversion": 0.2, "momentum": 0.25, "volatility": 0.15, "stat_arb": 0.15}
    combined = weighted_signal_combination(
        {"trend": ts, "mean_reversion": mr, "momentum": mo, "volatility": vo, "stat_arb": sa},
        weights
    )

    # Convert to simple signal (neutral if confidence < 50%)
    if combined["confidence"] < 50:
        return "neutral"
    return combined["signal"].lower()  # Ensures "bullish" or "bearish"

if __name__ == "__main__":
        signal = get_technical_signal("MSFT")  # or any other ticker symbol
        print(f"Technical signal for AAPL: {signal}")
