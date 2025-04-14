# Hedge Fund Portfolio Analysis System (Single Input + Radar Chart)

from dotenv import load_dotenv
load_dotenv()

from agents.sentiment import get_sentiment_score,fetch_recent_news  
from agents.risk import get_risk_score, RiskScoreInput
from agents.fundamental import get_fundamentals
from agents.technicals import get_technical_signal

from typing import Dict
from pydantic import BaseModel
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- Add this at the top with other imports ---
import google.generativeai as genai
import os

# --- Configure Gemini API ---
def configure_gemini():
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    return genai.GenerativeModel('gemini-2.0-flash')

# --- ✅ Final Verdict Section (Modified) ---
def generate_final_verdict(agent_output: Dict):
    model = configure_gemini()
    
    t, d = list(agent_output.items())[0]
    
    # Prepare the data for Gemini
    analysis_data = {
        "Ticker": t,
        "Sentiment Score": d['sentiment'],
        "Risk Score": d['risk'],
        "Technical Signal": d['technical'],
        "Fundamentals": d.get('fundamentals', {})
    }
    
    prompt = f"""
    You are a professional hedge fund analyst. Based on the following analysis data, provide:
    1. A final verdict (strictly only one word: 'Buy', 'Sell', or 'Hold')
    2. A concise reasoning (1-2 sentences) supporting your verdict
    
    Analysis Data:
    {analysis_data}
    
    Guidelines:
    - Sentiment Score ranges from -1 (very negative) to 1 (very positive)
    - Risk Score ranges from 0 (low risk) to 1 (high risk)
    - Technical Signal is either 'Bullish', 'Bearish', or 'Neutral'
    - Consider fundamentals like P/E ratio, earnings, etc.
    
    Your response must be in this exact format:
    VERDICT: [Buy/Sell/Hold]
    REASONING: [your reasoning here]
    """
    
    try:
        response = model.generate_content(prompt)
        result = response.text.strip()
        
        # Parse the response
        verdict = "Neutral"  # default
        reasoning = "Unable to determine"  # default
        
        if "VERDICT:" in result and "REASONING:" in result:
            verdict = result.split("VERDICT:")[1].split("REASONING:")[0].strip()
            reasoning = result.split("REASONING:")[1].strip()
        elif "Buy" in result or "Sell" in result or "Hold" in result:
            verdict = result.split("\n")[0].strip()
            reasoning = "\n".join(result.split("\n")[1:]).strip()
        
        print("\n✅ Final Verdict:", verdict)
        print("📌 Reasoning:", reasoning)
        return verdict, reasoning
        
    except Exception as e:
        print("\n⚠️ Could not generate AI verdict. Using fallback analysis.")
        print("Error:", str(e))
        
        # Fallback basic logic
        sentiment = d['sentiment']
        risk = d['risk']
        
        if sentiment > 0.5 and risk < 0.5:
            verdict = "Buy"
            reasoning = "Strong positive sentiment with manageable risk"
        elif sentiment < -0.5 and risk > 0.7:
            verdict = "Sell"
            reasoning = "Negative sentiment combined with high risk"
        else:
            verdict = "Hold"
            reasoning = "Mixed signals or neutral indicators"
            
        print("\n✅ Final Verdict:", verdict)
        print("📌 Reasoning:", reasoning)
        return verdict, reasoning

# --- 📊 Input Model ---
class PortfolioInput(BaseModel):
    ticker: str

# --- 🧠 Run All Agents ---
def run_all_agents(portfolio: PortfolioInput) -> Dict:
    t = portfolio.ticker
    risk_input = RiskScoreInput(ticker=t)
    fundamentals_data = get_fundamentals(t)
    return {
        t: {
            "sentiment": get_sentiment_score(t),
            "risk": get_risk_score(risk_input),
            "fundamentals": fundamentals_data,
            "technical": get_technical_signal(t),
        }
    }

def analyze_ticker(ticker):
    portfolio_input = PortfolioInput(ticker=ticker)
    agent_output = run_all_agents(portfolio_input)
    verdict, reasoning = generate_final_verdict(agent_output)
    headlines = fetch_recent_news(ticker)  # This comes from sentiment.py
    
    # You can also extract parts like risk, sentiment, etc.
    t, d = list(agent_output.items())[0]
    
    return {
        "sentiment": d["sentiment"],
        "risk": d["risk"],
        "technical": d["technical"],
        "fundamentals": d["fundamentals"],
        "verdict": verdict,
        "reasoning": reasoning,
        "agent_output": agent_output,
        "headlines": headlines
    }


# --- 📈 Combined Radar Chart ---
# --- 📈 Combined Radar Chart ---
def plot_radar_chart(agent_output: Dict):
    t, d = list(agent_output.items())[0]

    sentiment = (d['sentiment'] + 1) / 2  # Normalize -1 to 1 → 0 to 1
    risk = d['risk']  # Already 0 to 1

    fundamentals = d.get('fundamentals', {})
    pe_ratio = fundamentals.get('pe_ratio', 0)
    earnings = fundamentals.get('earnings', 0)

    # Normalize P/E and Earnings (assume max reasonable value)
    pe_ratio = min(pe_ratio / 50, 1.0) if pe_ratio else 0
    earnings = min(earnings / 10, 1.0) if earnings else 0

    # Technical signal to number
    tech_signal = d.get('technical', 'neutral')
    tech_score = {"bearish": 0, "neutral": 0.5, "bullish": 1.0}.get(tech_signal.lower(), 0.5)

    metrics = {
        "Sentiment": sentiment,
        "Risk": risk,
        "P/E Ratio": pe_ratio,
        "Earnings": earnings,
        "Technical": tech_score
    }

    labels = list(metrics.keys())
    values = list(metrics.values())
    
    # Number of variables we're plotting
    num_vars = len(labels)
    
    # Calculate angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # The plot is circular, so we need to "complete the loop" by appending the start to the end
    values += values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.fill(angles, values, color='skyblue', alpha=0.4)

    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Go through labels and adjust alignment based on where it is in the circle
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        if angle in (0, np.pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')

    # Ensure y-labels go from 0 to 1
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    
    ax.set_title(f"📊 Overall Profile: {t}", size=14, y=1.1)
    plt.tight_layout()
    return fig
# --- 📋 Text Summary ---
def generate_summary(agent_output: Dict):
    print("\n📋 Summary Report")
    print("=" * 40)
    for t, d in agent_output.items():
        print(f"\n🔹 {t}")
        print(f"  Sentiment: {d['sentiment']}")
        print(f"  Risk: {d['risk']}")
        print(f"  Technical Signal: {d['technical']}")

        f = d.get("fundamentals", {})
        if isinstance(f, dict):
            print("  Fundamentals:")
            print(f"    - Earnings per Share: {f.get('earnings', 'N/A')}")
            print(f"    - P/E Ratio: {f.get('pe_ratio', 'N/A')}")
            print(f"    - P/B Ratio: {f.get('pb_ratio', 'N/A')}")
            print(f"    - Signal: {f.get('signal', 'neutral')}")
        else:
            print("  Fundamentals: No data available")


# --- 🚀 Entry Point ---
if __name__ == "__main__":
    user_input = input("Enter a ticker: ").strip().upper()
    portfolio_input = PortfolioInput(ticker=user_input)

    agent_output = run_all_agents(portfolio_input)
    plot_radar_chart(agent_output)
    generate_summary(agent_output)
    generate_final_verdict(agent_output)  # Changed from print_final_verdict()