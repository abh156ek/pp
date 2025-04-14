import streamlit as st
import pandas as pd
import numpy as np
from main import analyze_ticker, plot_radar_chart

st.set_page_config(page_title="Hedge Fund Dashboard", layout="wide")
st.title("💼 Hedge Fund Stock Analyzer")

# First Row – Ticker Input + Analyze Button
col_input, col_button = st.columns([1, 0.5])

with col_input:
    ticker = st.text_input("Enter Ticker", key="ticker", max_chars=10)

with col_button:
    analyze = st.button("🔍 Analyze")

# If user clicks analyze
if analyze and ticker:
    data = analyze_ticker(ticker)

    # Spider Chart
    st.markdown("### 🕸️ Spider Chart")
    fig = plot_radar_chart(data["agent_output"])
    st.pyplot(fig)

    # Second Row – Risk, Headlines, Fundamentals
    col1, col2, col3 = st.columns([1, 1.8, 1.2])
    with col1:
        st.markdown("### Risk Score")
        st.text_input("", value=round(data["risk"], 3), key="risk_score")

    with col2:
        st.markdown("### Headlines")
        headlines = data.get("headlines", ["No headlines available"])
        st.text_area("", value="\n\n".join(headlines), height=200, key="headlines")


    with col3:
        f = data["fundamentals"]
        st.markdown("### Fundamentals")
        st.text_input("Earnings per Share", value=f.get("earnings", "N/A"))
        st.text_input("P/E Ratio", value=f.get("pe_ratio", "N/A"))
        st.text_input("P/B Ratio", value=f.get("pb_ratio", "N/A"))
        st.text_input("Signal", value=f.get("signal", "neutral"))

    st.markdown("---")

    # Third Row – Sentiment, Technicals
    col4, col5, col6 = st.columns([1, 0.2, 1])
    with col4:
        st.markdown("### Sentiment Score")
        st.text_input("", value=round(data["sentiment"], 3), key="sentiment_score")

    with col6:
        st.markdown("### Technical Signal")
        st.text_input("", value=data["technical"], key="technical_signal")

    # Final Verdict
    col7, col8 = st.columns([1, 2])
    with col7:
        st.markdown("### Final Verdict")
        st.text_input("", value=data["verdict"], key="final_verdict")

    with col8:
        st.markdown("### Reasoning")
        st.text_area("", value=data["reasoning"], height=200, key="reasoning")


    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.experimental_rerun()

