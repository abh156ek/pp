import sys
import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import questionary
import matplotlib.pyplot as plt
import pandas as pd
from colorama import Fore, Style, init
import numpy as np
from llm.models import LLM_ORDER, get_model_info
from utils.analysts import ANALYST_ORDER
from main import run_hedge_fund
from tools.api import (
    get_company_news,
    get_price_data,
    get_prices,
    get_financial_metrics,
    get_insider_trades,
)
from utils.display import print_backtest_results, format_backtest_row
from typing_extensions import Callable

init(autoreset=True)

class Backtester:
    def __init__(self, agent: Callable, tickers: list[str], start_date: str, end_date: str, initial_capital: float, 
                 model_name: str = "gpt-4o", model_provider: str = "OpenAI", selected_analysts: list[str] = [], 
                 initial_margin_requirement: float = 0.0):
        self.agent = agent
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.model_name = model_name
        self.model_provider = model_provider
        self.selected_analysts = selected_analysts
        self.portfolio_values = []
        self.portfolio = {
            "cash": initial_capital,
            "margin_used": 0.0,
            "margin_requirement": initial_margin_requirement,
            "positions": {ticker: {"long": 0, "short": 0, "long_cost_basis": 0.0, "short_cost_basis": 0.0, "short_margin_used": 0.0} for ticker in tickers},
            "realized_gains": {ticker: {"long": 0.0, "short": 0.0} for ticker in tickers}
        }

    def execute_trade(self, ticker: str, action: str, quantity: float, current_price: float):
        if quantity <= 0: return 0
        quantity = int(quantity)
        position = self.portfolio["positions"][ticker]

        if action == "buy":
            cost = quantity * current_price
            if cost <= self.portfolio["cash"]:
                old_shares, old_cost_basis = position["long"], position["long_cost_basis"]
                total_shares = old_shares + quantity
                position["long_cost_basis"] = ((old_cost_basis * old_shares) + cost) / total_shares if total_shares > 0 else 0
                position["long"] += quantity
                self.portfolio["cash"] -= cost
                return quantity
            max_quantity = int(self.portfolio["cash"] / current_price)
            if max_quantity > 0:
                cost = max_quantity * current_price
                old_shares, old_cost_basis = position["long"], position["long_cost_basis"]
                total_shares = old_shares + max_quantity
                position["long_cost_basis"] = ((old_cost_basis * old_shares) + cost) / total_shares if total_shares > 0 else 0
                position["long"] += max_quantity
                self.portfolio["cash"] -= cost
                return max_quantity
            return 0

        elif action == "sell":
            quantity = min(quantity, position["long"])
            if quantity > 0:
                realized_gain = (current_price - position["long_cost_basis"]) * quantity
                self.portfolio["realized_gains"][ticker]["long"] += realized_gain
                position["long"] -= quantity
                self.portfolio["cash"] += quantity * current_price
                if position["long"] == 0:
                    position["long_cost_basis"] = 0.0
                return quantity

        elif action == "short":
            proceeds, margin_required = current_price * quantity, current_price * quantity * self.portfolio["margin_requirement"]
            if margin_required <= self.portfolio["cash"]:
                old_short_shares, old_cost_basis = position["short"], position["short_cost_basis"]
                total_shares = old_short_shares + quantity
                position["short_cost_basis"] = ((old_cost_basis * old_short_shares) + (current_price * quantity)) / total_shares if total_shares > 0 else 0
                position["short"] += quantity
                position["short_margin_used"] += margin_required
                self.portfolio["margin_used"] += margin_required
                self.portfolio["cash"] += proceeds - margin_required
                return quantity
            max_quantity = int(self.portfolio["cash"] / (current_price * self.portfolio["margin_requirement"]))
            if max_quantity > 0:
                proceeds, margin_required = current_price * max_quantity, current_price * max_quantity * self.portfolio["margin_requirement"]
                old_short_shares, old_cost_basis = position["short"], position["short_cost_basis"]
                total_shares = old_short_shares + max_quantity
                position["short_cost_basis"] = ((old_cost_basis * old_short_shares) + (current_price * max_quantity)) / total_shares if total_shares > 0 else 0
                position["short"] += max_quantity
                position["short_margin_used"] += margin_required
                self.portfolio["margin_used"] += margin_required
                self.portfolio["cash"] += proceeds - margin_required
                return max_quantity
            return 0

        elif action == "cover":
            quantity = min(quantity, position["short"])
            if quantity > 0:
                cover_cost, margin_to_release = quantity * current_price, (quantity / position["short"]) * position["short_margin_used"]
                realized_gain = (position["short_cost_basis"] - current_price) * quantity
                position["short"] -= quantity
                position["short_margin_used"] -= margin_to_release
                self.portfolio["margin_used"] -= margin_to_release
                self.portfolio["cash"] += margin_to_release - cover_cost
                self.portfolio["realized_gains"][ticker]["short"] += realized_gain
                if position["short"] == 0:
                    position["short_cost_basis"], position["short_margin_used"] = 0.0, 0.0
                return quantity
        return 0

    def calculate_portfolio_value(self, current_prices):
        total_value = self.portfolio["cash"]
        for ticker in self.tickers:
            position = self.portfolio["positions"][ticker]
            total_value += position["long"] * current_prices[ticker]
            if position["short"] > 0:
                total_value += position["short"] * (position["short_cost_basis"] - current_prices[ticker])
        return total_value

    def prefetch_data(self):
        print("\nPre-fetching data for the entire backtest period...")
        start_date = (datetime.strptime(self.end_date, "%Y-%m-%d") - relativedelta(years=1)).strftime("%Y-%m-%d")
        for ticker in self.tickers:
            get_prices(ticker, start_date, self.end_date)
            get_financial_metrics(ticker, self.end_date, limit=10)
            get_insider_trades(ticker, self.end_date, start_date=self.start_date, limit=1000)
            get_company_news(ticker, self.end_date, start_date=self.start_date, limit=1000)
        print("Data pre-fetch complete.")

    def parse_agent_response(self, agent_output):
        try:
            if isinstance(agent_output, dict):
                return agent_output
            return json.loads(agent_output)
        except Exception as e:
            print(f"Error parsing agent output: {e}")
            return {"action": "hold", "quantity": 0}



    
    def run_backtest(self):
        self.prefetch_data()
        dates = pd.date_range(self.start_date, self.end_date, freq="B")
        table_rows = []
        performance_metrics = { 
            'sharpe_ratio': None, 'sortino_ratio': None, 'max_drawdown': None, 
            'long_short_ratio': None, 'gross_exposure': None, 'net_exposure': None
        }

        print("\nStarting backtest...")
        self.portfolio_values = [{"Date": dates[0], "Portfolio Value": self.initial_capital}]

        for current_date in dates:
            lookback_start = (current_date - timedelta(days=30)).strftime("%Y-%m-%d")
            current_date_str, previous_date_str = current_date.strftime("%Y-%m-%d"), (current_date - timedelta(days=1)).strftime("%Y-%m-%d")

            if lookback_start == current_date_str:
                continue

            current_prices = {}
            missing_data = False
            for ticker in self.tickers:
                try:
                    price_data = get_price_data(ticker, previous_date_str, current_date_str)
                    if price_data.empty:
                        print(f"Warning: No price data for {ticker} on {current_date_str}")
                        missing_data = True
                        break
                    current_prices[ticker] = price_data.iloc[-1]["close"]
                except Exception as e:
                    print(f"Error fetching price for {ticker} between {previous_date_str} and {current_date_str}: {e}")
                    missing_data = True
                    break
            if missing_data:
                print(f"Skipping trading day {current_date_str} due to missing price data")
                continue

            # Run agent to get trading decisions
            agent_input = {
                "tickers": self.tickers,
                "start_date": lookback_start,
                "end_date": current_date_str,
                "portfolio": self.portfolio
            }
            
            try:
                agent_output = self.agent(
                    tickers=self.tickers,
                    model_name=self.model_name,
                    model_provider=self.model_provider,
                    selected_analysts=self.selected_analysts
                )
                
                # Process agent output and execute trades
                for ticker in self.tickers:
                    if ticker in agent_output:
                        action = agent_output[ticker].get("action", "hold")
                        quantity = agent_output[ticker].get("quantity", 0)
                        
                        if action in ["buy", "sell", "short", "cover"] and ticker in current_prices:
                            self.execute_trade(ticker, action, quantity, current_prices[ticker])
                
                # Calculate and store portfolio value
                portfolio_value = self.calculate_portfolio_value(current_prices)
                self.portfolio_values.append({"Date": current_date, "Portfolio Value": portfolio_value})
                
                # Format and store table row for display
                table_rows.append(format_backtest_row(
                    current_date, 
                    current_prices, 
                    self.portfolio, 
                    agent_output, 
                    portfolio_value
                ))
                
            except Exception as e:
                print(f"Error during agent execution on {current_date_str}: {e}")
            
        # Calculate performance metrics
        returns = [
            (pv["Portfolio Value"] / self.portfolio_values[i]["Portfolio Value"]) - 1 
            for i, pv in enumerate(self.portfolio_values[1:], 1)
        ]
        
        if returns:
            performance_metrics["sharpe_ratio"] = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            negative_returns = [r for r in returns if r < 0]
            performance_metrics["sortino_ratio"] = np.mean(returns) / np.std(negative_returns) if negative_returns and np.std(negative_returns) > 0 else 0
        
            # Calculate maximum drawdown
            cum_returns = np.cumprod(1 + np.array(returns))
            max_drawdown = 0
            peak = cum_returns[0]
            
            for ret in cum_returns:
                if ret > peak:
                    peak = ret
                drawdown = (peak - ret) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            performance_metrics["max_drawdown"] = max_drawdown
            
            # Calculate exposure metrics
            long_exposure = sum(p["long"] * current_prices[t] for t, p in self.portfolio["positions"].items() if t in current_prices)
            short_exposure = sum(p["short"] * current_prices[t] for t, p in self.portfolio["positions"].items() if t in current_prices)
            
            if long_exposure > 0 or short_exposure > 0:
                performance_metrics["long_short_ratio"] = long_exposure / short_exposure if short_exposure > 0 else float('inf')
                performance_metrics["gross_exposure"] = (long_exposure + short_exposure) / self.calculate_portfolio_value(current_prices)
                performance_metrics["net_exposure"] = (long_exposure - short_exposure) / self.calculate_portfolio_value(current_prices)
        
        # Print backtest results
        print_backtest_results(table_rows, performance_metrics)
        
        # Create a DataFrame for portfolio values and plot results
        portfolio_df = pd.DataFrame(self.portfolio_values)
        portfolio_df["Date"] = pd.to_datetime(portfolio_df["Date"])
        portfolio_df.set_index("Date", inplace=True)
        
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_df.index, portfolio_df["Portfolio Value"])
        plt.title("Portfolio Value Over Time")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        return self.portfolio_values, performance_metrics