# app.py
# Backend server for the AI Stock Trading Bot
# Phase 1: Foundation - Server, Finnhub Connection, Portfolio Management

import os
import time
import requests
from flask import Flask, jsonify, request
from threading import Thread, Lock

# --- Configuration ---
# Using the Finnhub API. The user confirmed this choice.
FINNHUB_API_KEY = "d25mi11r01qhge4das6gd25mi11r01qhge4das70" 
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
INITIAL_CASH = 5000.00

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Portfolio Manager ---
class PortfolioManager:
    """
    Manages the bot's financial state, including cash, owned stocks,
    and overall portfolio value. This class is thread-safe.
    """
    def __init__(self, initial_cash, api_client):
        self._lock = Lock()
        self.cash = initial_cash
        self.stocks = {}  # { "symbol": {"quantity": int, "avg_price": float} }
        self.api_client = api_client
        self.initial_value = initial_cash
        print("Portfolio Manager initialized.")

    def get_portfolio_status(self):
        """
        Calculates the current total value of the portfolio (cash + stocks)
        and returns a detailed status dictionary.
        """
        with self._lock:
            stock_values = 0.0
            detailed_stocks = {}

            for symbol, data in self.stocks.items():
                current_price = self.api_client.get_quote(symbol)
                if current_price is not None:
                    value = data['quantity'] * current_price
                    stock_values += value
                    detailed_stocks[symbol] = {
                        "quantity": data['quantity'],
                        "average_buy_price": data['avg_price'],
                        "current_price": current_price,
                        "current_value": value
                    }

            total_value = self.cash + stock_values
            profit_loss = total_value - self.initial_value
            
            return {
                "cash": self.cash,
                "owned_stocks": detailed_stocks,
                "total_portfolio_value": total_value,
                "profit_loss": profit_loss
            }

    def buy_stock(self, symbol, quantity):
        """Simulates buying a stock."""
        # This is a placeholder for Phase 3.
        print(f"INFO: Placeholder for buying {quantity} of {symbol}")
        return True

    def sell_stock(self, symbol, quantity):
        """Simulates selling a stock."""
        # This is a placeholder for Phase 3.
        print(f"INFO: Placeholder for selling {quantity} of {symbol}")
        return True


# --- Finnhub API Client ---
class FinnhubClient:
    """
    A client to interact with the Finnhub API.
    Handles requests for stock quotes and company news.
    """
    def __init__(self, api_key):
        self.api_key = api_key
        print("Finnhub Client initialized.")

    def _make_request(self, endpoint, params=None):
        """Helper function to make requests to the Finnhub API."""
        if params is None:
            params = {}
        params['token'] = self.api_key
        try:
            response = requests.get(f"{FINNHUB_BASE_URL}/{endpoint}", params=params)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Finnhub API request failed: {e}")
            return None

    def get_quote(self, symbol):
        """Fetches the current price of a stock."""
        data = self._make_request('quote', {'symbol': symbol.upper()})
        if data and 'c' in data:
            return data['c']  # 'c' is the current price
        return None

    def get_company_news(self, symbol, from_date, to_date):
        """Fetches company news for a given symbol and date range."""
        params = {'symbol': symbol.upper(), 'from': from_date, 'to': to_date}
        return self._make_request('company-news', params)


# --- Global Instances ---
# We create single instances of our clients to be used across the app.
finnhub_client = FinnhubClient(FINNHUB_API_KEY)
portfolio_manager = PortfolioManager(INITIAL_CASH, finnhub_client)


# --- API Endpoints (Routes) ---
@app.route("/")
def index():
    """A simple welcome message to confirm the server is running."""
    return "<h1>AI Stock Bot Backend is Running (Finnhub)</h1><p>Use the API endpoints to interact with the bot.</p>"

@app.route("/api/portfolio", methods=['GET'])
def get_portfolio():
    """
    Returns the current status of the portfolio.
    This will be used by the admin dashboard.
    """
    status = portfolio_manager.get_portfolio_status()
    return jsonify(status)

@app.route("/api/stock/quote/<symbol>", methods=['GET'])
def get_stock_quote(symbol):
    """
    Returns the current price for a given stock symbol.
    Example: /api/stock/quote/AAPL
    """
    price = finnhub_client.get_quote(symbol)
    if price is not None:
        return jsonify({"symbol": symbol.upper(), "price": price})
    else:
        return jsonify({"error": f"Could not retrieve price for {symbol.upper()}"}), 404

# --- Main Execution ---
if __name__ == "__main__":
    # Running in debug mode is not recommended for production.
    # We use threaded=True to handle multiple requests.
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
