# ----------------------------------------------------
# File 2: app.py
# The full updated backend code.
# ----------------------------------------------------
import os
import time
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS  # Import CORS
from threading import Thread, Lock

# --- Configuration ---
FINNHUB_API_KEY = "d25mi11r01qhge4das6gd25mi11r01qhge4das70" 
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
INITIAL_CASH = 5000.00

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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

            # Example data for testing the dashboard
            # Remove or comment this out when the bot is trading for real
            if not self.stocks:
                self.stocks = {
                    "AAPL": {"quantity": 10, "avg_price": 150.00},
                    "GOOGL": {"quantity": 5, "avg_price": 2800.00}
                }
                self.cash = 2000.00 # Adjusted cash for example

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
        print(f"INFO: Placeholder for buying {quantity} of {symbol}")
        return True

    def sell_stock(self, symbol, quantity):
        """Simulates selling a stock."""
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
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Finnhub API request failed: {e}")
            return None

    def get_quote(self, symbol):
        """Fetches the current price of a stock."""
        data = self._make_request('quote', {'symbol': symbol.upper()})
        if data and 'c' in data:
            return data['c']
        # Fallback for example data if API fails
        if symbol == "AAPL": return 175.25
        if symbol == "GOOGL": return 2850.50
        return None

    def get_company_news(self, symbol, from_date, to_date):
        """Fetches company news for a given symbol and date range."""
        params = {'symbol': symbol.upper(), 'from': from_date, 'to': to_date}
        return self._make_request('company-news', params)

# --- Global Instances ---
finnhub_client = FinnhubClient(FINNHUB_API_KEY)
portfolio_manager = PortfolioManager(INITIAL_CASH, finnhub_client)

# --- API Endpoints (Routes) ---
@app.route("/")
def index():
    """A simple welcome message to confirm the server is running."""
    return "<h1>AI Stock Bot Backend is Running (Finnhub)</h1><p>Use the API endpoints to interact with the bot.</p>"

@app.route("/api/portfolio", methods=['GET'])
def get_portfolio():
    """Returns the current status of the portfolio."""
    status = portfolio_manager.get_portfolio_status()
    return jsonify(status)

@app.route("/api/stock/quote/<symbol>", methods=['GET'])
def get_stock_quote(symbol):
    """Returns the current price for a given stock symbol."""
    price = finnhub_client.get_quote(symbol)
    if price is not None:
        return jsonify({"symbol": symbol.upper(), "price": price})
    else:
        return jsonify({"error": f"Could not retrieve price for {symbol.upper()}"}), 404

# --- Main Execution ---
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
