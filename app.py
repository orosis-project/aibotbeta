# app.py
# Final Version: Backend with Initial Buy-in and Delayed AI Learning
# and fixed thread startup for Gunicorn

import os
import time
import requests
import json
import sqlite3
import google.generativeai as genai
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from threading import Thread, Lock
from datetime import datetime, timedelta
import random
import re

# --- Configuration ---
# NOTE: Render will inject a PORT environment variable, which the code
# below already correctly handles with os.environ.get('PORT', 8000).

# IMPORTANT: You should set these as environment variables in Render's dashboard.
# DO NOT store them directly in your code for security reasons.
# GEMINI_API_KEY = "AIzaSyCFShQd4JEqv8AQUqtDyQ7iCDNWMHjId_c"
# FINNHUB_API_KEY = "d25mi11r01qhge4das6gd25mi11r01qhge4das70"

# Using os.environ to get keys from Render's environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
INITIAL_CASH = 5000.00
DB_FILE = "trades.db"
TRADE_AMOUNT_USD = 500
LOOP_INTERVAL_SECONDS = 300  # 5 minutes
STOCKS_TO_SCAN_PER_CYCLE = 10
# Advanced Risk Management: Adjust these values to control risk appetite
CONFIDENCE_MULTIPLIER = 2.0  # Scales the trade size by confidence.
MIN_CONFIDENCE_FOR_RISKY_TRADE = 0.8  # AI must be this confident for a large trade.
# Delayed AI Learning configuration
AI_LEARNING_TRADE_THRESHOLD = 3
INITIAL_BUY_COUNT = 10
AI_LEARNING_ENABLED = False  # Global flag for AI learning status

# --- Bot State ---
bot_status_lock = Lock()
bot_is_running = True # Bot starts in a running state by default

# --- AI Configuration ---
ai_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        ai_model = genai.GenerativeModel('gemini-1.5-flash')
        print("Gemini AI model configured successfully.")
    except Exception as e:
        print(f"ERROR: Failed to configure Gemini AI: {e}")
else:
    print("WARNING: Gemini API key not set. AI features will be disabled.")


# --- Database Functions ---
# NOTE: For a production app on Render, you might consider an external database
# like Render's PostgreSQL or an S3 bucket to store the SQLite file, as the
# local filesystem is ephemeral and resets on each redeploy. For this example,
# the SQLite file will be created on each deploy.
def get_db_connection():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            reasoning TEXT,
            confidence REAL
        )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized.")

def get_trade_count():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(id) FROM trades")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        print(f"ERROR: Could not count trades: {e}")
        return 0

def get_recent_trades(limit=50):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,))
        trades = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return trades
    except Exception as e:
        print(f"ERROR: Could not fetch recent trades: {e}")
        return []

# --- Flask App ---
app = Flask(__name__)
CORS(app)

# Global Error Handler: Ensures all API errors return JSON, not HTML
@app.errorhandler(Exception)
def handle_exception(e):
    # Log the error for debugging on Render's side
    print(f"An unexpected error occurred: {e}")
    # Return a JSON response with a 500 status code
    response = {
        "error": "An internal server error occurred.",
        "details": str(e)
    }
    return jsonify(response), 500

# --- Portfolio Manager ---
class PortfolioManager:
    def __init__(self, initial_cash, api_client, db_file):
        self._lock = Lock()
        self.initial_cash = initial_cash
        self.api_client = api_client
        self.db_file = db_file
        self.reset()
        print("Portfolio Manager initialized.")

    def reset(self):
        with self._lock:
            # Check if the database file exists. If not, this is a fresh start.
            is_fresh_start = not os.path.exists(self.db_file)

            self.cash = self.initial_cash
            self.stocks = {}
            self.initial_value = self.initial_cash
            if os.path.exists(self.db_file):
                try:
                    os.remove(self.db_file)
                except OSError as e:
                    print(f"Error removing database file: {e}")
            init_db()
            print("Portfolio has been reset.")
            
            global AI_LEARNING_ENABLED
            AI_LEARNING_ENABLED = False

            # If this is the very first time the app is running, buy 10 random stocks.
            if is_fresh_start:
                self.buy_initial_10_stocks()

    def buy_initial_10_stocks(self):
        """Buys 10 random stocks to bootstrap the portfolio and trade history."""
        sp500 = self.api_client.get_sp500_constituents()
        if not sp500:
            print("Failed to fetch S&P 500 list for initial stocks. Cannot perform initial buy.")
            return

        # Select 10 random stocks to start with
        stocks_to_buy = random.sample(sp500, INITIAL_BUY_COUNT)
        for symbol in stocks_to_buy:
            print(f"Initiating a forced initial buy of {symbol}...")
            price = self.api_client.get_quote(symbol)
            if price:
                quantity = TRADE_AMOUNT_USD / price
                self.buy_stock(
                    symbol=symbol,
                    quantity=quantity,
                    price=price,
                    reasoning="Forced buy to seed portfolio with 10 random stocks.",
                    confidence=0.1
                )
            else:
                print(f"Could not get quote for {symbol}. Skipping initial buy.")

    def get_portfolio_status(self):
        with self._lock:
            stock_values = 0.0
            detailed_stocks = {}
            for symbol, data in self.stocks.items():
                current_price = self.api_client.get_quote(symbol) or data['avg_price']
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
                "cash": self.cash, "owned_stocks": detailed_stocks,
                "total_portfolio_value": total_value, "profit_loss": profit_loss
            }

    def buy_stock(self, symbol, quantity, price, reasoning, confidence):
        with self._lock:
            cost = quantity * price
            if self.cash < cost: return False
            self.cash -= cost
            if symbol in self.stocks:
                total_qty = self.stocks[symbol]['quantity'] + quantity
                self.stocks[symbol]['avg_price'] = ((self.stocks[symbol]['avg_price'] * self.stocks[symbol]['quantity']) + cost) / total_qty
                self.stocks[symbol]['quantity'] = total_qty
            else:
                self.stocks[symbol] = {'quantity': quantity, 'avg_price': price}
            self._log_trade(symbol, 'BUY', quantity, price, reasoning, confidence)
            print(f"SUCCESS: Bought {quantity:.4f} of {symbol} @ ${price:.2f}")
            return True

    def sell_stock(self, symbol, quantity, price, reasoning, confidence):
        with self._lock:
            if symbol not in self.stocks or self.stocks[symbol]['quantity'] < quantity: return False
            self.cash += quantity * price
            self.stocks[symbol]['quantity'] -= quantity
            if self.stocks[symbol]['quantity'] < 1e-6: del self.stocks[symbol]
            self._log_trade(symbol, 'SELL', quantity, price, reasoning, confidence)
            print(f"SUCCESS: Sold {quantity:.4f} of {symbol} @ ${price:.2f}")
            return True

    def _log_trade(self, symbol, action, quantity, price, reasoning, confidence):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('INSERT INTO trades (symbol, action, quantity, price, reasoning, confidence) VALUES (?, ?, ?, ?, ?, ?)',
                           (symbol, action, quantity, price, reasoning, confidence))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"ERROR: Failed to log trade: {e}")

# --- Finnhub Client ---
class FinnhubClient:
    def __init__(self, api_key):
        self.api_key = api_key
        # Hardcoded list of S&P 500 symbols to avoid the forbidden API call on the free plan
        self.sp500_symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "GOOG", "BRK.B",
            "UNH", "JPM", "JNJ", "V", "XOM", "MA", "PG", "HD", "CVX", "LLY", "ABBV",
            "PFE", "BAC", "KO", "TMO", "PEP", "AVGO", "WMT", "COST", "MCD", "CSCO"
        ]
        print("Finnhub Client initialized.")
    def _make_request(self, endpoint, params=None):
        if params is None: params = {}
        params['token'] = self.api_key
        try:
            r = requests.get(f"{FINNHUB_BASE_URL}/{endpoint}", params=params)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Finnhub request failed: {e}")
            return None
    def get_quote(self, symbol):
        data = self._make_request('quote', {'symbol': symbol.upper()})
        return data.get('c') if data else None
    
    def get_stock_candles(self, symbol, resolution="D", days=20):
        to_ts = int(time.time())
        from_ts = to_ts - (days * 24 * 60 * 60)
        return self._make_request('stock/candle', {'symbol': symbol.upper(), 'resolution': resolution, 'from': from_ts, 'to': to_ts})

    def get_company_news(self, symbol):
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        return self._make_request('company-news', {'symbol': symbol.upper(), 'from': from_date, 'to': to_date})

    def get_market_news(self):
        return self._make_request('news', {'category': 'general'})
    
    def get_sp500_constituents(self):
        return self.sp500_symbols

# --- Helper Functions for Indicators & Analysis ---
def calculate_sma(candles, days=20):
    if not candles or 'c' not in candles or len(candles['c']) < days:
        return None
    closing_prices = candles['c'][-days:]
    return sum(closing_prices) / len(closing_prices) if closing_prices else None

def analyze_sentiment(news_headlines):
    positive_keywords = ['exceeds', 'surges', 'strong', 'growth', 'positive', 'boosts', 'upgraded']
    negative_keywords = ['misses', 'sinks', 'weak', 'decline', 'negative', 'downgraded']
    
    sentiment_score = 0
    for headline in news_headlines:
        for keyword in positive_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', headline, re.IGNORECASE):
                sentiment_score += 1
        for keyword in negative_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', headline, re.IGNORECASE):
                sentiment_score -= 1
    
    return "Positive" if sentiment_score > 0 else ("Negative" if sentiment_score < 0 else "Neutral")

# --- AI Decision Making ---
def get_ai_decision(symbol, price, news, portfolio, past_performance, market_news, sma):
    if not ai_model: return None
    
    news_headlines = [f"- {item['headline']}" for item in news[:5]] if news else ["No recent news for this stock."]
    market_headlines = [f"- {item['headline']}" for item in market_news[:5]] if market_news else ["No general market news."]
    
    sentiment = analyze_sentiment(news_headlines)
    
    startup_mode_prompt = ""
    if past_performance['new_trades_count'] < AI_LEARNING_TRADE_THRESHOLD:
        startup_mode_prompt = f"""
        **Current Operational Directive: Initial Learning Phase**
        The system has executed its initial {INITIAL_BUY_COUNT} trades to seed the portfolio. Your primary mission now is to make an additional 3 trades based on strong signals to begin a robust learning cycle. You must identify promising BUY opportunities to add new positions. Prioritize making a well-reasoned trade over waiting, with a slightly lower confidence threshold to ensure action. Once the 3 new trades are complete, you will transition to a more advanced decision-making mode.
        """

    prompt = f"""
    You are an expert stock trading analyst bot. Your goal is to learn from your actions and maximize portfolio value.
    {startup_mode_prompt}
    **Current Portfolio Status:**
    - Cash: ${portfolio['cash']:.2f}
    - Total Portfolio Value: ${portfolio['total_portfolio_value']:.2f}
    - Owned Stocks: {json.dumps(portfolio['owned_stocks'], indent=2)}
    - Total Profit/Loss: ${portfolio['profit_loss']:.2f}

    **Your Trading History (Your Memory & Learning Data):**
    - Your last 5 trades: {json.dumps(past_performance['recent_trades'], indent=2)}
    - Summary of your trade history:
        - Total trades: {past_performance['total_trades']}
        - Total profit/loss from all trades: ${past_performance['total_profit_loss']:.2f}
        - Total profit from winning trades: ${past_performance['total_profit_from_winning_trades']:.2f}
        - Number of winning trades: {past_performance['winning_trades']}
        - Number of losing trades: {past_performance['losing_trades']}
        - New trades count since initial seeding: {past_performance['new_trades_count']}

    **General Market News (Overall Sentiment):**
    {chr(10).join(market_headlines)}
    
    **Stock to Analyze:** {symbol}
    - Current Price: ${price:.2f}
    - 20-Day Simple Moving Average (SMA): ${sma:.2f}
    - Recent News Headlines for {symbol} (Sentiment: {sentiment}):
    {chr(10).join(news_headlines)}
    
    **Decision Logic & Learning:**
    1. Assess the general market sentiment from the market news.
    2. Review your own trade history to learn from past successes and failures.
    3. Analyze the specific stock. Compare the current price to the SMA to understand its recent trend (e.g., above SMA is bullish, below is bearish).
    4. Consider the news sentiment for the stock. A strong positive sentiment could be a buy signal.
    5. Based on all available data (market sentiment, your memory, stock price vs. SMA, and specific stock news/sentiment), make a strategic decision.
    6. **BUY:** If signals are strong, the market outlook is favorable, and you have cash. Consider a large, "smart" risk trade if confidence is high.
    7. **SELL:** If signals are negative, the market is turning, or to secure profits. Only sell if you own the stock.
    8. **HOLD:** If signals are mixed or waiting is the best strategy.
    
    **Your Response MUST be in the following JSON format ONLY:**
    {{
      "action": "BUY", "symbol": "{symbol}", "confidence": 0.85,
      "reasoning": "The recent positive news headlines and a bullish technical signal (price is above the 20-day SMA) create a compelling buy opportunity. My past profitable trades in this sector support taking a calculated, aggressive position."
    }}
    Provide your analysis and decision now.
    """
    try:
        response = ai_model.generate_content(prompt)
        decision_text = response.text.strip().replace("```json", "").replace("```", "")
        decision = json.loads(decision_text)
        print(f"AI Decision for {symbol}: {decision}")
        return decision
    except Exception as e:
        print(f"ERROR: Failed to get AI chat response: {e}")
        return jsonify({"answer": "I encountered an error."}), 500

# --- New Route for Admin Panel ---
@app.route("/pannel")
def admin_pannel():
    return render_template("admin_pannel.html")


@app.before_first_request
def start_bot_thread():
    """Starts the bot thread before the first request is served."""
    # This is a good solution for a single-threaded server.
    # In a multi-process Gunicorn environment, this will run for each worker.
    # This is an acceptable trade-off for simplicity in a non-critical context.
    if GEMINI_API_KEY and FINNHUB_API_KEY:
        bot_thread = Thread(target=bot_trading_loop, args=(portfolio_manager, finnhub_client), daemon=True)
        bot_thread.start()
    else:
        print("WARNING: API keys not set. Bot loop will not start.")


if __name__ == "__main__":
    init_db()
    # For local development only. Gunicorn will handle this in production.
    # Use 'gunicorn app:app --bind 0.0.0.0:$PORT' as the start command on Render.
    port = int(os.environ.get('PORT', 8000))
    print(f"Flask app is running on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, threaded=True, debug=False)
