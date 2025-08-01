# ----------------------------------------------------
# File 2: app.py
# The full updated backend code with the AI trading loop.
# ----------------------------------------------------
import os
import time
import requests
import json
import sqlite3
import google.generativeai as genai
from flask import Flask, jsonify, request
from flask_cors import CORS
from threading import Thread, Lock
from datetime import datetime, timedelta

# --- Configuration ---
# --- IMPORTANT: PASTE YOUR GEMINI API KEY HERE ---
GEMINI_API_KEY = "AIzaSyCFShQd4JEqv8AQUqtDyQ7iCDNWMHjId_c" 
# --- IMPORTANT: FINNHUB API KEY ---
FINNHUB_API_KEY = "d25mi11r01qhge4das6gd25mi11r01qhge4das70" 

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
INITIAL_CASH = 5000.00
DB_FILE = "trades.db"
STOCKS_TO_MONITOR = ['AAPL', 'TSLA', 'NVDA', 'MSFT'] # Stocks the bot will watch
TRADE_AMOUNT_USD = 500 # Amount in USD for each buy/sell trade
LOOP_INTERVAL_SECONDS = 300 # 5 minutes between each cycle

# --- AI Configuration ---
try:
    genai.configure(api_key=GEMINI_API_KEY)
    ai_model = genai.GenerativeModel('gemini-1.5-flash')
    print("Gemini AI model configured successfully.")
except Exception as e:
    print(f"ERROR: Failed to configure Gemini AI. Please check your API key. Error: {e}")
    ai_model = None

# --- Database Setup ---
def init_db():
    """Initializes the SQLite database and creates the trades table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
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

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Portfolio Manager ---
class PortfolioManager:
    """Manages the bot's financial state and executes trades."""
    def __init__(self, initial_cash, api_client):
        self._lock = Lock()
        self.cash = initial_cash
        self.stocks = {}  # { "symbol": {"quantity": float, "avg_price": float} }
        self.api_client = api_client
        self.initial_value = initial_cash
        print("Portfolio Manager initialized.")

    def get_portfolio_status(self):
        """Calculates and returns the current portfolio status."""
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
                "cash": self.cash,
                "owned_stocks": detailed_stocks,
                "total_portfolio_value": total_value,
                "profit_loss": profit_loss
            }

    def buy_stock(self, symbol, quantity_to_buy, current_price, reasoning, confidence):
        """Buys a stock, updates portfolio, and logs the trade."""
        with self._lock:
            cost = quantity_to_buy * current_price
            if self.cash < cost:
                print(f"WARNING: Not enough cash to buy {quantity_to_buy} of {symbol}. Need {cost:.2f}, have {self.cash:.2f}")
                return False
            
            self.cash -= cost
            
            if symbol in self.stocks:
                current_quantity = self.stocks[symbol]['quantity']
                current_avg_price = self.stocks[symbol]['avg_price']
                new_total_quantity = current_quantity + quantity_to_buy
                new_avg_price = ((current_avg_price * current_quantity) + cost) / new_total_quantity
                self.stocks[symbol]['quantity'] = new_total_quantity
                self.stocks[symbol]['avg_price'] = new_avg_price
            else:
                self.stocks[symbol] = {'quantity': quantity_to_buy, 'avg_price': current_price}
            
            self._log_trade(symbol, 'BUY', quantity_to_buy, current_price, reasoning, confidence)
            print(f"SUCCESS: Bought {quantity_to_buy:.4f} of {symbol} at ${current_price:.2f}")
            return True

    def sell_stock(self, symbol, quantity_to_sell, current_price, reasoning, confidence):
        """Sells a stock, updates portfolio, and logs the trade."""
        with self._lock:
            if symbol not in self.stocks or self.stocks[symbol]['quantity'] < quantity_to_sell:
                print(f"WARNING: Not enough shares to sell {quantity_to_sell} of {symbol}.")
                return False

            revenue = quantity_to_sell * current_price
            self.cash += revenue
            self.stocks[symbol]['quantity'] -= quantity_to_sell
            
            if self.stocks[symbol]['quantity'] < 1e-6: # Use a small threshold for floating point
                del self.stocks[symbol]
                
            self._log_trade(symbol, 'SELL', quantity_to_sell, current_price, reasoning, confidence)
            print(f"SUCCESS: Sold {quantity_to_sell:.4f} of {symbol} at ${current_price:.2f}")
            return True

    def _log_trade(self, symbol, action, quantity, price, reasoning, confidence):
        """Logs a trade to the SQLite database."""
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO trades (symbol, action, quantity, price, reasoning, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (symbol, action, quantity, price, reasoning, confidence))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"ERROR: Failed to log trade to database. Error: {e}")

# --- Finnhub API Client ---
class FinnhubClient:
    """A client to interact with the Finnhub API."""
    def __init__(self, api_key):
        self.api_key = api_key
        print("Finnhub Client initialized.")

    def _make_request(self, endpoint, params=None):
        if params is None: params = {}
        params['token'] = self.api_key
        try:
            response = requests.get(f"{FINNHUB_BASE_URL}/{endpoint}", params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Finnhub API request failed: {e}")
            return None

    def get_quote(self, symbol):
        data = self._make_request('quote', {'symbol': symbol.upper()})
        return data.get('c') if data else None

    def get_company_news(self, symbol):
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        params = {'symbol': symbol.upper(), 'from': from_date, 'to': to_date}
        return self._make_request('company-news', params)

# --- AI Decision Making ---
def get_ai_decision(symbol, current_price, news, portfolio):
    """Asks the Gemini AI for a trading decision."""
    if not ai_model:
        print("ERROR: AI model not available for decision making.")
        return None

    news_headlines = [f"- {item['headline']}" for item in news[:5]] if news else ["No recent news."]
    
    prompt = f"""
    You are an expert stock trading analyst bot. Your goal is to maximize portfolio value.
    Analyze the following information and decide whether to BUY, SELL, or HOLD the stock.

    **Current Portfolio Status:**
    - Cash: ${portfolio['cash']:.2f}
    - Total Value: ${portfolio['total_portfolio_value']:.2f}
    - Stocks Owned: {json.dumps(portfolio['owned_stocks'], indent=2)}

    **Stock to Analyze:** {symbol}
    - Current Price: ${current_price:.2f}

    **Recent News Headlines:**
    {chr(10).join(news_headlines)}

    **Decision Logic:**
    1.  **BUY:** If you see strong positive news or believe the stock is undervalued and there is sufficient cash.
    2.  **SELL:** If you see strong negative news, believe the stock is overvalued, or want to take profits. Only sell if the stock is currently owned.
    3.  **HOLD:** If the signals are mixed, news is neutral, or it's better to wait.

    **Your Response MUST be in the following JSON format ONLY:**
    {{
      "action": "BUY",
      "symbol": "{symbol}",
      "confidence": 0.85,
      "reasoning": "The positive product launch news and strong earnings report suggest a high probability of upward movement."
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
        print(f"ERROR: Failed to get or parse AI decision for {symbol}. Error: {e}")
        return None

# --- Main Bot Loop ---
def bot_trading_loop(portfolio_manager, finnhub_client):
    """The main autonomous loop for the trading bot."""
    print("Bot trading loop started.")
    while True:
        print("\n--- Starting new trading cycle ---")
        for symbol in STOCKS_TO_MONITOR:
            print(f"Analyzing {symbol}...")
            current_price = finnhub_client.get_quote(symbol)
            if not current_price:
                print(f"Could not get price for {symbol}, skipping.")
                continue

            news = finnhub_client.get_company_news(symbol)
            portfolio_status = portfolio_manager.get_portfolio_status()

            ai_decision = get_ai_decision(symbol, current_price, news, portfolio_status)

            if ai_decision and ai_decision.get('confidence', 0) > 0.7:
                action = ai_decision.get('action', '').upper()
                reasoning = ai_decision.get('reasoning', '')
                confidence = ai_decision.get('confidence', 0)
                
                if action == 'BUY':
                    quantity = TRADE_AMOUNT_USD / current_price
                    portfolio_manager.buy_stock(symbol, quantity, current_price, reasoning, confidence)
                elif action == 'SELL':
                    if symbol in portfolio_status['owned_stocks']:
                        quantity = TRADE_AMOUNT_USD / current_price
                        # Ensure we don't sell more than we own
                        quantity_to_sell = min(quantity, portfolio_status['owned_stocks'][symbol]['quantity'])
                        portfolio_manager.sell_stock(symbol, quantity_to_sell, current_price, reasoning, confidence)
                    else:
                        print(f"AI wants to SELL {symbol}, but we don't own any. Holding.")
            else:
                print(f"AI confidence too low or no decision for {symbol}. Holding.")
            
            time.sleep(15) # Small delay to avoid hitting API rate limits too quickly in a single cycle

        print(f"--- Trading cycle finished. Waiting for {LOOP_INTERVAL_SECONDS} seconds. ---")
        time.sleep(LOOP_INTERVAL_SECONDS)

# --- Global Instances ---
finnhub_client = FinnhubClient(FINNHUB_API_KEY)
portfolio_manager = PortfolioManager(INITIAL_CASH, finnhub_client)

# --- API Endpoints ---
@app.route("/")
def index():
    return "<h1>AI Stock Bot Backend is Running</h1>"

@app.route("/api/portfolio", methods=['GET'])
def get_portfolio():
    return jsonify(portfolio_manager.get_portfolio_status())

@app.route("/api/trades", methods=['GET'])
def get_trades():
    """Returns the last 50 trades from the database."""
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 50")
        trades = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return jsonify(trades)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Main Execution ---
if __name__ == "__main__":
    init_db()
    # Start the bot's trading loop in a separate thread
    if GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
        bot_thread = Thread(target=bot_trading_loop, args=(portfolio_manager, finnhub_client), daemon=True)
        bot_thread.start()
    else:
        print("WARNING: Gemini API key not set. The trading bot loop will not start.")
    
    # Start the Flask web server
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)

