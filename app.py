# app.py
# Final Version: Backend with Proactive Startup Mode

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
import random

# --- Configuration ---
# NOTE: Your API keys are kept in your file. For this example, we use placeholder keys.
GEMINI_API_KEY = "AIzaSyCFShQd4JEqv8AQUqtDyQ7iCDNWMHjId_c"
FINNHUB_API_KEY = "d25mi11r01qhge4das6gd25mi11r01qhge4das70"

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
INITIAL_CASH = 5000.00
DB_FILE = "trades.db"
TRADE_AMOUNT_USD = 500
LOOP_INTERVAL_SECONDS = 300 # 5 minutes
STOCKS_TO_SCAN_PER_CYCLE = 10

# --- Bot State ---
bot_status_lock = Lock()
bot_is_running = True # Bot starts in a running state by default

# --- AI Configuration ---
try:
    genai.configure(api_key=GEMINI_API_KEY)
    ai_model = genai.GenerativeModel('gemini-1.5-flash')
    print("Gemini AI model configured successfully.")
except Exception as e:
    print(f"ERROR: Failed to configure Gemini AI: {e}")
    ai_model = None

# --- Database Functions ---
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

            # If this is the very first time the app is running, buy 1 META stock.
            if is_fresh_start:
                self.buy_initial_stock_meta()

    def buy_initial_stock_meta(self):
        """Buys a single META stock to bootstrap the portfolio."""
        symbol = "META"
        print(f"Initiating a forced initial buy of {symbol}...")
        price = self.api_client.get_quote(symbol)
        if price:
            quantity = TRADE_AMOUNT_USD / price
            self.buy_stock(
                symbol=symbol,
                quantity=quantity,
                price=price,
                reasoning="Initial portfolio seeding. This is a forced buy to start the learning process.",
                confidence=0.99
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
    def get_company_news(self, symbol):
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        return self._make_request('company-news', {'symbol': symbol.upper(), 'from': from_date, 'to': to_date})
    def get_market_news(self):
        return self._make_request('news', {'category': 'general'})
    def get_sp500_constituents(self):
        # We are no longer making an API call for this list due to the 403 error on the free plan.
        # This function now just returns the hardcoded list.
        return self.sp500_symbols

# --- AI Decision Making ---
def get_ai_decision(symbol, price, news, portfolio, recent_trades, market_news, trade_count):
    if not ai_model: return None
    news_headlines = [f"- {item['headline']}" for item in news[:5]] if news else ["No recent news for this stock."]
    market_headlines = [f"- {item['headline']}" for item in market_news[:5]] if market_news else ["No general market news."]
    
    startup_mode_prompt = ""
    # The initial trades are now handled by the PortfolioManager.reset method.
    # The proactive mode is adjusted for a smoother learning curve after the first two trades.
    if trade_count < 5:
        startup_mode_prompt = """
        **Current Operational Directive: Startup Protocol**
        Your primary mission is to learn from your initial trades and continue to find promising BUY opportunities. Inaction is not an acceptable outcome if a strong signal is present. From the stocks you are analyzing, you **must** identify the single most promising BUY opportunity. Prioritize making a well-reasoned trade over waiting, but with a higher confidence threshold than the very first trades.
        """

    prompt = f"""
    You are an expert stock trading analyst bot. Your goal is to learn from your actions and maximize portfolio value.
    {startup_mode_prompt}
    **Current Portfolio Status:**
    {json.dumps(portfolio, indent=2)}
    **Your 5 Most Recent Trades (Your Memory):**
    {json.dumps(recent_trades, indent=2)}
    **General Market News (Overall Sentiment):**
    {chr(10).join(market_headlines)}
    **Stock to Analyze:** {symbol}
    - Current Price: ${price:.2f}
    - Recent News Headlines for {symbol}:
    {chr(10).join(news_headlines)}
    **Decision Logic & Learning:**
    1. Assess the general market sentiment from the market news. Is it bullish, bearish, or neutral?
    2. Review your recent trades. Learn from your profitable and unprofitable decisions.
    3. Analyze the specific news for {symbol}. Is it a startup with high potential, a stable tech giant, or something else?
    4. Based on all available data (market sentiment, your memory, and specific stock info), make a strategic decision.
    5. **BUY:** If signals are strong, the market outlook is favorable, and you have cash.
    6. **SELL:** If signals are negative, the market is turning, or to secure profits. Only sell if you own the stock.
    7. **HOLD:** If signals are mixed or waiting is the best strategy.
    **Your Response MUST be in the following JSON format ONLY:**
    {{
      "action": "BUY", "symbol": "{symbol}", "confidence": 0.85,
      "reasoning": "Despite mixed market news, the specific product announcement for this company is a significant positive catalyst. My recent profitable trade in a similar sector supports this aggressive position."
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
        print(f"ERROR: Failed to get AI decision for {symbol}: {e}")
        return None

# --- Main Bot Loop ---
def bot_trading_loop(portfolio_manager, finnhub_client):
    print("Bot trading loop started.")
    while True:
        with bot_status_lock:
            is_running = bot_is_running
        
        if not is_running:
            print("Bot is paused. Skipping trading cycle.")
            time.sleep(30)
            continue

        print("\n--- Starting new trading cycle ---")
        
        trade_count = get_trade_count()
        # Set a very low threshold for the first trade, then a slightly higher one for the next few.
        if trade_count < 2:
             # The first two trades are now handled at startup, so we don't need a low threshold here anymore.
            confidence_threshold = 0.65
        elif trade_count < 5:
            confidence_threshold = 0.65
        else:
            confidence_threshold = 0.70
        
        print(f"Current trade count: {trade_count}. Confidence threshold set to {confidence_threshold*100}%.")

        sp500 = finnhub_client.get_sp500_constituents()
        if not sp500:
            print("Could not fetch S&P 500 list, waiting for next cycle.")
            time.sleep(LOOP_INTERVAL_SECONDS)
            continue
        
        discovered_stocks = random.sample(sp500, STOCKS_TO_SCAN_PER_CYCLE)
        portfolio = portfolio_manager.get_portfolio_status()
        owned_stocks = list(portfolio['owned_stocks'].keys())
        market_news = finnhub_client.get_market_news()
        stocks_to_analyze = list(set(discovered_stocks + owned_stocks))
        print(f"This cycle, analyzing: {stocks_to_analyze}")

        for symbol in stocks_to_analyze:
            with bot_status_lock:
                if not bot_is_running:
                    print("Bot paused mid-cycle. Aborting current cycle.")
                    break
            
            print(f"Analyzing {symbol}...")
            price = finnhub_client.get_quote(symbol)
            if not price: continue
            
            news = finnhub_client.get_company_news(symbol)
            trades = get_recent_trades(5)
            current_portfolio_status = portfolio_manager.get_portfolio_status()
            ai_decision = get_ai_decision(symbol, price, news, current_portfolio_status, trades, market_news, trade_count)

            if ai_decision:
                action = ai_decision.get('action', '').upper()
                reasoning = ai_decision.get('reasoning', '')
                confidence = ai_decision.get('confidence', 0)

                # The startup trade logic has been moved to the reset method.
                # Now the bot will proceed with normal confidence thresholds.
                if confidence > confidence_threshold:
                    if action == 'BUY':
                        quantity = TRADE_AMOUNT_USD / price
                        portfolio_manager.buy_stock(symbol, quantity, price, reasoning, confidence)
                    elif action == 'SELL':
                        if symbol in current_portfolio_status['owned_stocks']:
                            quantity_to_sell = min(TRADE_AMOUNT_USD / price, current_portfolio_status['owned_stocks'][symbol]['quantity'])
                            portfolio_manager.sell_stock(symbol, quantity_to_sell, price, reasoning, confidence)
            
            time.sleep(20)

        print(f"--- Cycle finished. Waiting {LOOP_INTERVAL_SECONDS}s. ---")
        time.sleep(LOOP_INTERVAL_SECONDS)

# --- Global Instances ---
finnhub_client = FinnhubClient(FINNHUB_API_KEY)
portfolio_manager = PortfolioManager(INITIAL_CASH, finnhub_client, DB_FILE)

# --- API Endpoints ---
@app.route("/")
def index(): return "<h1>AI Stock Bot Backend is Running</h1>"
@app.route("/api/portfolio", methods=['GET'])
def get_portfolio(): return jsonify(portfolio_manager.get_portfolio_status())
@app.route("/api/trades", methods=['GET'])
def get_trades(): return jsonify(get_recent_trades())
@app.route("/api/portfolio/reset", methods=['POST'])
def reset_portfolio():
    portfolio_manager.reset()
    return jsonify({"message": "Portfolio reset successfully."})

@app.route("/api/bot/start", methods=['POST'])
def start_bot():
    global bot_is_running
    with bot_status_lock:
        bot_is_running = True
    print("Received command: START BOT")
    return jsonify({"status": "running"})

@app.route("/api/bot/pause", methods=['POST'])
def pause_bot():
    global bot_is_running
    with bot_status_lock:
        bot_is_running = False
    print("Received command: PAUSE BOT")
    return jsonify({"status": "paused"})

@app.route("/api/bot/status", methods=['GET'])
def get_bot_status():
    with bot_status_lock:
        status = "running" if bot_is_running else "paused"
    return jsonify({"status": status})

@app.route("/api/ask", methods=['POST'])
def ask_ai():
    question = request.json.get('question')
    if not question or not ai_model:
        return jsonify({"answer": "AI is unavailable or no question was asked."}), 400
    
    portfolio = portfolio_manager.get_portfolio_status()
    trades = get_recent_trades(5)
    market_news = finnhub_client.get_market_news()
    market_headlines = [f"- {item['headline']}" for item in market_news[:5]] if market_news else ["No general market news."]

    prompt = f"""
    You are the AI core of a stock trading bot. An administrator is asking you a question.
    Based on your current status and recent history, provide a helpful and concise answer.
    **Current Portfolio Status:**
    {json.dumps(portfolio, indent=2)}
    **Your 5 Most Recent Trades (Your Memory):**
    {json.dumps(trades, indent=2)}
    **General Market News (Overall Sentiment):**
    {chr(10).join(market_headlines)}
    **Administrator's Question:** "{question}"
    **Your Answer:**
    """
    try:
        response = ai_model.generate_content(prompt)
        return jsonify({"answer": response.text})
    except Exception as e:
        print(f"ERROR: Failed to get AI chat response: {e}")
        return jsonify({"answer": "I encountered an error."}), 500

# --- Main Execution ---
if __name__ == "__main__":
    init_db()
    if GEMINI_API_KEY and "YOUR_GEMINI_API_KEY" not in GEMINI_API_KEY:
        bot_thread = Thread(target=bot_trading_loop, args=(portfolio_manager, finnhub_client), daemon=True)
        bot_thread.start()
    else:
        print("WARNING: Gemini API key not set. Bot loop will not start.")
    
    # Get the port from the environment variable 'PORT', which is required by hosting services like Render.
    # If 'PORT' is not set (e.g., when running locally), it defaults to 8000.
    port = int(os.environ.get('PORT', 8000))
    
    # The host '0.0.0.0' makes the app accessible from outside the container,
    # which is also necessary for deployment on services like Render.
    app.run(host='0.0.0.0', port=port, threaded=True, debug=False)
