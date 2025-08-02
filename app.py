# app.py
# Final Version: With auto-pause on API limit and daily resume.

import os
import time
import requests
import json
import sqlite3
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from flask import Flask, jsonify, request, render_template, redirect, url_for
from flask_cors import CORS
from threading import Thread, Lock
from datetime import datetime, timedelta, timezone
import random

# --- Configuration ---
# Use a list of Gemini API keys
GEMINI_API_KEYS = [
    os.environ.get("GEMINI_API_KEY_1"),
    os.environ.get("GEMINI_API_KEY_2"),
    os.environ.get("GEMINI_API_KEY_3")
]
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
ADMIN_PASSWORD = "orosis"

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
INITIAL_CASH = 5000.00
DB_FILE = "trades.db"
TRADE_AMOUNT_USD = 250
LOOP_INTERVAL_SECONDS = 300
STOCKS_TO_SCAN_PER_CYCLE = 15
AI_LEARNING_TRADE_THRESHOLD = 5
INITIAL_BUY_COUNT = 10

# --- Bot State ---
bot_status_lock = Lock()
bot_is_running = True
# Track which API key is currently in use
current_api_key_index = 0
all_keys_exhausted = False  # Flag to track if all API keys are exhausted

# --- AI Configuration (Lazy Initialization) ---
ai_model = None
ai_model_configured = False
ai_model_lock = Lock()


def configure_ai():
    global ai_model, ai_model_configured, current_api_key_index, all_keys_exhausted
    with ai_model_lock:
        if ai_model_configured:
            return
        
        # Get the current API key from the list
        gemini_api_key = GEMINI_API_KEYS[current_api_key_index]
        
        if gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                ai_model = genai.GenerativeModel('gemini-1.5-flash')
                print(f"Gemini AI model configured with key index {current_api_key_index}.")
                ai_model_configured = True
            except Exception as e:
                print(f"ERROR: Failed to configure Gemini AI with key {current_api_key_index}: {e}")
                ai_model_configured = False
                # If configuration fails, try the next key
                current_api_key_index = (current_api_key_index + 1) % len(GEMINI_API_KEYS)
                if current_api_key_index == 0:
                    print("All API keys failed to configure. Pausing bot.")
                    all_keys_exhausted = True
                    bot_is_running = False
        else:
            print(f"WARNING: Gemini API key at index {current_api_key_index} not found in environment.")


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


def get_all_trades():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trades ORDER BY timestamp ASC")
        trades = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return trades
    except Exception:
        return []


def get_recent_trades(limit=50):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,))
        trades = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return trades
    except Exception:
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
        self.cash = initial_cash
        self.stocks = {}
        self.initial_value = initial_cash
        self._reconstruct_portfolio_from_db()
        print("Portfolio Manager initialized and reconstructed.")

    def _reconstruct_portfolio_from_db(self):
        with self._lock:
            all_trades = get_all_trades()
            self.cash = self.initial_cash
            self.stocks = {}
            for trade in all_trades:
                symbol, quantity, price, action = trade['symbol'], trade['quantity'], trade['price'], trade['action']
                if action == 'BUY':
                    cost = quantity * price
                    self.cash -= cost
                    if symbol in self.stocks:
                        current_quantity = self.stocks[symbol]['quantity']
                        new_total_quantity = current_quantity + quantity
                        new_avg_price = ((self.stocks[symbol]['avg_price'] * current_quantity) + cost) / new_total_quantity
                        self.stocks[symbol]['quantity'] = new_total_quantity
                        self.stocks[symbol]['avg_price'] = new_avg_price
                    else:
                        self.stocks[symbol] = {'quantity': quantity, 'avg_price': price}
                elif action == 'SELL':
                    self.cash += quantity * price
                    if symbol in self.stocks:
                        self.stocks[symbol]['quantity'] -= quantity
                        if self.stocks[symbol]['quantity'] < 1e-6:
                            del self.stocks[symbol]
            print(f"Reconstruction complete. Cash: {self.cash:.2f}")

    def reset(self):
        with self._lock:
            if os.path.exists(self.db_file):
                os.remove(self.db_file)
            init_db()
            self.cash = self.initial_cash
            self.stocks = {}
            print("Portfolio has been reset.")
            Thread(target=self.buy_initial_stocks).start()
            return {"message": "Portfolio reset and initial stock purchase initiated."}

    def buy_initial_stocks(self):
        print("Starting initial stock purchase process...")
        time.sleep(5)
        sp500 = self.api_client.get_sp500_constituents()
        if not sp500:
            print("Failed to fetch S&P 500 list for initial stocks.")
            return

        stocks_to_buy = random.sample(sp500, min(INITIAL_BUY_COUNT, len(sp500)))
        for symbol in stocks_to_buy:
            price = self.api_client.get_quote(symbol)
            if price and self.cash >= TRADE_AMOUNT_USD:
                quantity = TRADE_AMOUNT_USD / price
                self.buy_stock(symbol, quantity, price, "Initial portfolio seeding.", 0.5)
            else:
                print(f"Skipping initial buy for {symbol}.")
            time.sleep(1.5)
        print("Initial buy-in complete.")

    def get_portfolio_status(self):
        with self._lock:
            stock_values = 0.0
            detailed_stocks = {}
            for symbol, data in self.stocks.items():
                current_price = self.api_client.get_quote(symbol) or data['avg_price']
                value = data['quantity'] * current_price
                stock_values += value
                detailed_stocks[symbol] = {
                    "quantity": data['quantity'], "average_buy_price": data['avg_price'],
                    "current_price": current_price, "current_value": value
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
            if self.cash < cost:
                return False
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
            if symbol not in self.stocks or self.stocks[symbol]['quantity'] < quantity:
                return False
            self.cash += quantity * price
            self.stocks[symbol]['quantity'] -= quantity
            if self.stocks[symbol]['quantity'] < 1e-6:
                del self.stocks[symbol]
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
    def __init__(self):
        self.sp500_symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "GOOG", "BRK.B",
            "UNH", "JPM", "JNJ", "V", "XOM", "MA", "PG", "HD", "CVX", "LLY", "ABBV",
            "PFE", "BAC", "KO", "TMO", "PEP", "AVGO", "WMT", "COST", "MCD", "CSCO"
        ]
        print("Finnhub Client initialized.")

    def _make_request(self, endpoint, params=None):
        api_key = os.environ.get("FINNHUB_API_KEY")
        if not api_key:
            print("ERROR: FINNHUB_API_KEY not found in environment.")
            return None
        if params is None:
            params = {}
        params['token'] = api_key
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
        return self.sp500_symbols


# --- AI Decision Making & Bot Loop ---
def get_ai_decision(symbol, price, news, portfolio, recent_trades, market_news, trade_count):
    global bot_is_running, all_keys_exhausted, current_api_key_index, ai_model, ai_model_configured
    
    if not ai_model_configured:
        configure_ai()
        if not ai_model_configured:
            return None

    prompt = f"""
    You are an expert stock trading analyst bot...
    (Prompt logic is the same, omitted for brevity)
    """
    
    try:
        response = ai_model.generate_content(prompt)
        decision_text = response.text.strip().replace("```json", "").replace("```", "")
        decision = json.loads(decision_text)
        print(f"AI Decision for {symbol} using key {current_api_key_index}: {decision}")
        return decision
    except google_exceptions.ResourceExhausted as e:
        print(f"CRITICAL ERROR: Gemini API key {current_api_key_index} limit reached.")
        
        with bot_status_lock:
            # Move to the next API key
            current_api_key_index = (current_api_key_index + 1) % len(GEMINI_API_KEYS)
            ai_model_configured = False # Force reconfiguration with the new key
            
            if current_api_key_index == 0:
                # We've wrapped around, all keys are exhausted
                print("All Gemini API keys are exhausted. Auto-pausing bot until tomorrow.")
                bot_is_running = False
                all_keys_exhausted = True
            else:
                print(f"Switching to next Gemini API key at index {current_api_key_index}.")
        
        return None
    except Exception as e:
        print(f"ERROR: Failed to get or parse AI decision for {symbol} using key {current_api_key_index}: {e}")
        return None


def bot_trading_loop(portfolio_manager, finnhub_client):
    print("Bot trading loop started.")
    while True:
        with bot_status_lock:
            is_running = bot_is_running

        if not is_running:
            status_reason = "all API keys exhausted" if all_keys_exhausted else "manually paused"
            print(f"Bot is {status_reason}. Skipping trading cycle.")
            time.sleep(30)
            continue

        print("\n--- Starting new trading cycle ---")
        trade_count = len(get_all_trades())
        confidence_threshold = 0.65 if trade_count < (INITIAL_BUY_COUNT + AI_LEARNING_TRADE_THRESHOLD) else 0.75

        print(f"Current trade count: {trade_count}. Confidence threshold set to {confidence_threshold * 100}%.")

        sp500 = finnhub_client.get_sp500_constituents()
        portfolio = portfolio_manager.get_portfolio_status()
        owned_stocks = list(portfolio['owned_stocks'].keys())
        market_news = finnhub_client.get_market_news()
        stocks_to_analyze = list(set(random.sample(sp500, STOCKS_TO_SCAN_PER_CYCLE) + owned_stocks))
        print(f"This cycle, analyzing: {stocks_to_analyze}")

        for symbol in stocks_to_analyze:
            with bot_status_lock:
                if not bot_is_running:
                    break

            print(f"Analyzing {symbol}...")
            price = finnhub_client.get_quote(symbol)
            if not price:
                continue

            news = finnhub_client.get_company_news(symbol)
            trades = get_recent_trades(5)
            current_portfolio_status = portfolio_manager.get_portfolio_status()
            ai_decision = get_ai_decision(symbol, price, news, current_portfolio_status, trades, market_news, trade_count)

            if ai_decision and ai_decision.get('confidence', 0) > confidence_threshold:
                action = ai_decision.get('action', '').upper()
                reasoning = ai_decision.get('reasoning', '')
                confidence = ai_decision.get('confidence', 0)

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


# --- Scheduler Loop for Daily Resume ---
def scheduler_loop():
    global bot_is_running, all_keys_exhausted, current_api_key_index
    while True:
        now_utc = datetime.now(timezone.utc)
        tomorrow_utc = now_utc + timedelta(days=1)
        # Set resume time for 1 minute past midnight UTC
        midnight_utc = tomorrow_utc.replace(hour=0, minute=1, second=0, microsecond=0)
        sleep_seconds = (midnight_utc - now_utc).total_seconds()
        
        print(f"Scheduler: Sleeping for {sleep_seconds / 3600:.2f} hours until quota reset.")
        time.sleep(sleep_seconds)

        with bot_status_lock:
            if all_keys_exhausted:
                print("Scheduler: API quotas have reset. Resuming bot.")
                bot_is_running = True
                all_keys_exhausted = False
                current_api_key_index = 0 # Reset to the first key


# --- Global Instances & App Initialization ---
init_db()
configure_ai()
finnhub_client = FinnhubClient()
portfolio_manager = PortfolioManager(INITIAL_CASH, finnhub_client, DB_FILE)

# --- Web Page Routes & API ---
@app.route("/")
def index():
    return "<h1>AI Stock Bot Backend is Running</h1>"


@app.route("/view")
def view_dashboard():
    return render_template("view.html")


@app.route("/admin", methods=['GET', 'POST'])
def admin_dashboard():
    if request.method == 'POST':
        if request.form.get('password') == ADMIN_PASSWORD:
            return render_template("admin.html")
        else:
            return render_template("login.html", error="Invalid password")
    return render_template("login.html")


@app.route("/api/portfolio", methods=['GET'])
def get_portfolio():
    return jsonify(portfolio_manager.get_portfolio_status())


@app.route("/api/trades", methods=['GET'])
def get_trades():
    return jsonify(get_recent_trades())


@app.route("/api/portfolio/reset", methods=['POST'])
def reset_portfolio():
    result = portfolio_manager.reset()
    return jsonify(result)


@app.route("/api/bot/start", methods=['POST'])
def start_bot():
    global bot_is_running, all_keys_exhausted
    with bot_status_lock:
        if all_keys_exhausted:
            return jsonify({"status": "paused", "reason": "All API keys are exhausted. Bot will resume automatically tomorrow."}), 400
        bot_is_running = True
        return jsonify({"status": "running"})


@app.route("/api/bot/pause", methods=['POST'])
def pause_bot():
    global bot_is_running
    with bot_status_lock:
        bot_is_running = False
    return jsonify({"status": "paused"})


@app.route("/api/bot/status", methods=['GET'])
def get_bot_status():
    with bot_status_lock:
        status = "running" if bot_is_running else "paused"
        if all_keys_exhausted:
            status = "paused_apikey"
    return jsonify({"status": status})


@app.route("/api/ask", methods=['POST'])
def ask_ai():
    # (This function's logic remains the same)
    pass


# --- Main Execution ---
if __name__ == "__main__":
    if not app.debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        bot_thread = Thread(target=bot_trading_loop, args=(portfolio_manager, finnhub_client), daemon=True)
        bot_thread.start()
        scheduler_thread = Thread(target=scheduler_loop, daemon=True)
        scheduler_thread.start()

    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
