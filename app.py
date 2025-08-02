# app.py
# Final Version: Immediate Initial Buy-in and AI Chat Fix

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
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
INITIAL_CASH = 5000.00
DB_FILE = "trades.db"
TRADE_AMOUNT_USD = 250 # Reduced trade amount for more diversification
LOOP_INTERVAL_SECONDS = 300
STOCKS_TO_SCAN_PER_CYCLE = 10
AI_LEARNING_TRADE_THRESHOLD = 5
INITIAL_BUY_COUNT = 10

# --- Bot State ---
bot_status_lock = Lock()
bot_is_running = True

# --- AI Configuration ---
ai_model = None
def configure_ai():
    global ai_model
    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            ai_model = genai.GenerativeModel('gemini-1.5-flash')
            print("Gemini AI model configured successfully.")
        except Exception as e:
            print(f"ERROR: Failed to configure Gemini AI: {e}")
            ai_model = None
    else:
        print("WARNING: Gemini API key not set. AI features will be disabled.")

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
    except Exception:
        return 0

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
        # **FIXED**: On first ever startup, trigger the initial buy.
        is_fresh_start = not os.path.exists(self.db_file)
        self.reset(perform_initial_buy=is_fresh_start)
        print("Portfolio Manager initialized.")

    def reset(self, perform_initial_buy=False):
        with self._lock:
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
            if perform_initial_buy:
                # Run in a thread to avoid blocking server startup
                Thread(target=self.buy_initial_stocks).start()
                return {"message": "Portfolio reset and initial stock purchase initiated."}
        return {"message": "Portfolio reset. No initial buy performed."}


    def buy_initial_stocks(self):
        print("Starting immediate initial stock purchase process...")
        time.sleep(5) # Wait for server to be ready
        sp500 = self.api_client.get_sp500_constituents()
        if not sp500:
            print("Failed to fetch S&P 500 list for initial stocks.")
            return

        stocks_to_buy = random.sample(sp500, min(INITIAL_BUY_COUNT, len(sp500)))
        purchased_count = 0
        for symbol in stocks_to_buy:
            price = self.api_client.get_quote(symbol)
            if price and self.cash >= TRADE_AMOUNT_USD:
                quantity = TRADE_AMOUNT_USD / price
                self.buy_stock(symbol, quantity, price, "Initial portfolio seeding.", 0.5)
                purchased_count += 1
            else:
                print(f"Skipping initial buy for {symbol}.")
            time.sleep(1.5) # Stagger API calls to avoid rate limiting
        print(f"Initial buy-in complete. Purchased {purchased_count} stocks.")
        return {"message": f"Initial buy-in complete. Purchased {purchased_count} stocks."}

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
        self.sp500_symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "GOOG", "BRK.B",
            "UNH", "JPM", "JNJ", "V", "XOM", "MA", "PG", "HD", "CVX", "LLY", "ABBV",
            "PFE", "BAC", "KO", "TMO", "PEP", "AVGO", "WMT", "COST", "MCD", "CSCO"
        ]
        print("Finnhub Client initialized.")
    def _make_request(self, endpoint, params=None):
        if not self.api_key: return None
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
        return self.sp500_symbols

# --- AI Decision Making ---
def get_ai_decision(symbol, price, news, portfolio, recent_trades, market_news, trade_count):
    if not ai_model: return None
    # This function was missing its logic in the user-provided file. It is now restored.
    news_headlines = [f"- {item['headline']}" for item in news[:5]] if news else ["No recent news."]
    market_headlines = [f"- {item['headline']}" for item in market_news[:5]] if market_news else ["No general market news."]
    
    startup_mode_prompt = ""
    if trade_count < INITIAL_BUY_COUNT + AI_LEARNING_TRADE_THRESHOLD:
        startup_mode_prompt = f"""
        **Current Operational Directive: Initial Learning Phase**
        You are in a learning phase. Your goal is to make {AI_LEARNING_TRADE_THRESHOLD} trades based on strong signals to build a robust learning history. Prioritize making well-reasoned trades over waiting.
        """
    prompt = f"""
    You are an expert stock trading analyst bot.
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
    Analyze all available data to make a strategic decision. Assess news for fundamental signals. Learn from your past performance.
    **Your Response MUST be in the following JSON format ONLY:**
    {{
      "action": "BUY", "symbol": "{symbol}", "confidence": 0.85,
      "reasoning": "The stock is showing a bullish trend, which is supported by recent positive news."
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
        print(f"ERROR: Failed to get or parse AI decision for {symbol}: {e}")
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
        if trade_count < INITIAL_BUY_COUNT + AI_LEARNING_TRADE_THRESHOLD:
            confidence_threshold = 0.65
        else:
            confidence_threshold = 0.75
        
        print(f"Current trade count: {trade_count}. Confidence threshold set to {confidence_threshold*100}%.")

        sp500 = finnhub_client.get_sp500_constituents()
        portfolio = portfolio_manager.get_portfolio_status()
        owned_stocks = list(portfolio['owned_stocks'].keys())
        market_news = finnhub_client.get_market_news()
        stocks_to_analyze = list(set(random.sample(sp500, STOCKS_TO_SCAN_PER_CYCLE) + owned_stocks))
        print(f"This cycle, analyzing: {stocks_to_analyze}")

        for symbol in stocks_to_analyze:
            with bot_status_lock:
                if not bot_is_running: break
            
            print(f"Analyzing {symbol}...")
            price = finnhub_client.get_quote(symbol)
            if not price: continue
            
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
    result = portfolio_manager.reset(perform_initial_buy=True)
    return jsonify(result)

@app.route("/api/bot/start", methods=['POST'])
def start_bot():
    global bot_is_running
    with bot_status_lock:
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
    return jsonify({"status": status})

@app.route("/api/ask", methods=['POST'])
def ask_ai():
    question = request.json.get('question')
    if not question:
        return jsonify({"answer": "No question was provided."}), 400
    
    if not ai_model:
        return jsonify({"answer": "AI Core is offline. The GEMINI_API_KEY is likely missing or invalid in your Render environment variables."}), 503

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
        error_message = f"The AI model failed to respond. This is often due to an invalid API key or a temporary problem with the Google AI service. Please double-check your GEMINI_API_KEY in Render. Error details: {str(e)}"
        print(f"ERROR: Failed to get AI chat response: {e}")
        return jsonify({"answer": error_message}), 500

# --- Main Execution ---
# Configure AI and start DB when the app module is first loaded.
init_db()
configure_ai()

if GEMINI_API_KEY and FINNHUB_API_KEY:
    if not app.debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        bot_thread = Thread(target=bot_trading_loop, args=(portfolio_manager, finnhub_client), daemon=True)
        bot_thread.start()
else:
    print("WARNING: API keys not set. Bot loop will not start.")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)
