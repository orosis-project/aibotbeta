# app.py
# Final Version: Backend with Proactive Startup Mode, Enhanced Data, and Advanced Risk Management

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
GEMINI_API_KEY = "AIzaSyCFShQd4JEqv8AQUqtDyQ7iCDNWMHjId_c"
FINNHUB_API_KEY = "d25mi11r01qhge4das6gd25mi11r01qhge4das70"

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
INITIAL_CASH = 5000.00
DB_FILE = "trades.db"
TRADE_AMOUNT_USD = 500  # Base trade amount, now used for calculation
LOOP_INTERVAL_SECONDS = 300  # 5 minutes
STOCKS_TO_SCAN_PER_CYCLE = 10
# Advanced Risk Management: Adjust these values to control risk appetite
CONFIDENCE_MULTIPLIER = 2.0  # Scales the trade size by confidence.
MIN_CONFIDENCE_FOR_RISKY_TRADE = 0.8  # AI must be this confident for a large trade.
# New: Spam buy configuration
SPAM_BUY_DURATION_SECONDS = 60 # Duration of the spam buying phase

# --- Bot State ---
bot_status_lock = Lock()
bot_is_running = True # Bot starts in a running state by default
bot_start_time = None # Tracks when the bot was started

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
            
            global bot_start_time
            bot_start_time = time.time() # Reset the bot start time on a reset

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
    
    def get_stock_candles(self, symbol, resolution="D", days=20):
        # We'll use "D" for daily candles
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
    if len(past_performance['recent_trades']) < 5:
        startup_mode_prompt = """
        **Current Operational Directive: Startup Protocol**
        Your primary mission is to learn from your initial trades and continue to find promising BUY opportunities. Inaction is not an acceptable outcome if a strong signal is present. From the stocks you are analyzing, you **must** identify the single most promising BUY opportunity. Prioritize making a well-reasoned trade over waiting, but with a higher confidence threshold than the very first trades.
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
        print(f"ERROR: Failed to get AI decision for {symbol}: {e}")
        return None

# --- Main Bot Loop ---
def bot_trading_loop(portfolio_manager, finnhub_client):
    global bot_start_time
    if bot_start_time is None:
        bot_start_time = time.time()
    
    print("Bot trading loop started.")
    while True:
        with bot_status_lock:
            is_running = bot_is_running
        
        if not is_running:
            print("Bot is paused. Skipping trading cycle.")
            time.sleep(30)
            continue
        
        # New: Spam buying logic for the first minute of operation
        if time.time() - bot_start_time < SPAM_BUY_DURATION_SECONDS:
            print("\n--- Starting SPAM BUY cycle ---")
            sp500 = finnhub_client.get_sp500_constituents()
            if sp500:
                symbol = random.choice(sp500)
                price = finnhub_client.get_quote(symbol)
                if price:
                    quantity = TRADE_AMOUNT_USD / price
                    portfolio_manager.buy_stock(
                        symbol=symbol,
                        quantity=quantity,
                        price=price,
                        reasoning="Forced buy during initial spam-buying phase to seed portfolio.",
                        confidence=0.1
                    )
            time.sleep(5) # Shorter delay for faster buying
            continue # Skip the rest of the loop and start again

        print("\n--- Starting new trading cycle ---")
        
        trade_count = get_trade_count()
        # The confidence threshold now adjusts based on total trades, getting stricter over time.
        confidence_threshold = 0.55 if trade_count < 2 else (0.65 if trade_count < 5 else 0.70)
        
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
        
        # New: Prepare a more detailed past performance summary for the AI
        all_trades = get_recent_trades(200) # Get a larger history
        total_profit_loss = sum(t['quantity'] * (portfolio['owned_stocks'].get(t['symbol'], {}).get('current_price', t['price']) - t['price']) for t in all_trades if t['action'] == 'BUY')
        
        winning_trades = sum(1 for t in all_trades if t['action'] == 'SELL' and (t['price'] - portfolio['owned_stocks'].get(t['symbol'], {}).get('average_buy_price', t['price'])) > 0)
        losing_trades = sum(1 for t in all_trades if t['action'] == 'SELL' and (t['price'] - portfolio['owned_stocks'].get(t['symbol'], {}).get('average_buy_price', t['price'])) < 0)
        total_profit_from_winning_trades = sum(t['quantity'] * (t['price'] - portfolio['owned_stocks'].get(t['symbol'], {}).get('average_buy_price', t['price'])) for t in all_trades if t['action'] == 'SELL' and (t['price'] - portfolio['owned_stocks'].get(t['symbol'], {}).get('average_buy_price', t['price'])) > 0)
        
        past_performance = {
            "recent_trades": get_recent_trades(5),
            "total_trades": len(all_trades),
            "total_profit_loss": total_profit_loss,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "total_profit_from_winning_trades": total_profit_from_winning_trades
        }

        for symbol in stocks_to_analyze:
            with bot_status_lock:
                if not bot_is_running:
                    print("Bot paused mid-cycle. Aborting current cycle.")
                    break
            
            print(f"Analyzing {symbol}...")
            price = finnhub_client.get_quote(symbol)
            if not price: continue

            # New: Get historical data for SMA
            candles = finnhub_client.get_stock_candles(symbol)
            sma = calculate_sma(candles)
            if not sma:
                print(f"Could not calculate SMA for {symbol}, skipping analysis.")
                continue
            
            news = finnhub_client.get_company_news(symbol)
            current_portfolio_status = portfolio_manager.get_portfolio_status()
            ai_decision = get_ai_decision(symbol, price, news, current_portfolio_status, past_performance, market_news, sma)

            if ai_decision:
                action = ai_decision.get('action', '').upper()
                reasoning = ai_decision.get('reasoning', '')
                confidence = ai_decision.get('confidence', 0)

                if confidence > confidence_threshold:
                    if action == 'BUY':
                        # New: Dynamic trade sizing based on confidence
                        trade_size_multiplier = 1.0
                        if confidence >= MIN_CONFIDENCE_FOR_RISKY_TRADE:
                            trade_size_multiplier = CONFIDENCE_MULTIPLIER
                            print(f"AI confidence is high ({confidence}), increasing trade size by {trade_size_multiplier}x.")
                            
                        trade_amount = TRADE_AMOUNT_USD * trade_size_multiplier
                        quantity = trade_amount / price
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

@app.route("/api/portfolio/summary", methods=['GET'])
def get_portfolio_summary():
    """A new endpoint to get a detailed performance summary for the admin panel."""
    portfolio = portfolio_manager.get_portfolio_status()
    all_trades = get_recent_trades(200) # Get a larger history
    
    total_profit_loss = sum(t['quantity'] * (portfolio['owned_stocks'].get(t['symbol'], {}).get('current_price', t['price']) - t['price']) for t in all_trades if t['action'] == 'BUY')
    
    # A simple but effective way to calculate winning/losing trades.
    winning_trades = sum(1 for t in all_trades if t['action'] == 'SELL' and (t['price'] - portfolio['owned_stocks'].get(t['symbol'], {}).get('average_buy_price', t['price'])) > 0)
    losing_trades = sum(1 for t in all_trades if t['action'] == 'SELL' and (t['price'] - portfolio['owned_stocks'].get(t['symbol'], {}).get('average_buy_price', t['price'])) < 0)
    
    summary = {
        "portfolio": portfolio,
        "recent_trades": get_recent_trades(20),
        "total_trades": len(all_trades),
        "total_profit_loss": total_profit_loss,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
    }
    
    return jsonify(summary)

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
    global bot_is_running, bot_start_time
    with bot_status_lock:
        bot_is_running = True
        bot_start_time = time.time() # Start the timer for the spam buying
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

# --- New Route for Admin Panel ---
@app.route("/pannel")
def admin_pannel():
    return render_template("admin_pannel.html")

# --- Main Execution ---
if __name__ == "__main__":
    init_db()
    if GEMINI_API_KEY and "YOUR_GEMINI_API_KEY" not in GEMINI_API_KEY:
        bot_thread = Thread(target=bot_trading_loop, args=(portfolio_manager, finnhub_client), daemon=True)
        bot_thread.start()
    else:
        print("WARNING: Gemini API key not set. Bot loop will not start.")
    
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, threaded=True, debug=False)
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Trading Bot Admin Panel</title>
    <script src="[https://cdn.tailwindcss.com](https://cdn.tailwindcss.com)"></script>
    <style>
        body { font-family: "Inter", sans-serif; background-color: #1a202c; color: #e2e8f0; }
        .card { background-color: #2d3748; }
        .green { color: #38a169; }
        .red { color: #e53e3e; }
        .orange { color: #dd6b20; }
        .status-running { color: #38a169; }
        .status-paused { color: #e53e3e; }
    </style>
</head>
<body class="p-8">
    <div class="max-w-7xl mx-auto">
        <h1 class="text-4xl font-bold mb-8 text-center text-white">AI Trading Bot Admin Panel</h1>
        
        <div class="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
            <!-- Bot Status Card -->
            <div class="card p-6 rounded-lg shadow-lg">
                <h2 class="text-xl font-semibold mb-4 text-white">Bot Status</h2>
                <div class="flex items-center space-x-4">
                    <div id="status-indicator" class="w-4 h-4 rounded-full bg-gray-500 animate-pulse"></div>
                    <span id="bot-status" class="text-lg font-bold">Loading...</span>
                </div>
                <div class="flex mt-4 space-x-4">
                    <button id="start-btn" class="bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out">Start</button>
                    <button id="pause-btn" class="bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out">Pause</button>
                    <button id="reset-btn" class="bg-orange-500 hover:bg-orange-600 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out">Reset Portfolio</button>
                </div>
            </div>

            <!-- Portfolio Summary Card -->
            <div class="card p-6 rounded-lg shadow-lg">
                <h2 class="text-xl font-semibold mb-4 text-white">Portfolio Summary</h2>
                <div class="space-y-2">
                    <p>Cash: <span id="cash" class="font-bold">$0.00</span></p>
                    <p>Total Value: <span id="total-value" class="font-bold">$0.00</span></p>
                    <p>Profit/Loss: <span id="profit-loss" class="font-bold">$0.00</span></p>
                    <p>Owned Stocks: <span id="owned-stocks-count" class="font-bold">0</span></p>
                </div>
            </div>

            <!-- Trade History Summary Card -->
            <div class="card p-6 rounded-lg shadow-lg">
                <h2 class="text-xl font-semibold mb-4 text-white">Trade History Summary</h2>
                <div class="space-y-2">
                    <p>Total Trades: <span id="total-trades" class="font-bold">0</span></p>
                    <p>Winning Trades: <span id="winning-trades" class="font-bold text-green-500">0</span></p>
                    <p>Losing Trades: <span id="losing-trades" class="font-bold text-red-500">0</span></p>
                </div>
            </div>
        </div>

        <!-- Owned Stocks List -->
        <div class="card p-6 rounded-lg shadow-lg mb-8">
            <h2 class="text-xl font-semibold mb-4 text-white">Owned Stocks</h2>
            <div id="owned-stocks-container" class="space-y-4">
                <!-- Stock cards will be injected here by JavaScript -->
                <p id="no-stocks-message" class="text-gray-400">No stocks currently owned.</p>
            </div>
        </div>

        <!-- Recent Trades Table -->
        <div class="card p-6 rounded-lg shadow-lg">
            <h2 class="text-xl font-semibold mb-4 text-white">Recent Trades</h2>
            <div class="overflow-x-auto">
                <table class="min-w-full divide-y divide-gray-700">
                    <thead class="bg-gray-700">
                        <tr>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Timestamp</th>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Symbol</th>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Action</th>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Quantity</th>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Price</th>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Confidence</th>
                            <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Reasoning</th>
                        </tr>
                    </thead>
                    <tbody id="trades-table-body" class="divide-y divide-gray-700">
                        <!-- Trade rows will be injected here by JavaScript -->
                        <tr>
                            <td colspan="7" class="px-6 py-4 whitespace-nowrap text-center text-gray-400">Loading trades...</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        const API_BASE = window.location.origin + '/api';

        // Helper function to fetch data and handle errors
        async function fetchData(endpoint) {
            try {
                const response = await fetch(`${API_BASE}${endpoint}`);
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return await response.json();
            } catch (error) {
                console.error(`Could not fetch data from ${endpoint}:`, error);
                return null;
            }
        }

        // Helper function to send commands (start, pause, reset)
        async function sendCommand(endpoint) {
            try {
                const response = await fetch(`${API_BASE}${endpoint}`, { method: 'POST' });
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                const data = await response.json();
                console.log(data.message || data.status);
                // Refresh all data after a command
                await refreshData();
            } catch (error) {
                console.error(`Failed to send command to ${endpoint}:`, error);
            }
        }

        // Updates the bot status display
        async function updateBotStatus() {
            const data = await fetchData('/bot/status');
            const statusElement = document.getElementById('bot-status');
            const indicatorElement = document.getElementById('status-indicator');
            const startBtn = document.getElementById('start-btn');
            const pauseBtn = document.getElementById('pause-btn');

            if (data && data.status) {
                statusElement.textContent = data.status.toUpperCase();
                indicatorElement.classList.remove('bg-gray-500', 'animate-pulse', 'status-running', 'status-paused');
                if (data.status === 'running') {
                    indicatorElement.classList.add('status-running', 'bg-green-500');
                    startBtn.disabled = true;
                    pauseBtn.disabled = false;
                } else {
                    indicatorElement.classList.add('status-paused', 'bg-red-500');
                    startBtn.disabled = false;
                    pauseBtn.disabled = true;
                }
            }
        }

        // Updates the portfolio summary and owned stocks list
        async function updatePortfolio() {
            const data = await fetchData('/portfolio/summary');
            if (data && data.portfolio) {
                document.getElementById('cash').textContent = `$${data.portfolio.cash.toFixed(2)}`;
                document.getElementById('total-value').textContent = `$${data.portfolio.total_portfolio_value.toFixed(2)}`;
                
                const profitLossElement = document.getElementById('profit-loss');
                profitLossElement.textContent = `$${data.portfolio.profit_loss.toFixed(2)}`;
                profitLossElement.classList.remove('green', 'red');
                if (data.portfolio.profit_loss > 0) {
                    profitLossElement.classList.add('green');
                } else if (data.portfolio.profit_loss < 0) {
                    profitLossElement.classList.add('red');
                }

                document.getElementById('owned-stocks-count').textContent = Object.keys(data.portfolio.owned_stocks).length;
                document.getElementById('total-trades').textContent = data.total_trades;
                document.getElementById('winning-trades').textContent = data.winning_trades;
                document.getElementById('losing-trades').textContent = data.losing_trades;

                // Update owned stocks list
                const ownedStocksContainer = document.getElementById('owned-stocks-container');
                ownedStocksContainer.innerHTML = '';
                if (Object.keys(data.portfolio.owned_stocks).length === 0) {
                    ownedStocksContainer.innerHTML = '<p id="no-stocks-message" class="text-gray-400">No stocks currently owned.</p>';
                } else {
                    for (const [symbol, stockData] of Object.entries(data.portfolio.owned_stocks)) {
                        const stockCard = document.createElement('div');
                        stockCard.className = 'card p-4 rounded-lg border border-gray-600 flex justify-between items-center';
                        stockCard.innerHTML = `
                            <div>
                                <h3 class="text-lg font-bold">${symbol}</h3>
                                <p class="text-sm">Quantity: ${stockData.quantity.toFixed(4)}</p>
                                <p class="text-sm">Avg. Buy Price: $${stockData.average_buy_price.toFixed(2)}</p>
                            </div>
                            <div class="text-right">
                                <p class="text-sm font-semibold">Current Value</p>
                                <p class="text-lg font-bold">$${stockData.current_value.toFixed(2)}</p>
                            </div>
                        `;
                        ownedStocksContainer.appendChild(stockCard);
                    }
                }
            }
        }

        // Updates the trade history table
        async function updateTrades() {
            const data = await fetchData('/trades');
            const tableBody = document.getElementById('trades-table-body');
            tableBody.innerHTML = '';
            if (data && data.length > 0) {
                data.forEach(trade => {
                    const row = document.createElement('tr');
                    const isBuy = trade.action === 'BUY';
                    const actionClass = isBuy ? 'green' : 'red';
                    row.className = 'hover:bg-gray-700 transition duration-150 ease-in-out';
                    row.innerHTML = `
                        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-300">${new Date(trade.timestamp).toLocaleString()}</td>
                        <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-white">${trade.symbol}</td>
                        <td class="px-6 py-4 whitespace-nowrap text-sm font-semibold ${actionClass}">${trade.action}</td>
                        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-300">${trade.quantity.toFixed(4)}</td>
                        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-300">$${trade.price.toFixed(2)}</td>
                        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-300">${trade.confidence.toFixed(2)}</td>
                        <td class="px-6 py-4 whitespace-normal text-sm text-gray-400 max-w-xs">${trade.reasoning}</td>
                    `;
                    tableBody.appendChild(row);
                });
            } else {
                tableBody.innerHTML = '<tr><td colspan="7" class="px-6 py-4 whitespace-nowrap text-center text-gray-400">No trades yet.</td></tr>';
            }
        }

        async function refreshData() {
            await updateBotStatus();
            await updatePortfolio();
            await updateTrades();
        }

        // Event Listeners for buttons
        document.getElementById('start-btn').addEventListener('click', () => sendCommand('/bot/start'));
        document.getElementById('pause-btn').addEventListener('click', () => sendCommand('/bot/pause'));
        document.getElementById('reset-btn').addEventListener('click', () => {
            if (confirm("Are you sure you want to reset the entire portfolio? This action cannot be undone.")) {
                sendCommand('/portfolio/reset');
            }
        });

        // Initial data load and periodic refresh
        document.addEventListener('DOMContentLoaded', () => {
            refreshData();
            // Refresh every 30 seconds
            setInterval(refreshData, 30000); 
        });
    </script>
</body>
</html>
