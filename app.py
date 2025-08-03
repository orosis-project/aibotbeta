# app.py
# Final Version: API rate limiting, instant AI activation, multi-asset trading, dynamic trade sizing, and auto-pause.

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
import asyncio
import httpx

# --- Configuration ---
GEMINI_API_KEYS = [
    os.environ.get("GEMINI_API_KEY"),
    os.environ.get("GEMINI_API_KEY_2"),
    os.environ.get("GEMINI_API_KEY_3"),
    os.environ.get("GEMINI_API_KEY_4"),
    os.environ.get("GEMINI_API_KEY_5")
]
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
ADMIN_PASSWORD = "orosis"

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
INITIAL_CASH = 5000.00
DB_FILE = "trades.db"
BASE_TRADE_PERCENTAGE = 0.05
LOOP_INTERVAL_SECONDS = 300
STOCKS_TO_SCAN_PER_CYCLE = 15
INITIAL_BUY_COUNT = 10
FINNHUB_RATE_LIMIT_SECONDS = 2.0
GEMINI_RATE_LIMIT_SECONDS = 10.0

# --- Bot State ---
bot_status_lock = Lock()
bot_is_running = True
current_api_key_index = 0
all_keys_exhausted = False
historical_performance = []

# --- AI Configuration (Lazy Initialization) ---
ai_models = {}
ai_model_lock = Lock()
ai_model_configured = False
_last_gemini_request_time = 0

def _enforce_gemini_rate_limit():
    """Ensures a minimum delay between Gemini API requests."""
    global _last_gemini_request_time
    elapsed = time.time() - _last_gemini_request_time
    if elapsed < GEMINI_RATE_LIMIT_SECONDS:
        time.sleep(GEMINI_RATE_LIMIT_SECONDS - elapsed)
    _last_gemini_request_time = time.time()


def get_ai_model(api_key_index):
    """Retrieves a configured AI model or falls back to another key."""
    if not ai_model_configured:
        configure_ai_models()
        if not ai_model_configured:
            return None
    
    _enforce_gemini_rate_limit()

    # Try to use the preferred key, then cycle through others
    key_order = list(range(len(GEMINI_API_KEYS)))
    # Ensure the preferred key is checked first
    key_order.remove(api_key_index)
    key_order.insert(0, api_key_index)

    for i in key_order:
        model_name = f'gemini-1.5-flash-{i}'
        if model_name in ai_models:
            return ai_models[model_name]
            
    return None

class RiskData:
    """Predefined risk profiles for different market sectors."""
    SECTOR_RISKS = {
        "Tech": {"risk_level": "High", "description": "High growth, high volatility."},
        "Healthcare": {"risk_level": "Medium", "description": "Stable but can have high-risk biotech plays."},
        "Finance": {"risk_level": "Medium", "description": "Established, but sensitive to economic changes."},
        "Consumer Staples": {"risk_level": "Low", "description": "Stable, less susceptible to market swings."},
        "Crypto": {"risk_level": "Very High", "description": "Extremely volatile, high-reward, high-risk."},
        "Forex": {"risk_level": "Medium", "description": "Stable, but influenced by global events."},
        "Default": {"risk_level": "Medium", "description": "General market risk."}
    }

def get_sector_for_symbol(symbol):
    if symbol.startswith("BINANCE:"):
        return "Crypto"
    elif symbol.startswith("OANDA:"):
        return "Forex"
    tech_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
    healthcare_stocks = ["UNH", "JNJ", "LLY", "ABBV", "PFE"]
    finance_stocks = ["JPM", "V", "MA", "GS", "MS", "AXP", "C", "WFC", "BAC"]
    consumer_staples = ["PG", "KO", "PEP", "WMT", "COST", "MCD"]
    
    if symbol in tech_stocks:
        return "Tech"
    elif symbol in healthcare_stocks:
        return "Healthcare"
    elif symbol in finance_stocks:
        return "Finance"
    elif symbol in consumer_staples:
        return "Consumer Staples"
    else:
        return "Default"

def configure_ai_models():
    """Configures all AI models on startup."""
    global ai_models, ai_model_configured, all_keys_exhausted
    with ai_model_lock:
        if ai_model_configured:
            return
        
        if not GEMINI_API_KEYS or all(key is None for key in GEMINI_API_KEYS):
            print("ERROR: No Gemini API keys found in environment variables. Bot will not run.")
            all_keys_exhausted = True
            bot_is_running = False
            return
            
        for i, api_key in enumerate(GEMINI_API_KEYS):
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    ai_models[f'gemini-1.5-flash-{i}'] = genai.GenerativeModel('gemini-1.5-flash')
                    print(f"Gemini AI model configured for key index {i}.")
                except Exception as e:
                    print(f"ERROR: Failed to configure Gemini AI with key index {i}: {e}")
        
        if ai_models:
            ai_model_configured = True
        else:
            print("All API keys failed to configure. Pausing bot.")
            all_keys_exhausted = True
            bot_is_running = False


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
        print("Portfolio Manager initialized.")
        if not get_all_trades():
            print("No trades found. Initiating initial stock purchase.")
            Thread(target=self.buy_initial_stocks).start()

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
        assets_to_buy = self.api_client.get_initial_assets_to_buy()
        if not assets_to_buy:
            print("Failed to get initial assets to buy.")
            return

        for symbol in assets_to_buy:
            price = self.api_client.get_quote(symbol)
            initial_trade_amount = self.initial_cash * BASE_TRADE_PERCENTAGE
            if price and self.cash >= initial_trade_amount:
                quantity = initial_trade_amount / price
                self.buy_stock(symbol, quantity, price, "Initial portfolio seeding.", 0.5)
            else:
                print(f"Skipping initial buy for {symbol}.")
        print("Initial buy-in complete.")
        self._reconstruct_portfolio_from_db()


    def get_portfolio_status(self):
        self._reconstruct_portfolio_from_db()
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
            
    def get_historical_performance(self):
        with self._lock:
            all_trades = get_all_trades()
            historical_data = []
            
            temp_cash = self.initial_cash
            temp_stocks = {}

            if not all_trades:
                 historical_data.append({
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "value": self.initial_cash
                })
                 return historical_data

            all_symbols = list(set([trade['symbol'] for trade in all_trades]))
            
            all_current_prices = asyncio.run(self.api_client.get_quotes_async(all_symbols))

            for trade in all_trades:
                symbol, quantity, price, action = trade['symbol'], trade['quantity'], trade['price'], trade['action']
                timestamp = trade['timestamp']
                
                if action == 'BUY':
                    temp_cash -= quantity * price
                    temp_stocks[symbol] = temp_stocks.get(symbol, 0) + quantity
                elif action == 'SELL':
                    temp_cash += quantity * price
                    if symbol in temp_stocks:
                        temp_stocks[symbol] -= quantity
                        if temp_stocks[symbol] < 1e-6:
                            del temp_stocks[symbol]
                            
                current_value = temp_cash
                for stock_symbol, qty in temp_stocks.items():
                    current_value += qty * all_current_prices.get(stock_symbol, 0)
                        
                historical_data.append({
                    "timestamp": timestamp,
                    "value": current_value
                })
            
            return historical_data


    def buy_stock(self, symbol, quantity, price, reasoning, confidence):
        with self._lock:
            cost = quantity * price
            if self.cash < cost:
                print(f"FAILURE: Insufficient cash to buy {quantity:.4f} of {symbol}. Cash: ${self.cash:.2f}, Cost: ${cost:.2f}")
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
        self.nyse_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "GOOG", "BRK.B", "UNH", "JPM", "JNJ", "V", "XOM", "MA", "PG", "HD", "CVX", "LLY", "ABBV", "PFE", "BAC", "KO", "TMO", "PEP", "AVGO", "WMT", "COST", "MCD", "CSCO", "SPY", "QQQ", "DIA", "IWM", "GS", "MS", "AXP", "C", "WFC", "T", "VZ", "TMUS", "TSLA", "F", "GM", "RIVN", "F", "GM", "RIVN", "F", "GM", "RIVN", "F", "GM", "RIVN"]
        self.crypto_pairs = ["BINANCE:BTCUSDT", "BINANCE:ETHUSDT", "BINANCE:XRPUSDT", "BINANCE:DOGEUSDT"]
        self.forex_pairs = ["OANDA:EURUSD", "OANDA:GBPUSD", "OANDA:USDJPY", "OANDA:USDCAD"]
        self._last_request_time = 0
        print("Finnhub Client initialized.")

    def _enforce_rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < FINNHUB_RATE_LIMIT_SECONDS:
            time.sleep(FINNHUB_RATE_LIMIT_SECONDS - elapsed)
        self._last_request_time = time.time()

    async def get_quotes_async(self, symbols):
        self._enforce_rate_limit()
        async with httpx.AsyncClient() as client:
            tasks = []
            for symbol in symbols:
                params = {'symbol': symbol.upper(), 'token': os.environ.get("FINNHUB_API_KEY")}
                tasks.append(client.get(f"{FINNHUB_BASE_URL}/quote", params=params))
                
            responses = await asyncio.gather(*tasks)
            
            quotes = {}
            for i, response in enumerate(responses):
                if response.status_code == 200:
                    data = response.json()
                    quotes[symbols[i]] = data.get('c')
                else:
                    print(f"ERROR: Finnhub request failed for {symbols[i]}: {response.status_code}")
                    quotes[symbols[i]] = None
            
            return quotes

    def _make_request(self, endpoint, params=None):
        api_key = os.environ.get("FINNHUB_API_KEY")
        if not api_key:
            print("ERROR: FINNHUB_API_KEY not found in environment.")
            return None
        if params is None:
            params = {}
        params['token'] = api_key
        
        self._enforce_rate_limit()

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

    def get_initial_assets_to_buy(self):
        all_assets = self.nyse_stocks + self.crypto_pairs + self.forex_pairs
        return random.sample(all_assets, min(INITIAL_BUY_COUNT, len(all_assets)))

    def get_assets_to_analyze(self, owned_assets):
        num_stocks_to_sample = STOCKS_TO_SCAN_PER_CYCLE - len(owned_assets)
        if num_stocks_to_sample < 0:
            num_stocks_to_sample = 0
        
        all_tradable_assets = self.nyse_stocks + self.crypto_pairs + self.forex_pairs
        random_sample = random.sample(all_tradable_assets, min(num_stocks_to_sample, len(all_tradable_assets)))
        
        return list(set(random_sample + owned_assets))


# --- AI Decision Making & Bot Loop ---
def get_ai_decision_and_analysis(symbol, price, news, portfolio, recent_trades, market_news, trade_count):
    """
    Combines AI analysis and decision into a single function to save API calls.
    Uses GEMINI_API_KEY (index 0) with fallback.
    """
    ai_model = get_ai_model(0)
    if not ai_model: return None

    sector = get_sector_for_symbol(symbol)
    risk_profile = RiskData.SECTOR_RISKS.get(sector, RiskData.SECTOR_RISKS["Default"])

    prompt = f"""
    You are an expert stock and currency trading analyst bot. Your goal is to analyze real-time market data and make a strategic trading decision. The assets you can analyze include stocks, cryptocurrencies, and forex pairs.
    
    **Instructions for Analysis and Decision:**
    - First, perform a multi-factor analysis of the asset's potential, including its risk profile, news sentiment, and market position.
    - Second, based on your analysis, make a final, deliberate BUY, SELL, or HOLD decision.
    - Your strategy should be to look at the bigger picture and long-term trends. Do not sell at the first sign of minor trouble unless the analysis strongly indicates a fundamental change.
    - Use the `trade_size_multiplier` to take "SMART risks":
        - For a stable asset (e.g., in Consumer Staples) with positive analysis, use a higher multiplier.
        - For a high-risk, high-reward asset (e.g., in Tech or Crypto) with positive analysis, use a carefully chosen moderate multiplier to limit exposure while still capitalizing on potential gains.
    - Your `reasoning` must clearly justify your decision by referencing the provided data and your strategic outlook.

    **Current Information:**
    - **Current Time:** {datetime.now()}
    - **Asset Symbol:** {symbol}
    - **Current Price:** ${price:.2f}
    - **Recent News for {symbol}:** {json.dumps(news, indent=2)}
    - **Current Portfolio Status:** {json.dumps(portfolio, indent=2)}
    - **Recent Trades by Bot:** {json.dumps(recent_trades, indent=2)}
    - **General Market News:** {json.dumps(market_news[:5], indent=2)}
    - **Total Trades Made:** {trade_count}
    - **Asset Sector:** {sector}
    - **Sector Risk Profile:** {risk_profile['description']}

    Format your response as a JSON object with the following keys:
    {{
        "action": "BUY" or "SELL" or "HOLD",
        "reasoning": "A concise explanation for your decision, based on the analysis and bigger picture.",
        "confidence": 0.0-1.0,
        "trade_size_multiplier": 0.5-2.0
    }}
    """
    
    try:
        response = ai_model.generate_content(prompt)
        decision_text = response.text.strip().replace("```json", "").replace("```", "")
        decision = json.loads(decision_text)
        print(f"AI Analysis and Decision for {symbol}: {decision}")
        return decision
    except google_exceptions.ResourceExhausted as e:
        print(f"CRITICAL ERROR: Gemini API key limit reached. Falling back.")
        return None
    except Exception as e:
        print(f"ERROR: Failed to get or parse AI decision for {symbol}: {e}")
        return None

def get_ai_inquiry(question, portfolio_status, recent_trades):
    """AI #3: Admin Panel Inquiry. Uses GEMINI_API_KEY_2 or fallback."""
    
    ai_model_inquiry = get_ai_model(2)
    if not ai_model_inquiry: return "Error: AI model for inquiries is not available."
    
    try:
        prompt = f"""
        You are an AI stock bot assistant. Your task is to answer questions about the bot's portfolio, trading strategy, and market conditions based on the provided data.
        
        **Bot's Current Portfolio:**
        {json.dumps(portfolio_status, indent=2)}
        
        **Bot's Recent Trades:**
        {json.dumps(recent_trades, indent=2)}
        
        **User's Question:**
        {question}
        
        Provide a helpful and concise answer to the user's question.
        """
        response = ai_model_inquiry.generate_content(prompt)
        answer = response.text
        return answer
        
    except google_exceptions.ResourceExhausted as e:
        print(f"CRITICAL ERROR: Gemini API key limit reached for inquiries. Falling back.")
        return "Error: API quota for inquiries has been exhausted. Please try again later."
    except Exception as e:
        print(f"Error in ask_ai with key #3: {e}")
        return "Error: Failed to get a response from the AI."


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
        confidence_threshold = 0.65 

        print(f"Current trade count: {trade_count}. Confidence threshold set to {confidence_threshold * 100}%.")

        portfolio = portfolio_manager.get_portfolio_status()
        owned_assets = list(portfolio['owned_stocks'].keys())
        market_news = finnhub_client.get_market_news()
        assets_to_analyze = finnhub_client.get_assets_to_analyze(owned_assets)
        print(f"This cycle, analyzing: {assets_to_analyze}")
        print("Starting AI decision-making for this cycle.")

        for symbol in assets_to_analyze:
            with bot_status_lock:
                if not bot_is_running:
                    break

            print(f"Analyzing {symbol}...")
            price = finnhub_client.get_quote(symbol)
            if not price:
                continue

            news = finnhub_client.get_company_news(symbol)
            trades = get_recent_trades(5)
            current_portfolio_status = portfolio
            
            ai_decision = get_ai_decision_and_analysis(symbol, price, news, current_portfolio_status, trades, market_news, trade_count)
            
            if ai_decision and ai_decision.get('confidence', 0) > confidence_threshold:
                action = ai_decision.get('action', '').upper()
                reasoning = ai_decision.get('reasoning', '')
                confidence = ai_decision.get('confidence', 0)
                trade_size_multiplier = ai_decision.get('trade_size_multiplier', 1.0)
                
                dynamic_trade_amount = (current_portfolio_status['total_portfolio_value'] * BASE_TRADE_PERCENTAGE) * trade_size_multiplier
                dynamic_trade_amount = min(dynamic_trade_amount, current_portfolio_status['cash'])
                
                print(f"Decision: {action} with confidence {confidence:.2f}. Dynamic Trade Amount: ${dynamic_trade_amount:.2f}")

                if action == 'BUY':
                    if dynamic_trade_amount >= price:
                        quantity = dynamic_trade_amount / price
                        portfolio_manager.buy_stock(symbol, quantity, price, reasoning, confidence)
                elif action == 'SELL':
                    if symbol in current_portfolio_status['owned_stocks']:
                        quantity_to_sell = min(dynamic_trade_amount / price, current_portfolio_status['owned_stocks'][symbol]['quantity'])
                        if quantity_to_sell > 0:
                            portfolio_manager.sell_stock(symbol, quantity_to_sell, price, reasoning, confidence)
        
        print(f"--- Cycle finished. Waiting {LOOP_INTERVAL_SECONDS}s. ---")
        time.sleep(LOOP_INTERVAL_SECONDS)


# --- Scheduler Loop for Daily Resume ---
def scheduler_loop():
    global bot_is_running, all_keys_exhausted, current_api_key_index
    while True:
        now_utc = datetime.now(timezone.utc)
        tomorrow_utc = now_utc + timedelta(days=1)
        midnight_utc = tomorrow_utc.replace(hour=0, minute=1, second=0, microsecond=0)
        sleep_seconds = (midnight_utc - now_utc).total_seconds()
        
        print(f"Scheduler: Sleeping for {sleep_seconds / 3600:.2f} hours until quota reset.")
        time.sleep(sleep_seconds)

        with bot_status_lock:
            if all_keys_exhausted:
                print("Scheduler: API quotas have reset. Resuming bot.")
                bot_is_running = True
                all_keys_exhausted = False
                current_api_key_index = 0

# --- Global Instances & App Initialization ---
init_db()
configure_ai_models()
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

@app.route("/api/portfolio/history", methods=['GET'])
def get_portfolio_history():
    return jsonify(portfolio_manager.get_historical_performance())

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
    global ai_model_configured
    if not ai_model_configured:
        configure_ai_models()
        if not ai_model_configured:
            return jsonify({"answer": "Error: AI models are not configured."}), 500
    
    question = request.json.get('question')
    if not question:
        return jsonify({"answer": "Error: No question provided."}), 400
        
    try:
        portfolio_status = portfolio_manager.get_portfolio_status()
        recent_trades = get_recent_trades(5)
        
        answer = get_ai_inquiry(question, portfolio_status, recent_trades)
        return jsonify({"answer": answer})
        
    except Exception as e:
        print(f"Error in ask_ai: {e}")
        return jsonify({"answer": "Error: Failed to get a response from the AI."}), 500

# --- Main Execution ---
if __name__ == "__main__":
    bot_thread = Thread(target=bot_trading_loop, args=(portfolio_manager, finnhub_client), daemon=True)
    bot_thread.start()
    scheduler_thread = Thread(target=scheduler_loop, daemon=True)
    scheduler_thread.start()

    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)

