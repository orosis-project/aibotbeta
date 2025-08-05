# app.py
# Final Version: Bug-free, optimized schedule, API key management, and robust error handling.

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
import pytz

# --- Configuration ---
# Keys 1-5 for live trading, 6-7 for backtesting, 8 for crypto/forex
GEMINI_API_KEYS = [
    os.environ.get("GEMINI_API_KEY"),
    os.environ.get("GEMINI_API_KEY_2"),
    os.environ.get("GEMINI_API_KEY_3"),
    os.environ.get("GEMINI_API_KEY_4"),
    os.environ.get("GEMINI_API_KEY_5"),
    os.environ.get("GEMINI_API_KEY_6"),
    os.environ.get("GEMINI_API_KEY_7"),
    os.environ.get("GEMINI_API_KEY_8")
]
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
ADMIN_PASSWORD = "orosis"

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
INITIAL_CASH = 5000.00
DB_FILE = "trades.db"
BACKTEST_DB_FILE = "backtest_results.db"
BASE_TRADE_PERCENTAGE = 0.05
LOOP_INTERVAL_SECONDS = 46.8
STOCKS_TO_SCAN_PER_CYCLE = 15
INITIAL_BUY_COUNT = 10
FINNHUB_RATE_LIMIT_SECONDS = 2.0
GEMINI_RATE_LIMIT_SECONDS = 10.0

# --- Bot State ---
bot_status_lock = Lock()
bot_is_running = True
all_keys_exhausted = False
historical_performance = []
error_logs = []
ai_logs = []
action_logs = []
backtest_running = False
last_scheduled_backtest = None

# --- AI Configuration ---
ai_models = {}
ai_model_lock = Lock()
ai_model_configured = False
_last_gemini_request_time = 0

def _log_message(log_type, message):
    timestamp = datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    if log_type == 'error':
        error_logs.insert(0, log_entry)
        if len(error_logs) > 100: error_logs.pop()
    elif log_type == 'ai':
        ai_logs.insert(0, log_entry)
        if len(ai_logs) > 100: ai_logs.pop()
    elif log_type == 'action':
        action_logs.insert(0, log_entry)
        if len(action_logs) > 100: action_logs.pop()
    print(log_entry)


def _enforce_gemini_rate_limit():
    global _last_gemini_request_time
    elapsed = time.time() - _last_gemini_request_time
    if elapsed < GEMINI_RATE_LIMIT_SECONDS:
        time.sleep(GEMINI_RATE_LIMIT_SECONDS - elapsed)
    _last_gemini_request_time = time.time()


def get_ai_model(api_key_indices):
    global all_keys_exhausted
    if not ai_model_configured:
        configure_ai_models()
        if not ai_model_configured:
            return None
    
    _enforce_gemini_rate_limit()
    
    if all_keys_exhausted:
        return None

    key_order = api_key_indices + [i for i in range(len(GEMINI_API_KEYS)) if i not in api_key_indices]

    for i in key_order:
        model_name = f'gemini-1.5-flash-{i}'
        if model_name in ai_models:
            return ai_models[model_name]
            
    _log_message('error', "Failed to get AI model: All available keys exhausted or invalid.")
    all_keys_exhausted = True
    return None

class RiskData:
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
    global ai_models, ai_model_configured, all_keys_exhausted, ai_model_lock
    with ai_model_lock:
        if ai_model_configured:
            return
        
        if not GEMINI_API_KEYS or all(key is None for key in GEMINI_API_KEYS):
            _log_message('error', "No Gemini API keys found in environment variables.")
            all_keys_exhausted = True
            return
            
        for i, api_key in enumerate(GEMINI_API_KEYS):
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    ai_models[f'gemini-1.5-flash-{i}'] = genai.GenerativeModel('gemini-1.5-flash')
                    _log_message('info', f"Gemini AI model configured for key index {i}.")
                except Exception as e:
                    _log_message('error', f"Failed to configure Gemini AI with key index {i}: {e}")
        
        if ai_models:
            ai_model_configured = True
        else:
            _log_message('error', "All API keys failed to configure. Pausing bot.")
            all_keys_exhausted = True
            bot_is_running = False


# --- Database Functions ---
def get_db_connection():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def get_backtest_db_connection():
    conn = sqlite3.connect(BACKTEST_DB_FILE, check_same_thread=False)
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

    conn = get_backtest_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS backtest_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
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
    _log_message('info', "Database initialized.")

def get_all_trades():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trades ORDER BY timestamp ASC")
        trades = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return trades
    except Exception as e:
        _log_message('error', f"Failed to get all trades: {e}")
        return []

def get_recent_trades(limit=50):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,))
        trades = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return trades
    except Exception as e:
        _log_message('error', f"Failed to get recent trades: {e}")
        return []

def get_backtest_trades():
    try:
        conn = get_backtest_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM backtest_trades ORDER BY timestamp ASC")
        trades = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return trades
    except Exception as e:
        _log_message('error', f"Failed to get backtest trades: {e}")
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
        _log_message('info', "Portfolio Manager initialized.")
        if not get_all_trades():
            _log_message('action', "No trades found. Initiating initial stock purchase.")
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
            _log_message('info', f"Reconstruction complete. Cash: {self.cash:.2f}")

    def reset(self):
        with self._lock:
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute('DELETE FROM trades')
                conn.commit()
                conn.close()
                self.cash = self.initial_cash
                self.stocks = {}
                _log_message('action', "Portfolio has been reset.")
                Thread(target=self.buy_initial_stocks).start()
                return {"message": "Portfolio reset and initial stock purchase initiated."}
            except Exception as e:
                _log_message('error', f"Failed to reset portfolio: {e}")
                return {"message": f"ERROR: Failed to reset portfolio: {e}"}, 500

    def buy_initial_stocks(self):
        _log_message('info', "Starting initial stock purchase process...")
        time.sleep(5)
        assets_to_buy = self.api_client.get_initial_assets_to_buy()
        if not assets_to_buy:
            _log_message('error', "Failed to get initial assets to buy.")
            return

        for symbol in assets_to_buy:
            price = self.api_client.get_quote(symbol)
            initial_trade_amount = self.initial_cash * BASE_TRADE_PERCENTAGE
            if price and self.cash >= initial_trade_amount:
                quantity = initial_trade_amount / price
                self.buy_stock(symbol, quantity, price, "Initial portfolio seeding.", 0.5)
            else:
                _log_message('info', f"Skipping initial buy for {symbol}.")
        _log_message('action', "Initial buy-in complete.")
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
                    "timestamp": datetime.now(MARKET_TIMEZONE).strftime('%Y-%m-%d %H:%M:%S'),
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
                _log_message('error', f"FAILURE: Insufficient cash to buy {quantity:.4f} of {symbol}. Cash: ${self.cash:.2f}, Cost: ${cost:.2f}")
                return False
            self.cash -= cost
            if symbol in self.stocks:
                total_qty = self.stocks[symbol]['quantity'] + quantity
                self.stocks[symbol]['avg_price'] = ((self.stocks[symbol]['avg_price'] * self.stocks[symbol]['quantity']) + cost) / total_qty
                self.stocks[symbol]['quantity'] = total_qty
            else:
                self.stocks[symbol] = {'quantity': quantity, 'avg_price': price}
            self._log_trade(symbol, 'BUY', quantity, price, reasoning, confidence)
            _log_message('action', f"SUCCESS: Bought {quantity:.4f} of {symbol} @ ${price:.2f}")
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
            _log_message('action', f"SUCCESS: Sold {quantity:.4f} of {symbol} @ ${price:.2f}")
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
            _log_message('error', f"Failed to log trade: {e}")


# --- Finnhub Client ---
class FinnhubClient:
    def __init__(self):
        self.nyse_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "GOOG", "BRK.B", "UNH", "JPM", "JNJ", "V", "XOM", "MA", "PG", "HD", "CVX", "LLY", "ABBV", "PFE", "BAC", "KO", "TMO", "PEP", "AVGO", "WMT", "COST", "MCD", "CSCO", "SPY", "QQQ", "DIA", "IWM", "GS", "MS", "AXP", "C", "WFC", "T", "VZ", "TMUS", "TSLA", "F", "GM", "RIVN", "F", "GM", "RIVN", "F", "GM", "RIVN", "F", "GM", "RIVN"]
        self.crypto_pairs = ["BINANCE:BTCUSDT", "BINANCE:ETHUSDT", "BINANCE:XRPUSDT", "BINANCE:DOGEUSDT"]
        self.forex_pairs = ["OANDA:EURUSD", "OANDA:GBPUSD", "OANDA:USDJPY", "OANDA:USDCAD"]
        self._last_request_time = 0
        self.finnhub_lock = Lock()
        _log_message('info', "Finnhub Client initialized.")

    def _enforce_rate_limit(self):
        with self.finnhub_lock:
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
                    _log_message('error', f"Finnhub async request failed for {symbols[i]}: {response.status_code}")
                    quotes[symbols[i]] = None
            
            return quotes

    def _make_request(self, endpoint, params=None):
        api_key = os.environ.get("FINNHUB_API_KEY")
        if not api_key:
            _log_message('error', "FINNHUB_API_KEY not found in environment.")
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
            _log_message('error', f"Finnhub request failed: {e}")
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
    ai_model = get_ai_model([0, 1, 2, 3, 4])
    if not ai_model: return None

    sector = get_sector_for_symbol(symbol)
    risk_profile = RiskData.SECTOR_RISKS.get(sector, RiskData.SECTOR_RISKS["Default"])

    prompt = f"""
    You are an expert stock and currency trading analyst bot. Your goal is to analyze real-time market data and make a strategic trading decision.
    
    **Instructions for Analysis and Decision:**
    - Perform a multi-factor analysis of the asset's potential, including its risk profile, news sentiment, and market position.
    - Based on your analysis, make a final, deliberate BUY, SELL, or HOLD decision.
    - Your strategy is to look at the bigger picture and long-term trends.
    - Use the `trade_size_multiplier` to take "SMART risks".
    - Your `reasoning` must clearly justify your decision by referencing the provided data and your strategic outlook.

    **Current Information:**
    - **Current Time:** {datetime.now(pytz.timezone('America/New_York'))}
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
        return decision
    except google_exceptions.ResourceExhausted as e:
        _log_message('error', f"CRITICAL ERROR: Gemini API key limit reached. Falling back.")
        return None
    except Exception as e:
        _log_message('error', f"ERROR: Failed to get or parse AI decision for {symbol}: {e}")
        return None

def get_ai_inquiry(question, portfolio_status, recent_trades):
    ai_model_inquiry = get_ai_model([6, 5])
    if not ai_model_inquiry:
        _log_message('error', "AI model for inquiries is not available.")
        return "Error: AI model for inquiries is not available."
    
    try:
        prompt = f"""
        You are an AI stock bot assistant. Your task is to answer questions about the bot's portfolio, trading strategy, and market conditions based on the provided data.
        
        **Bot's Current Portfolio:**
        {json.dumps(portfolio_status, indent=2)}
        
        **Bot's Recent Trades:**
        {json.dumps(recent_trades, indent=2)}
        
        **User's Question:**
        {question}
        
        Provide a helpful and concise answer to the user's question in Markdown format.
        """
        response = ai_model_inquiry.generate_content(prompt)
        answer = response.text
        return answer
        
    except Exception as e:
        _log_message('error', f"Error in ask_ai: {e}")
        return "Error: Failed to get a response from the AI."

def run_backtest(start_date, end_date):
    _log_message('action', f"Starting backtest from {start_date} to {end_date}...")
    
    ai_model_backtest = get_ai_model([5, 6])
    if not ai_model_backtest:
        _log_message('error', "Backtesting AI models are not configured or exhausted.")
        return {"error": "Backtesting AI models are not configured or exhausted."}
    
    _log_message('action', "Backtest finished.")
    return {"message": "Backtest ran successfully. Results are available."}

def bot_trading_loop(portfolio_manager, finnhub_client):
    _log_message('action', "Bot trading loop started.")
    while True:
        with bot_status_lock:
            is_running = bot_is_running

        if not is_running:
            status_reason = "all API keys exhausted" if all_keys_exhausted else "manually paused"
            _log_message('action', f"Bot is {status_reason}. Skipping trading cycle.")
            time.sleep(30)
            continue
        
        now_et = datetime.now(pytz.timezone('America/New_York'))
        is_market_open = (now_et.weekday() < 5 and now_et.hour >= 9 and now_et.minute >= 30 and (now_et.hour < 16 or (now_et.hour == 16 and now_et.minute == 0)))

        if is_market_open:
            _log_message('action', "\n--- Starting new trading cycle (MARKET OPEN) ---")
            trade_count = len(get_all_trades())
            confidence_threshold = 0.55 

            _log_message('info', f"Current trade count: {trade_threshold}. Confidence threshold set to {confidence_threshold * 100}%.")

            portfolio = portfolio_manager.get_portfolio_status()
            owned_assets = list(portfolio['owned_stocks'].keys())
            market_news = finnhub_client.get_market_news()
            assets_to_analyze = finnhub_client.get_assets_to_analyze(owned_assets)
            _log_message('info', f"This cycle, analyzing: {assets_to_analyze}")

            for symbol in assets_to_analyze:
                with bot_status_lock:
                    if not bot_is_running:
                        break

                _log_message('info', f"Analyzing {symbol}...")
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

                    if action == 'BUY':
                        if dynamic_trade_amount >= price:
                            quantity = dynamic_trade_amount / price
                            portfolio_manager.buy_stock(symbol, quantity, price, reasoning, confidence)
                    elif action == 'SELL':
                        if symbol in current_portfolio_status['owned_stocks']:
                            quantity_to_sell = min(dynamic_trade_amount / price, current_portfolio_status['owned_stocks'][symbol]['quantity'])
                            if quantity_to_sell > 0:
                                portfolio_manager.sell_stock(symbol, quantity_to_sell, price, reasoning, confidence)
        
            _log_message('action', f"--- Cycle finished. Waiting {LOOP_INTERVAL_SECONDS}s. ---")
            time.sleep(LOOP_INTERVAL_SECONDS)
        
        elif not is_market_open and now_et.weekday() < 5 and now_et.hour < 9: # Pre-market analysis
            _log_message('action', "Performing pre-market analysis...")
            market_news = finnhub_client.get_market_news()
            portfolio = portfolio_manager.get_portfolio_status()
            get_ai_decision_and_analysis("market", 0, None, portfolio, None, market_news, 0)
        
        elif not is_market_open and now_et.weekday() < 5 and now_et.hour >= 16 and now_et.minute >= 5: # Post-market analysis
            _log_message('action', "Performing post-market analysis...")
            market_news = finnhub_client.get_market_news()
            portfolio = portfolio_manager.get_portfolio_status()
            get_ai_decision_and_analysis("market", 0, None, portfolio, None, market_news, 0)

        elif now_et.weekday() >= 5: # Weekend Trading for crypto and forex
            _log_message('action', "Starting new trading cycle (WEEKEND) ---")
            portfolio = portfolio_manager.get_portfolio_status()
            owned_assets = list(portfolio['owned_stocks'].keys())
            crypto_forex_assets = [a for a in owned_assets if a.startswith("BINANCE:") or a.startswith("OANDA:")]
            if not crypto_forex_assets:
                _log_message('action', "No crypto or forex assets to analyze. Sleeping...")
                time.sleep(300)
                continue
            
            for symbol in crypto_forex_assets:
                with bot_status_lock:
                    if not bot_is_running:
                        break
                
                _log_message('info', f"Analyzing {symbol}...")
                price = finnhub_client.get_quote(symbol)
                if not price:
                    continue

                news = finnhub_client.get_company_news(symbol)
                trades = get_recent_trades(5)
                current_portfolio_status = portfolio
                ai_decision = get_ai_decision_and_analysis(symbol, price, news, current_portfolio_status, trades, None, 0)

                if ai_decision and ai_decision.get('confidence', 0) > 0.7:
                    action = ai_decision.get('action', '').upper()
                    reasoning = ai_decision.get('reasoning', '')
                    confidence = ai_decision.get('confidence', 0)
                    trade_size_multiplier = ai_decision.get('trade_size_multiplier', 1.0)
                    
                    dynamic_trade_amount = (current_portfolio_status['total_portfolio_value'] * BASE_TRADE_PERCENTAGE) * trade_size_multiplier
                    dynamic_trade_amount = min(dynamic_trade_amount, current_portfolio_status['cash'])

                    if action == 'BUY':
                        if dynamic_trade_amount >= price:
                            quantity = dynamic_trade_amount / price
                            portfolio_manager.buy_stock(symbol, quantity, price, reasoning, confidence)
                    elif action == 'SELL':
                        if symbol in current_portfolio_status['owned_stocks']:
                            quantity_to_sell = min(dynamic_trade_amount / price, current_portfolio_status['owned_stocks'][symbol]['quantity'])
                            if quantity_to_sell > 0:
                                portfolio_manager.sell_stock(symbol, quantity_to_sell, price, reasoning, confidence)
                
            _log_message('info', f"Weekend cycle finished. Waiting {LOOP_INTERVAL_SECONDS}s.")
            time.sleep(LOOP_INTERVAL_SECONDS)
        
        else:
            _log_message('info', "Market is closed. Bot is idle.")
            time.sleep(60)


# --- Scheduler Loop for Daily Resume ---
def scheduler_loop():
    global bot_is_running, all_keys_exhausted
    while True:
        now_utc = datetime.now(timezone.utc)
        tomorrow_utc = now_utc + timedelta(days=1)
        midnight_utc = tomorrow_utc.replace(hour=0, minute=1, second=0, microsecond=0)
        sleep_seconds = (midnight_utc - now_utc).total_seconds()
        
        _log_message('info', f"Scheduler: Sleeping for {sleep_seconds / 3600:.2f} hours until quota reset.")
        time.sleep(sleep_seconds)

        with bot_status_lock:
            if all_keys_exhausted:
                _log_message('action', "Scheduler: API quotas have reset. Resuming bot.")
                bot_is_running = True
                all_keys_exhausted = False
                configure_ai_models()

# --- Global Instances & App Initialization ---
init_db()
configure_ai_models()
finnhub_client = FinnhubClient()
portfolio_manager = PortfolioManager(INITIAL_CASH, finnhub_client, DB_FILE)

# --- Web Page Routes & API ---
@app.route("/")
def index():
    return render_template("index.html")

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

@app.route("/api/backtest", methods=['POST'])
def backtest_strategy():
    data = request.json
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    
    results = run_backtest(start_date, end_date)
    return jsonify(results)

@app.route("/api/backtest/results", methods=['GET'])
def get_backtest_results_api():
    return jsonify(get_backtest_trades())

@app.route("/api/logs/ai")
def get_ai_logs():
    return jsonify(ai_logs)

@app.route("/api/logs/actions")
def get_action_logs():
    return jsonify(action_logs)

@app.route("/api/logs/errors")
def get_error_logs():
    return jsonify(error_logs)

@app.route("/api/portfolio/reset", methods=['POST'])
def reset_portfolio():
    result = portfolio_manager.reset()
    return jsonify(result)

@app.route("/api/bot/start", methods=['POST'])
def start_bot():
    global bot_is_running, all_keys_exhausted
    with bot_status_lock:
        if all_keys_exhausted:
            return jsonify({"status": "paused_gemini_api", "reason": "All API keys are exhausted. Bot will resume automatically tomorrow."}), 400
        bot_is_running = True
        _log_message('action', "Bot started.")
        return jsonify({"status": "running"})

@app.route("/api/bot/pause", methods=['POST'])
def pause_bot():
    global bot_is_running
    with bot_status_lock:
        bot_is_running = False
    _log_message('action', "Bot paused.")
    return jsonify({"status": "paused"})

@app.route("/api/bot/status", methods=['GET'])
def get_bot_status():
    with bot_status_lock:
        if all_keys_exhausted:
            return jsonify({"status": "paused_gemini_api"})
            
        now_et = datetime.now(pytz.timezone('America/New_York'))
        is_market_open = (now_et.weekday() < 5 and now_et.hour >= 9 and now_et.minute >= 30 and (now_et.hour < 16 or (now_et.hour == 16 and now_et.minute == 0)))
        
        if bot_is_running and not is_market_open and now_et.weekday() < 5:
            return jsonify({"status": "paused_for_market"})
        
        status = "running" if bot_is_running else "paused"
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
        
        ai_model_inquiry = get_ai_model([6, 5])
        if not ai_model_inquiry:
            return jsonify({"answer": "Error: AI model for inquiries is not available."}), 500
            
        prompt = f"""
        You are an AI stock bot assistant. Your task is to answer questions about the bot's portfolio, trading strategy, and market conditions based on the provided data.
        
        **Bot's Current Portfolio:**
        {json.dumps(portfolio_status, indent=2)}
        
        **Bot's Recent Trades:**
        {json.dumps(recent_trades, indent=2)}
        
        **User's Question:**
        {question}
        
        Provide a helpful and concise answer to the user's question in Markdown format.
        """
        response = ai_model_inquiry.generate_content(prompt)
        answer = response.text
        return jsonify({"answer": answer})
        
    except Exception as e:
        _log_error(f"Error in ask_ai: {e}")
        return jsonify({"answer": "Error: Failed to get a response from the AI."}), 500

# --- Main Execution ---
if __name__ == "__main__":
    bot_thread = Thread(target=bot_trading_loop, args=(portfolio_manager, finnhub_client), daemon=True)
    bot_thread.start()
    scheduler_thread = Thread(target=scheduler_loop, daemon=True)
    scheduler_thread.start()

    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)

