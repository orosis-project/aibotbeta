AI Stock Trading Bot
This project is an advanced, self-learning AI stock trading bot built with Python and the Flask framework. It is designed to operate autonomously in a simulated environment, making strategic trading decisions based on market data, news analysis, and a history of its own performance.

Features
Autonomous Trading Loop: The bot runs continuously, scanning the market and executing trades without human intervention.

Initial Portfolio Seeding: Upon its first run, the bot automatically purchases 10 random stocks to create a starting portfolio and generate a foundational trade history.

Delayed AI Learning: To prevent overly cautious behavior, the bot operates in a rule-based mode for its first three trades. This "warm-up" period allows it to build a more robust data set before enabling the full AI.

Enhanced Data Analysis: The bot's decision-making process is informed by:

Market News Sentiment: It performs a basic sentiment analysis on recent news headlines to gauge market mood.

Technical Indicators: It calculates a Simple Moving Average (SMA) to identify recent price trends.

Comprehensive Trade History: It learns from a detailed history of its past trades to inform future decisions.

Dynamic Risk Management: The bot adjusts its trade size based on the AI's confidence level. It takes larger, "smart" risks on trades with a high confidence score, aiming for greater payoffs.

Admin Panel: A web-based admin panel provides a real-time view of the bot's status, portfolio value, trade history, and key performance metrics.

Secure API Access: All bot controls (start, pause, reset) are exposed via a secure REST API for integration and remote management.

Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

Prerequisites
Python 3.10 or higher

Poetry (for dependency management)

Finnhub API Key

Gemini API Key

Installation
Clone the repository:

git clone https://github.com/orosis-project/aibotbeta.git
cd aibotbeta

Install dependencies using Poetry:

poetry install

Set up API keys:
Create a file named .env in the project's root directory and add your API keys. Do not commit this file to your repository.

GEMINI_API_KEY="your-gemini-api-key"
FINNHUB_API_KEY="your-finnhub-api-key"

Create the templates directory:
The bot's admin panel requires a templates folder in the root directory.

mkdir templates

Copy the admin panel HTML:
Save the admin_pannel.html code (provided in the previous conversation) into the templates directory.

Running the Application Locally
Start the Flask application using Poetry:

poetry run python app.py

Your bot should now be running. You can access the admin panel at http://127.0.0.1:8000/pannel.

Deployment to Render
This project is configured for seamless deployment on Render. The following are the critical commands you need to set in the Render dashboard for a successful deployment.

Build Command:

poetry install

Start Command:

gunicorn --workers 4 --bind 0.0.0.0:$PORT app:app

Note: You must also set your GEMINI_API_KEY and FINNHUB_API_KEY as environment variables directly in the Render dashboard for the application to function correctly.

API Endpoints
GET /: Returns a simple message to confirm the backend is running.

GET /pannel: Serves the HTML for the admin panel.

GET /api/portfolio: Returns the current portfolio status in JSON format.

GET /api/portfolio/summary: Returns a detailed summary of the portfolio and trade history for the admin panel.

GET /api/trades: Returns a list of the most recent trades.

POST /api/bot/start: Starts the trading bot loop.

POST /api/bot/pause: Pauses the trading bot loop.

GET /api/bot/status: Returns the current status of the bot (running or paused).

POST /api/portfolio/reset: Resets the portfolio and database.

POST /api/ask: Sends a question to the AI and returns a response.

How It Works
The core of the bot's logic is a continuous trading loop that runs on a separate thread. This loop fetches real-time market data, company news, and historical data. It then constructs a detailed prompt for the Gemini AI, providing context about its current portfolio, past performance, and market indicators. The AI responds with a recommended trading action (BUY, SELL, or HOLD), and a confidence score.

Based on the confidence and the bot's current state (e.g., in its initial "warm-up" phase), a trade may be executed. This creates a feedback loop where the bot constantly learns from its own actions to improve its future decisions.

Future Improvements
Advanced AI Learning: Implement reinforcement learning from human feedback (RLHF) to allow user questions and suggestions to directly influence the bot's strategic learning.

Stop-Loss and Take-Profit Orders: Add automated risk management features to place stop-loss and take-profit orders for every new trade.

Multi-Strategy Trading: Allow the bot to manage multiple, distinct trading strategies simultaneously and allocate capital dynamically between them based on performance.
