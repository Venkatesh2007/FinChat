# db.py
import sqlite3
import requests
import datetime
from typing import List, Dict
from langdetect import detect
import os
from dotenv import load_dotenv

load_dotenv()

DB_FILE = "market_data.db"
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_KEY")

# -----------------------------
# DB Init
# -----------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS news (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset TEXT,
                    headline TEXT,
                    published_at TEXT
                )""")
    c.execute("""CREATE TABLE IF NOT EXISTS prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset TEXT,
                    date TEXT,
                    close REAL
                )""")
    conn.commit()
    conn.close()


# -----------------------------
# Fetch & Store News
# -----------------------------
def fetch_and_store_news(asset: str, api_key: str):
    """
    Fetch financial news for an asset using NewsAPI (example).
    Store into SQLite.
    """
    url = f"https://newsapi.org/v2/everything?q={asset}&sortBy=publishedAt&apiKey={api_key}"
    resp = requests.get(url).json()
    articles = resp.get("articles", [])

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    for art in articles[:20]:
        headline = art["title"].strip()
        published = art["publishedAt"]

        # Language filter
        try:
            if detect(headline) != "en":
                continue
        except:
            continue
        c.execute("INSERT INTO news (asset, headline, published_at) VALUES (?, ?, ?)",
                (asset, art["title"], art["publishedAt"]))
    conn.commit()
    conn.close()


# -----------------------------
# Fetch & Store Prices
# -----------------------------
def fetch_and_store_prices(asset: str, symbol: str, api_key: str):
    """
    Fetch daily prices using AlphaVantage API.
    """
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}"
    resp = requests.get(url).json()
    time_series = resp.get("Time Series (Daily)", {})

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    for date, values in list(time_series.items())[:90]:  # store last 90 days
        close_price = float(values["4. close"])
        c.execute("INSERT INTO prices (asset, date, close) VALUES (?, ?, ?)",
                (asset, date, close_price))

    conn.commit()
    conn.close()


# -----------------------------
# Query Functions
# -----------------------------
def get_latest_news(asset: str, limit=5) -> List[str]:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT headline FROM news WHERE asset=? ORDER BY published_at DESC LIMIT ?", (asset, limit))
    rows = c.fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_latest_prices(asset: str, limit=30) -> List[Dict]:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT date, close FROM prices WHERE asset=? ORDER BY date DESC LIMIT ?", (asset, limit))
    rows = c.fetchall()
    conn.close()
    return [{"date": r[0], "close": r[1]} for r in rows]


# -----------------------------
# Example run
# -----------------------------
if __name__ == "__main__":
    init_db()
    # Replace with valid API keys
    fetch_and_store_news("Cash", api_key=NEWSAPI_KEY)
    fetch_and_store_prices("Cash", "GC=F", api_key=ALPHAVANTAGE_KEY)

    print(get_latest_news("Cash"))
