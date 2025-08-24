import re
import time
import json
import urllib.request
import yfinance as yf
from urllib.error import HTTPError

def clean_input(user_input: str) -> str:
    """Remove $ and other non-alphanumeric characters"""
    cleaned = user_input.upper().strip()
    cleaned = re.sub(r'[^A-Z0-9 .&]', '', cleaned)
    return cleaned

def lookup_ticker_safe(company_name: str, max_retries=3) -> str:
    """Search Yahoo Finance API safely with retry"""
    company_name_clean = clean_input(company_name)
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={company_name_clean}"
    
    for attempt in range(max_retries):
        try:
            response = urllib.request.urlopen(url)
            data = json.loads(response.read())
            quotes = data.get("quotes", [])
            if not quotes:
                print(f"No ticker found for '{company_name}'")
                return None
            # Prefer EQUITY type
            for q in quotes:
                if q.get("quoteType") == "EQUITY":
                    return q["symbol"]
            return quotes[0]["symbol"]
        except HTTPError as e:
            if e.code == 429:  # Too Many Requests
                print("Rate limit hit. Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"HTTP Error: {e}")
                break
        except Exception as e:
            print(f"Lookup failed: {e}")
            break
    return None

def fetch_stock_safe(company_input: str):
    ticker = lookup_ticker_safe(company_input)
    if not ticker:
        print(f"Cannot fetch stock: No ticker found for {company_input}")
        return None

    ticker_obj = yf.Ticker(ticker)
    try:
        hist = ticker_obj.history(period="1y")["Close"]
        if len(hist) == 0:
            print(f"No historical price data for ticker {ticker}")
            return None

        current_price = hist[-1]
        expected_return = (hist[-1]-hist[0])/hist[0]*100 if len(hist)>1 else None
        risk_score = hist.pct_change().std()*100 if len(hist)>1 else None

        return {
            "ticker": ticker,
            "current_price": round(current_price,2),
            "expected_return_1yr": round(expected_return,2) if expected_return else None,
            "risk_score": round(risk_score,2) if risk_score else None
        }
    except Exception as e:
        print(f"Stock fetch failed: {e}")
        return None

# -------------------------
# Example
# -------------------------
if __name__ == "__main__":
    company = "Apple"
    stock_data = fetch_stock_safe(company)
    print(stock_data)
