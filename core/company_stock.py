from pydantic import BaseModel
from typing import Optional, List
from yahooquery import Ticker, search
import yfinance as yf
import re
import pandas as pd
import numpy as np

# -------------------------
# Pydantic Schemas
# -------------------------
class CompanyStockInfo(BaseModel):
    ticker: str
    company_name: str
    current_price: Optional[float] = None
    previous_close: Optional[float] = None
    open_price: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    market_cap: Optional[float] = None
    volume: Optional[int] = None
    historical_prices: Optional[List[float]] = None


class StockPrediction(BaseModel):
    ticker: str
    current_price: float
    expected_price: float
    lower_bound_5pct: float
    upper_bound_95pct: float


# -------------------------
# Helper Functions
# -------------------------
def clean_input(user_input: str) -> str:
    """Remove $ and non-alphanumeric characters"""
    cleaned = user_input.upper().strip()
    cleaned = re.sub(r'[^A-Z0-9 .&]', '', cleaned)
    return cleaned


def get_ticker_yahooquery(company_name: str) -> Optional[str]:
    cleaned_name = clean_input(company_name)
    results = search(cleaned_name)
    quotes = results.get('quotes', [])
    if not quotes:
        return None
    for q in quotes:
        if q.get('quoteType') == 'EQUITY':
            return q['symbol']
    return quotes[0]['symbol']  # fallback


def fetch_company_stock(company_input: str) -> Optional[CompanyStockInfo]:
    ticker_symbol = get_ticker_yahooquery(company_input)
    if not ticker_symbol:
        print(f"Ticker not found for '{company_input}'")
        return None

    # Yahooquery for current data
    t = Ticker(ticker_symbol)
    try:
        price_data = t.price.get(ticker_symbol, {})

        # YFinance for historical prices
        yf_ticker = yf.Ticker(ticker_symbol)
        hist = yf_ticker.history(period="1y")["Close"] if not yf_ticker.history(period="1y").empty else pd.Series()
        historical_prices = hist.tolist() if len(hist) > 0 else None

        return CompanyStockInfo(
            ticker=ticker_symbol.upper(),
            company_name=price_data.get("longName", ticker_symbol),
            current_price=price_data.get("regularMarketPrice"),
            previous_close=price_data.get("regularMarketPreviousClose"),
            open_price=price_data.get("regularMarketOpen"),
            day_high=price_data.get("regularMarketDayHigh"),
            day_low=price_data.get("regularMarketDayLow"),
            market_cap=price_data.get("marketCap"),
            volume=price_data.get("regularMarketVolume"),
            historical_prices=[round(p, 2) for p in historical_prices] if historical_prices else None
        )

    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None


# -------------------------
# Monte Carlo Simulation
# -------------------------
def monte_carlo_simulation(S0, mu, sigma, days=252, simulations=1000):
    """
    Simulate future stock prices using Geometric Brownian Motion.
    """
    dt = 1  # 1 day
    price_matrix = np.zeros((days, simulations))
    
    for sim in range(simulations):
        prices = [S0]
        for t in range(1, days):
            St = prices[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.random.normal())
            prices.append(St)
        price_matrix[:, sim] = prices
    
    return price_matrix


def predict_future_stock(stock_info: CompanyStockInfo, days: int = 252, simulations: int = 1000) -> Optional[StockPrediction]:
    if not stock_info.historical_prices or len(stock_info.historical_prices) < 2:
        return None

    hist_series = pd.Series(stock_info.historical_prices)
    daily_returns = hist_series.pct_change().dropna()
    mu = daily_returns.mean()
    sigma = daily_returns.std()
    S0 = hist_series.iloc[-1]

    simulated_prices = monte_carlo_simulation(S0, mu, sigma, days, simulations)
    expected_price = simulated_prices[-1, :].mean()
    lower_bound = np.percentile(simulated_prices[-1, :], 5)
    upper_bound = np.percentile(simulated_prices[-1, :], 95)

    return StockPrediction(
        ticker=stock_info.ticker,
        current_price=S0,
        expected_price=round(expected_price, 2),
        lower_bound_5pct=round(lower_bound, 2),
        upper_bound_95pct=round(upper_bound, 2)
    )


# -------------------------
# Example Usage
# -------------------------
if __name__ == "__main__":
    companies = ["Apple", "Tesla", "Microsoft"]
    for comp in companies:
        stock_info = fetch_company_stock(comp)
        if stock_info:
            prediction = predict_future_stock(stock_info)
            print(stock_info.model_dump())
            print(prediction.model_dump())
        else:
            print(f"No data available for {comp}")
