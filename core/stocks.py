from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import yfinance as yf

# -------------------------
# Stock Info Schema
# -------------------------
class StockInfo(BaseModel):
    ticker: str
    name: str
    current_price: float
    expected_return_1yr: Optional[float] = None  # historical 1-year return %
    risk_score: Optional[float] = None  # basic volatility measure


class StockRecommendation(BaseModel):
    asset_class: str
    suggested_stocks: List[StockInfo]


class PortfolioStockRecommendation(BaseModel):
    portfolio: Dict[str, float]  # asset allocation amounts
    recommendations: List[StockRecommendation]


# -------------------------
# Predefined stocks per asset class (can expand)
# -------------------------
ASSET_CLASS_STOCKS = {
    "Stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
    "Bonds": ["BND", "AGG", "TLT"],  # ETFs
    "Gold": ["GLD", "IAU"],
    "Crypto": ["BTC-USD", "ETH-USD"],
    "RealEstate": ["VNQ", "SCHH"]
}


# -------------------------
# Helper: Fetch stock info from Yahoo Finance
# -------------------------
def fetch_stock_info(ticker: str) -> StockInfo:
    data = yf.Ticker(ticker)
    price = data.history(period="1d")["Close"][-1]
    
    # Calculate simple 1-year historical return
    hist = data.history(period="1y")["Close"]
    if len(hist) > 1:
        expected_return = (hist[-1] - hist[0]) / hist[0] * 100
        risk_score = hist.pct_change().std() * 100  # simple volatility %
    else:
        expected_return = None
        risk_score = None
    
    return StockInfo(
        ticker=ticker,
        name=data.info.get("longName", ticker),
        current_price=round(price, 2),
        expected_return_1yr=round(expected_return, 2) if expected_return else None,
        risk_score=round(risk_score, 2) if risk_score else None
    )


# -------------------------
# Main Recommendation Function
# -------------------------
def recommend_stocks(portfolio_allocation: Dict[str, float]) -> PortfolioStockRecommendation:
    recommendations = []
    for asset_class, amount in portfolio_allocation.items():
        tickers = ASSET_CLASS_STOCKS.get(asset_class, [])
        stocks = [fetch_stock_info(t) for t in tickers]
        recommendations.append(StockRecommendation(
            asset_class=asset_class,
            suggested_stocks=stocks
        ))

    return PortfolioStockRecommendation(
        portfolio=portfolio_allocation,
        recommendations=recommendations
    )


# -------------------------
# Example Usage
# -------------------------
if __name__ == "__main__":
    sample_portfolio = {
        "Stocks": 5500,
        "Bonds": 2000,
        "Gold": 1000,
        "RealEstate": 500,
        "Crypto": 2000,
        "Cash": 0
    }

    result = recommend_stocks(sample_portfolio)
    print(result.model_dump())
