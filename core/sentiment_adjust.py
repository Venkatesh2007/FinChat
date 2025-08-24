# sentiment_adjust.py
from typing import Dict, List
from pydantic import BaseModel, Field
import numpy as np
from core.llm import get_llm  # 👈 Import your LLM loader
import os
from langchain.schema import HumanMessage, SystemMessage
import re
from core.portfolio import allocate_portfolio
from core.userInfo import UserProfile
from db.newsdb import get_latest_news




# -----------------------------
# Pydantic schema
# -----------------------------
class AdjustmentResult(BaseModel):
    adjusted_allocation: Dict[str, float] = Field(..., description="Adjusted portfolio weights")
    sentiment_summary: str = Field(..., description="Summary of sentiment reasoning")



# -----------------------------
# HuggingFace FinBERT Setup
# -----------------------------
# def load_finbert():
#     """
#     Load FinBERT using HuggingFaceEndpoint for text-classification.
#     Requires: HUGGINGFACEHUB_API_TOKEN in env.
#     """
#     return pipeline("text-classification", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert")


# -----------------------------
# Aggregate sentiment
# -----------------------------
# def analyze_sentiment(asset: str, headlines: List[str], sentiment_pipeline):
#     scores = []
#     summary = []

#     for h in headlines:
#         res = sentiment_pipeline(h)[0]  # returns [{'label': 'Positive', 'score': 0.98}]
#         label = res["label"]
#         score = res["score"]

#         summary.append(f"{h} → {label} ({score:.2f})")

#         if label == "Positive":
#             scores.append(score)
#         elif label == "Negative":
#             scores.append(-score)
#         else:
#             scores.append(0.0)

#     avg_score = float(np.mean(scores)) if scores else 0.0
#     return avg_score, "\n".join(summary)

# -----------------------------
# Load LLM
# -----------------------------
def load_llm_sentiment():
    """Load the LLM defined in llm.py"""
    return get_llm()


# -----------------------------
# Use LLM to analyze sentiment
# -----------------------------
def analyze_sentiment(asset: str, headlines: list[str], sentiment_llm):
    scores = []
    summary = []

    system_prompt = """You are a financial sentiment classifier.
For each news headline, respond with one of:
- Positive (score between 0.6 and 1.0)
- Negative (score between 0.6 and 1.0)
- Neutral (score exactly 0.0)

Respond strictly in JSON like:
{"label": "Positive", "score": 0.87}
"""

    for h in headlines:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Headline: {h}")
        ]

        raw = sentiment_llm.invoke(messages).content.strip()

        # Extract JSON (basic cleanup)
        match = re.search(r'{.*}', raw, re.DOTALL)
        if not match:
            label, score = "Neutral", 0.0
        else:
            try:
                data = eval(match.group())  # ⚠️ for production replace with json.loads
                label = data.get("label", "Neutral")
                score = float(data.get("score", 0.0))
            except Exception:
                label, score = "Neutral", 0.0

        summary.append(f"{h} → {label} ({score:.2f})")

        if label == "Positive":
            scores.append(score)
        elif label == "Negative":
            scores.append(-score)
        else:
            scores.append(0.0)

    avg_score = float(np.mean(scores)) if scores else 0.0
    return avg_score, "\n".join(summary)



# -----------------------------
# Adjust Portfolio
# -----------------------------
def adjust_portfolio(profile: UserProfile, base_alloc: Dict[str, float]) -> AdjustmentResult:
    sentiment_model = load_llm_sentiment()

    adjusted = base_alloc.copy()
    full_summary = []

    for asset in ["Stocks", "Gold", "Crypto","RealEstate"]:
        news = get_latest_news(asset, limit=3)
        if not news:
            continue

        avg_score, details = analyze_sentiment(asset, news, sentiment_model)
        full_summary.append(f"### {asset}\n{details}")

        if avg_score > 0.2:   # bullish
            adjusted[asset] = round(adjusted.get(asset, 0.0) + 0.05, 2)
        elif avg_score < -0.2:  # bearish
            adjusted[asset] = round(max(0.0, adjusted.get(asset, 0.0) - 0.05), 2)

    total = sum(adjusted.values()) or 1.0
    adjusted = {k: round(v / total, 2) for k, v in adjusted.items()}

    return AdjustmentResult(adjusted_allocation=adjusted, sentiment_summary="\n\n".join(full_summary))


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    from userInfo import extract_user_profile

    q = "I’m 30, earning 80,000, aggressive, building long-term wealth"
    profile = extract_user_profile(q)
    base_alloc = allocate_portfolio(profile)
    result = adjust_portfolio(profile, base_alloc)

    print("Base:", base_alloc)
    print("Adjusted:", result.adjusted_allocation)
    print("Sentiment reasoning:\n", result.sentiment_summary)
