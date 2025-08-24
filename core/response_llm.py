from typing import List, Dict, Any
from pydantic import BaseModel
from core.llm import get_llm
import json


# Define structured advice format
class FinancialAdvice(BaseModel):
    summary: str
    trustworthiness: str
    decision_validation: str
    investment_plan: str
    risk_analysis: str
    expected_returns: str
    step_by_step: List[str]
    sources: List[str]


def generate_financial_advice(
    user_query: str,
    user_profile: Dict[str, Any],
    stock_data: Dict[str, Any] = None,
    monte_carlo: Dict[str, Any] = None,
    portfolio: Dict[str, Any] = None
) -> FinancialAdvice:
    """
    Generate advanced, humanized financial advice using LLM reasoning
    by combining user query, profile, stock data, portfolio recommendations,
    and Monte Carlo simulations. Provides step-by-step actionable insights
    with expected returns and sources.
    """

    llm = get_llm()

    # Build advanced advisor prompt
    prompt = f"""
    You are acting as a **real-time advanced financial advisor**.
    Analyze everything and provide a complete, reliable, and humanized
    financial recommendation.

    ## User Query
    {user_query}

    ## User Profile
    {user_profile}

    ## Portfolio Recommendation
    {portfolio if portfolio else "Not provided"}

    ## Stock Data
    {stock_data if stock_data else "Not provided"}

    ## Monte Carlo Simulation (future risk/returns)
    {monte_carlo if monte_carlo else "Not provided"}

    ### Instructions:
    1. Start with a **clear summary** of the user’s situation and query.
    2. Provide a **decision validation** (is the user’s query/idea financially sound?).
    3. Give a **trustworthiness rating** and explain why the advice is reliable (based on diversification, SIPs, inflation, market history, etc).
    4. Suggest a **detailed investment plan**: asset classes, percentages, timelines (short-term vs long-term).
    5. Provide a **risk analysis**: best-case, average-case, worst-case scenarios.
    6. Give **expected returns** (with numbers & explanation).
    7. Create a **step-by-step actionable roadmap** for the user (from today to future).
    8. Include **sources/references** (e.g. market history, financial principles, or links if known).
    9. Format response strictly in JSON with fields:
    {{
        "summary": "...",
        "decision_validation": "...",
        "trustworthiness": "...",
        "investment_plan": "...",
        "risk_analysis": "...",
        "expected_returns": "...",
        "step_by_step": ["...", "..."],
        "sources": ["...", "..."]
    }}
    """

    # Get LLM response
    raw_response = llm.invoke(prompt)

    if hasattr(raw_response, "content"):
        response_text = raw_response.content
    else:
        response_text = str(raw_response)

    # Parse JSON safely
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        parsed = {
            "summary": "Unable to parse advice.",
            "decision_validation": "Could not verify user query.",
            "trustworthiness": "Limited reliability.",
            "investment_plan": "Fallback conservative plan: 60% equities, 30% bonds, 10% cash.",
            "risk_analysis": "No risk evaluation available.",
            "expected_returns": "Unknown.",
            "step_by_step": [response_text],
            "sources": ["LLM response without proper JSON format."]
        }

    # Ensure proper formatting
    clean_steps = [str(step) for step in parsed.get("step_by_step", [])]
    clean_sources = [str(src) for src in parsed.get("sources", [])]

    return FinancialAdvice(
        summary=str(parsed.get("summary", "")),
        decision_validation=str(parsed.get("decision_validation", "")),
        trustworthiness=str(parsed.get("trustworthiness", "")),
        investment_plan=str(parsed.get("investment_plan", "")),
        risk_analysis=str(parsed.get("risk_analysis", "")),
        expected_returns=str(parsed.get("expected_returns", "")),
        step_by_step=clean_steps,
        sources=clean_sources
    )
