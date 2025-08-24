# response.py
from core.userInfo import UserProfile
from core.sentiment_adjust import AdjustmentResult
from typing import Dict
from core.llm import get_llm
from langchain.schema import HumanMessage, SystemMessage
import textwrap

# -----------------------------
# LLM Setup
# -----------------------------
llm = get_llm()

# -----------------------------
# Professional Goal-Oriented Financial Advice
# -----------------------------
def generate_final_response(profile: UserProfile, base_alloc: Dict[str, float], adjusted: AdjustmentResult) -> str:
    """
    Generate full, professional, actionable financial advice to achieve the user's goal.
    Combines user profile, base & adjusted allocations, and news-driven sentiment reasoning.
    Highlights key allocations with emojis and provides warnings/tips.
    """
    system_prompt = textwrap.dedent("""
        You are a senior financial advisor. Your task is to provide a complete investment plan
        tailored to the user's goal. Consider the following:
        - User profile: age, income, risk tolerance, investment goal, horizon
        - Base portfolio allocation
        - Adjusted portfolio after news sentiment
        Provide:
        1. Clear summary of user's financial situation
        2. Explanation of base allocation with key allocations highlighted using emojis:
           - Bonds: 🔒 Safe
           - Cash: 💰 Liquidity
           - Stocks: 📈 Growth
           - Gold: 🏅 Inflation hedge
           - Crypto: 🚀 High risk
        3. How adjustments were made based on market news
        4. Detailed, actionable recommendations including warnings/tips:
           - Avoid putting too much in volatile assets at this stage
           - Rebalance annually or when major news affects the market
        5. Make it professional, empathetic, and fully actionable.
        6. Do not suggest consulting another financial advisor; act as the user's advisor.
        Limit output to ~250-300 words.
    """)

    user_context = textwrap.dedent(f"""
        User profile: {profile.dict()}
        Base portfolio allocation: {base_alloc}
        Adjusted portfolio allocation: {adjusted.adjusted_allocation}
        Reasoning from market sentiment/news:
        {adjusted.sentiment_summary}
    """)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_context)
    ]

    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        return f"[Error generating final response: {e}]"

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    from core.userInfo import extract_user_profile
    from core.portfolio import allocate_portfolio
    from core.sentiment_adjust import adjust_portfolio

    query = "I’m 25, earning 50,000 INR monthly, risk moderate, goal: buy a house in 10 years"
    profile = extract_user_profile(query)
    base_alloc = allocate_portfolio(profile)
    adjusted = adjust_portfolio(profile, base_alloc)

    advice = generate_final_response(profile, base_alloc, adjusted)
    print(advice)
