from pydantic import BaseModel
from typing import Optional, Literal, Dict
# from portfolio import allocate_portfolio
# from llm import get_llm  
# from userInfo import extract_user_profile
from core.portfolio import allocate_portfolio
from core.llm import get_llm  
from core.userInfo import extract_user_profile

# -------------------------
# User Profile Schema
# -------------------------
class UserProfile(BaseModel):
    age: Optional[int]
    monthly_income: Optional[float]
    risk_tolerance: Optional[Literal["Conservative", "Moderate", "Aggressive"]]
    investment_goal: Optional[str]
    investment_horizon_years: Optional[int]
    investment_amount: Optional[float]  # will be suggested by LLM


# -------------------------
# Output Schema
# -------------------------
class PortfolioOutput(BaseModel):
    total_investment: float
    allocation_percentages: Dict[str, float]
    allocation_amounts: Dict[str, float]
    user_profile: UserProfile


# -------------------------
# Helper: Ask LLM for investment amount
# -------------------------
def suggest_investment_amount(query: str, user_profile: UserProfile) -> float:
    """
    Use LLM to suggest a reasonable investment amount based on user's profile.
    """
    llm = get_llm()
    prompt = f"""
    The user has the following profile:
    Age: {user_profile.age}
    Monthly Income: {user_profile.monthly_income}
    Goal: {user_profile.investment_goal}
    Horizon: {user_profile.investment_horizon_years} years
    
    Suggest a beginner-friendly investment amount in INR they can invest as a lump sum
    or monthly, considering their income and goal. Only return a numeric value.
    """
    response = llm.invoke([{"role": "user", "content": prompt}])
    try:
        amount = float("".join(filter(lambda c: c.isdigit() or c=='.', response.content)))
    except:
        amount = 1000  # fallback default
    return amount


# -------------------------
# Main function
# -------------------------
def generate_portfolio(query: str) -> PortfolioOutput:
    # Step 1: Extract base profile (age, income, goal) from query
    profile_raw = extract_user_profile(query)

    # Step 2: Use LLM to suggest total investment
    investment_amount = suggest_investment_amount(query, profile_raw)
    profile = UserProfile(**profile_raw.model_dump(), investment_amount=investment_amount)

    # Step 3: Allocate portfolio
    allocation_percentages = allocate_portfolio(profile)
    allocation_amounts = {asset: round(profile.investment_amount * pct, 2)
                        for asset, pct in allocation_percentages.items()}

    # Step 4: Return structured Pydantic output
    return PortfolioOutput(
        total_investment=profile.investment_amount,
        allocation_percentages=allocation_percentages,
        allocation_amounts=allocation_amounts,
        user_profile=profile
    )


# -------------------------
# Example Usage
# -------------------------
if __name__ == "__main__":
    user_query = "I am 25 years old, earn 50,000 INR/month, and want to save for buying a house in 5 years."
    result = generate_portfolio(user_query)
    print(result.model_dump())
