# portfolio.py
from typing import Dict
# from core.userInfo import UserProfile
from userInfo import UserProfile


def allocate_portfolio(profile: UserProfile) -> Dict[str, float]:
    """
    Advanced portfolio allocation logic.
    Returns dict of asset classes with allocation percentages.
    """

    # Default conservative allocation (fallback if info missing)
    allocation = {
        "Stocks": 0.3,
        "Bonds": 0.4,
        "Gold": 0.1,
        "RealEstate": 0.1,
        "Crypto": 0.05,
        "Cash": 0.05
    }

    # -----------------------------
    # Step 1: Base on age
    # -----------------------------
    if profile.age:
        if profile.age < 30:
            allocation.update({"Stocks": 0.55, "Bonds": 0.2, "Gold": 0.1, "Crypto": 0.1, "Cash": 0.05})
        elif profile.age < 50:
            allocation.update({"Stocks": 0.45, "Bonds": 0.3, "Gold": 0.15, "RealEstate": 0.05, "Cash": 0.05})
        else:
            allocation.update({"Stocks": 0.25, "Bonds": 0.45, "Gold": 0.15, "RealEstate": 0.1, "Cash": 0.05})

    # -----------------------------
    # Step 2: Adjust by risk tolerance
    # -----------------------------
    if profile.risk_tolerance == "Aggressive":
        allocation["Stocks"] += 0.15
        allocation["Crypto"] += 0.05
        allocation["Bonds"] -= 0.1
        allocation["Cash"] -= 0.05
    elif profile.risk_tolerance == "Conservative":
        allocation["Stocks"] -= 0.15
        allocation["Bonds"] += 0.1
        allocation["Cash"] += 0.05

    # -----------------------------
    # Step 3: Adjust by income
    # -----------------------------
    if profile.monthly_income:
        if profile.monthly_income < 30000:  # low income → safer
            allocation["Cash"] += 0.1
            allocation["Stocks"] -= 0.05
            allocation["Crypto"] = 0.0
        elif profile.monthly_income > 100000:  # high income → afford risk
            allocation["Stocks"] += 0.1
            allocation["Crypto"] += 0.05

    # -----------------------------
    # Step 4: Adjust by goals
    # -----------------------------
    if profile.investment_goal:
        goal = profile.investment_goal.lower()
        if "retirement" in goal:
            allocation.update({"Bonds": 0.5, "Stocks": 0.25, "Gold": 0.15, "Cash": 0.1})
        elif "short" in goal or "short-term" in goal:
            allocation.update({"Cash": 0.4, "Bonds": 0.3, "Stocks": 0.2, "Gold": 0.1, "Crypto": 0.0})
        elif "wealth" in goal or "growth" in goal:
            allocation.update({"Stocks": 0.6, "Crypto": 0.1, "Bonds": 0.15, "Gold": 0.1, "Cash": 0.05})

    # -----------------------------
    # Normalize to sum = 1
    # -----------------------------
    total = sum(allocation.values())
    allocation = {k: round(v / total, 2) for k, v in allocation.items()}

    return allocation


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    from userInfo import extract_user_profile

    query = "I’m 18 years old and I earn 1000 rupees per month. I want to buy a car after 10 years."
    profile = extract_user_profile(query)
    alloc = allocate_portfolio(profile)
    print("User Profile:", profile.model_dump())
    print("Portfolio Allocation:", alloc)
