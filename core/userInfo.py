# userInfo.py
from typing import Optional, Literal
from pydantic import BaseModel, Field, ValidationError
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser
import re
import os
from dotenv import load_dotenv
# from core.llm import get_llm  
from llm import get_llm  


load_dotenv()


# -----------------------------
# Pydantic schema for user info
# -----------------------------
class UserProfile(BaseModel):
    age: Optional[int] = Field(None, description="Age of the user in years")
    monthly_income: Optional[float] = Field(None, description="Monthly income in INR")
    risk_tolerance: Optional[Literal["Conservative", "Moderate", "Aggressive"]] = Field(
        None, description="Risk tolerance category"
    )
    investment_goal: Optional[str] = Field(
        None, description="User's financial goal (retirement, wealth-building, short-term, etc.)"
    )
    investment_horizon_years: Optional[int] = Field(
        None, description="Investment horizon in years"
    )


# -----------------------------
# LLM setup (Gemini)
# -----------------------------
llm = get_llm()
parser = PydanticOutputParser(pydantic_object=UserProfile)


# -----------------------------
# Extract structured info
# -----------------------------
def extract_user_profile(user_query: str) -> UserProfile:
    """
    Extract structured user profile attributes from free-text query.
    """
    system_prompt = (
        "You are a financial assistant that extracts structured user profile information "
        "from free-text queries. Always output valid JSON ONLY that matches this schema:\n"
        "{\n"
        '  "age": int | null,\n'
        '  "monthly_income": float | null,\n'
        '  "risk_tolerance": "Conservative" | "Moderate" | "Aggressive" | null,\n'
        '  "investment_goal": string | null,\n'
        '  "investment_horizon_years": int | null\n'
        "}\n\n"
        "If info is missing, set it as null."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User query: {user_query}")
    ]

    try:
        raw_response = llm.invoke(messages)
        raw_text = raw_response.content.strip()

        # # Strip ```json fences if present
        # match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        # cleaned = match.group(0) if match else raw_text

        # profile = UserProfile.model_validate_json(cleaned)
        profile: UserProfile = parser.parse(raw_response.content)
        return profile

    except ValidationError as ve:
        return UserProfile(
            age=None,
            monthly_income=None,
            risk_tolerance=None,
            investment_goal=None,
            investment_horizon_years=None
        )
    except Exception:
        return UserProfile()


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    queries = [
        "I’m 20, earning 5000, what should I do?",
        "I’m 45, conservative, saving for retirement in 15 years.",
        "I want to aggressively invest my 1 lakh salary for short-term gains.",
        "I’m 18 years old and I earn 1000 rupees per month. I want to buy a car after 10 years.",
    ]

    for q in queries:
        print(f"Query: {q}")
        profile = extract_user_profile(q)
        print(profile.model_dump())
        print("-" * 40)
