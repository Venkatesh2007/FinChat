# intent.py
# Module for classifying user queries into intents:
# Portfolio_Allocation | Investment_Prediction | Knowledge
# Uses LangChain with Gemini/Groq models + Pydantic validation.

from typing import Literal
from pydantic import BaseModel, Field, ValidationError
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser
import re
import json
import os
from dotenv import load_dotenv
from core.llm import get_llm  

load_dotenv()

# -----------------------------
# Pydantic schema for validation
# -----------------------------
class IntentSchema(BaseModel):
    intent: Literal["Portfolio_Allocation", "Investment_Prediction", "Knowledge", "General_Chat"] = Field(
        ..., description="The classified intent of the user query."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0 and 1."
    )
    rationale: str = Field(
        ..., description="Short reasoning for why this intent was chosen."
    )


# -----------------------------
# LLM Config (Gemini or Groq)
# -----------------------------
# Option A: Gemini via LangChain
llm = get_llm()
parser = PydanticOutputParser(pydantic_object=IntentSchema)


# Option B: If you want Groq Llama 3:
# from langchain_groq import ChatGroq
# llm = ChatGroq(model="llama3-8b-8192", groq_api_key="YOUR_GROQ_API_KEY", temperature=0.0)

# -----------------------------
# Helper: Clean JSON from LLM output
# -----------------------------
def extract_json(text: str) -> str:
    """
    Extracts the first JSON object found in text (removes ```json fences).
    """
    try:
        # If wrapped in ```json ... ```
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)
        return text
    except Exception:
        return text


# -----------------------------
# Intent Detection Function
# -----------------------------
def detect_intent(user_query: str) -> IntentSchema:
    """
    Classify a user query into one of the predefined intents.
    Validates response against Pydantic schema.
    """
    system_prompt = (
        "You are an intent classifier for a financial advisor chatbot. "
        "You MUST respond strictly in valid JSON that fits this schema:\n\n"
        "{\n"
        "  'intent': 'Portfolio_Allocation' | 'Investment_Prediction' | 'Knowledge' | 'General_Chat',\n"
        "  'confidence': float between 0 and 1,\n"
        "  'rationale': short string explanation\n"
        "}\n\n"
        "Rules:\n"
        "- Portfolio_Allocation → User asks about distributing money, allocating assets, retirement planning, etc.\n"
        "- Investment_Prediction → User asks about future value, trends, 'next month', 'next year', or market forecast.\n"
        "- Knowledge → User asks for definitions, explanations, or general financial literacy concepts.\n"
        "- General_Chat → Greetings, small talk, thanks, jokes, or queries unrelated to finance.\n"
        "Output ONLY JSON. No extra text."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User query: {user_query}")
    ]

    try:
        raw_response = llm.invoke(messages)
        raw_text = raw_response.content.strip()
        # cleaned = extract_json(raw_text)


        # Try validation with Pydantic
        # intent_obj = IntentSchema.model_validate_json(raw_text)
        parsed: IntentSchema = parser.parse(raw_response.content)

        return parsed

    except ValidationError as ve:
        # If LLM output invalid, fallback to default
        return IntentSchema(
            intent="Knowledge",
            confidence=0.5,
            rationale=f"Fallback due to validation error: {ve}"
        )
    except Exception as e:
        return IntentSchema(
            intent="Knowledge",
            confidence=0.3,
            rationale=f"Error calling LLM: {e}"
        )


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    test_queries = [
        "I’m 20, I earn 5000. How should I invest?",
        "I want to invest in tesla is it a good choice or not?",
        "Can you explain what SIP means?",
        "what is happening with gold today?"
        "hello there!",
        "thanks a lot for your help",
        "how are you?"
    ]

    for q in test_queries:
        result = detect_intent(q)
        print(f"Query: {q}")
        print(result.model_dump())
        print("-" * 40)
