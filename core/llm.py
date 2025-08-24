# llm.py
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    """
    Returns a Groq LLM instance (gemma2-9b-it).
    Requires GROQ_API_KEY to be set in environment.
    """
    return ChatGroq(
        model="openai/gpt-oss-20b",       # gemma2-9b-it hosted on Groq
        temperature=0.0,                   # Deterministic output (good for reasoning & classification)
        api_key=os.getenv("GROQ_API_KEY"),
    )