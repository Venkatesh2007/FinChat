# llm.py
import os
from langchain_groq import ChatGroq


def get_llm():
    """
    Returns a Groq LLM instance (gemma2-9b-it).
    Requires GROQ_API_KEY to be set in environment.
    """
    return ChatGroq(
        model="gemma2-9b-it",       # gemma2-9b-it hosted on Groq
        temperature=0.0,                   # Deterministic output (good for reasoning & classification)
        api_key=os.getenv("GROQ_API_KEY"),
    )
