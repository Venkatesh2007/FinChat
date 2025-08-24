import json
# from llm import get_llm
from core.llm import get_llm

def decide_and_execute(user_query: str):
    llm = get_llm()
    classification_prompt = f"""
You are a highly intelligent financial assistant.  
Classify the user's intent and extract the company name if relevant.  

Valid intents: ["profile", "company"]  

Rules:  
1. "profile" → User is asking about personal financial planning, portfolio allocation, savings, risk assessment, retirement planning, or general investment strategy.  
2. "company" → User is asking about a specific company, stock ticker, or wants to invest in a specific company (e.g., "Tesla", "AAPL").  

Important:  
- Treat company names and tickers **case-insensitively**.  
- If the user mentions a company name or stock ticker anywhere, the intent is "company".  
- Return the company name **exactly as mentioned**.  
- If multiple companies are mentioned, return only the first one.  
- If no company is mentioned, set "company_name" to null.  

Examples:  
- "I want to invest in Tesla" → {{"intent": "company", "company_name": "Tesla"}}  
- "what about aapl stock?" → {{"intent": "company", "company_name": "AAPL"}}  
- "Help me plan my retirement portfolio" → {{"intent": "profile", "company_name": null}}  

Respond **ONLY in JSON** like this:  
{{
    "intent": "profile" or "company",
    "company_name": "Tesla" or null
}}

User query: "{user_query}"
"""


    response = llm.invoke(classification_prompt)
    response_text = getattr(response, "content", None)  # safe way
    if response_text is None:
        response_text = str(response)

    try:
        intent_obj = json.loads(response_text.strip())
        return {
            "intent": intent_obj.get("intent", "profile"),
            "company_name": intent_obj.get("company_name")
        }
    except Exception as e:
        # fallback default
        return {"intent": "profile", "company_name": None}
    
if __name__ == "__main__":
    result = decide_and_execute("im 18 and want to invest in tesla")
    print(result)
