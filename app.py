# app.py
import streamlit as st
from core.intent import detect_intent
from core.userInfo import extract_user_profile
from core.portfolio import allocate_portfolio
from core.sentiment_adjust import adjust_portfolio
from core.response import generate_final_response

st.set_page_config(page_title="Smart Wealth Chatbot", layout="wide")

st.title("💬 Smart Wealth Chatbot")
st.markdown("Ask me about your finances and I’ll guide you with reasoning + personalized suggestions.")

# User input
user_query = st.text_area("Type your query here", height=100)

if st.button("Ask") and user_query.strip():
    with st.spinner("🔍 Analyzing your query..."):
        # Step 1: Intent classification
        intent_obj = detect_intent(user_query)

        with st.expander("🧭 Step 1: Intent Detection", expanded=False):
            st.json(intent_obj.model_dump())

        if intent_obj.intent == "Portfolio_Allocation":
            # Step 2: Extract user profile
            profile = extract_user_profile(user_query)
            with st.expander("👤 Step 2: User Profile Extraction", expanded=False):
                st.json(profile.dict())

            # Step 3: Base portfolio allocation
            base_alloc = allocate_portfolio(profile)
            with st.expander("📊 Step 3: Base Portfolio Allocation", expanded=False):
                st.json(base_alloc)

            # Step 4: Sentiment-adjusted allocation
            result = adjust_portfolio(profile, base_alloc)
            with st.expander("📈 Step 4: Market Sentiment Adjustment", expanded=False):
                st.json(result.model_dump())

            # Step 5: Final engaging response
            final_text = generate_final_response(profile, base_alloc, result)
            with st.expander("💡 Final Advice (Engaging)", expanded=True):
                st.markdown(final_text)

        else:
            st.warning(f"⚠️ Intent '{intent_obj.intent}' not yet implemented in demo. Try portfolio queries like 'I’m 20 earning 5000, how should I invest?'")
