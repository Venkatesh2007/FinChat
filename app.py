# app.py
import streamlit as st
from core.intent import detect_intent
from core.userInfo import extract_user_profile
from core.portfolio import allocate_portfolio
from core.sentiment_adjust import adjust_portfolio
from core.response import generate_final_response
import plotly.graph_objects as go



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
            
            adjusted_dict = result.adjusted_allocation
            if adjusted_dict:
                fig = go.Figure(
                    data=[go.Pie(
                        labels=list(adjusted_dict.keys()),
                        values=list(adjusted_dict.values()),
                        hole=0.3,  # donut style
                        pull=[0.05]*len(adjusted_dict),  # pull slices slightly for 3D effect
                        marker=dict(line=dict(color='#000000', width=2))
                    )]
                )
                fig.update_traces(textinfo='percent+label')
                fig.update_layout(title="📊 Adjusted Portfolio Allocation (3D-style)")
                st.plotly_chart(fig, use_container_width=True)



            # Step 5: Final engaging response
            final_text = generate_final_response(profile, base_alloc, result)
            with st.expander("💡 Final Advice (Engaging)", expanded=True):
                st.markdown(final_text)
        elif intent_obj.intent == "General_Chat":
            # Mood Detection Prompt
            from core.llm import get_llm
            llm = get_llm()

            # Generate Chatbot Response in Same Mood
            chat_prompt = f"""
            You are FinChat, a friendly and trustworthy financial mentor.  
            adapt your tone to the user's mood.
            [happy, sad, angry, confused, neutral, grateful, casual].
            
            Respond in the same tone/mood. Keep it natural, short, and conversational.
            User said: "{user_query}"
            """
            bot_resp = llm.invoke([{"role":"user","content": chat_prompt}])

            st.markdown(bot_resp.content)

        else:
            st.warning(f"⚠️ Intent '{intent_obj.intent}' not yet implemented in demo. Try portfolio queries like 'I’m 20 earning 5000, how should I invest?'")
