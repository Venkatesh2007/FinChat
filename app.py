# app.py
import streamlit as st
import pandas as pd
from core.intent import detect_intent
from core.userInfo import extract_user_profile
from core.portfolio import allocate_portfolio
from core.sentiment_adjust import adjust_portfolio
from core.response import generate_final_response
from core.decide_and_execute import decide_and_execute
from core.stocks import recommend_stocks
from core.company_stock import fetch_company_stock, predict_future_stock
from core.response_llm import generate_financial_advice
import plotly.graph_objects as go
import re 
import json

def run_portfolio_allocation(intent_obj, profile):
    """
    Complete workflow for portfolio allocation including:
    - User profile extraction
    - Base allocation
    - Sentiment-adjusted allocation
    - Visualization
    - Final advice generation
    """


    # Step 2: Extract user profile
    with st.expander("👤 Step 2: User Profile Extraction", expanded=False):
        st.json(profile.model_dump())

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
        # Visualization: Donut-style pie chart
        fig = go.Figure(
            data=[go.Pie(
                labels=list(adjusted_dict.keys()),
                values=list(adjusted_dict.values()),
                hole=0.3,  # donut style
                pull=[0.05]*len(adjusted_dict),  # slight pull for 3D effect
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

    # Return values if needed for further processing
    return {
        "base_allocation": base_alloc,
        "adjusted_allocation": adjusted_dict,
        "final_text": final_text
    }


st.set_page_config(page_title="Smart Wealth Chatbot", layout="wide")

st.title("💬 Smart Wealth Chatbot")
st.markdown("Ask me about your finances and I’ll guide you with reasoning + personalized suggestions.")

# User input
user_query = st.text_area("Type your query here", height=100)

if st.button("Ask") and user_query.strip():
    with st.spinner("🔍 Analyzing your query..."):
        # Step 1: Intent classification
        intent_obj = detect_intent(user_query)
        profile = extract_user_profile(user_query)
        adjusted_dict = None

        with st.expander("🧭 Step 1: Intent Detection", expanded=False):
            st.json(intent_obj.model_dump())

        stock_data, monte_carlo, portfolio = None, None, None
        if intent_obj.intent == "Portfolio_Allocation":
            results = run_portfolio_allocation(intent_obj, profile)


        elif intent_obj.intent == "Investment_Prediction":
            company = decide_and_execute(user_query)
            results = run_portfolio_allocation(intent_obj, profile)
            adjusted_dict = results["adjusted_allocation"]   
            print(company)
            if company["intent"] == "profile":
                stock_info = recommend_stocks(adjusted_dict)
            else:
                stock_info = fetch_company_stock(company["company_name"])
                
            if stock_info:
                # Convert to dict (Pydantic v2)
                stock_data = stock_info.model_dump()
                
                # Predict future stock (Monte Carlo simulation)
                monte_carlo = predict_future_stock(stock_info, days=252, simulations=500)
                
                # Display raw JSON
                with st.expander("📊 Stock Data", expanded=False):
                    st.json(stock_data)
                with st.expander("🔮 Monte Carlo Prediction", expanded=False):
                    st.json(monte_carlo)
                
                # -----------------------
                # Plot Historical Prices & Monte Carlo
                # -----------------------
                
                # Historical prices
                historical_prices = stock_data.get("historical_prices", [])

                if historical_prices:
                    # Create DataFrame with artificial dates
                    df = pd.DataFrame({"price": historical_prices})
                    df['date'] = pd.date_range(end=pd.Timestamp.today(), periods=len(df))

                    # Initialize Plotly figure
                    fig = go.Figure()

                    # Plot historical prices
                    fig.add_trace(go.Scatter(
                        x=df['date'],
                        y=df['price'],
                        mode='lines+markers',
                        name='Historical Price',
                        line=dict(color='blue')
                    ))

                    # Ensure Monte Carlo is a dict (handle Pydantic)
                    if hasattr(monte_carlo, "model_dump"):
                        mc_data = monte_carlo.model_dump()
                    else:
                        mc_data = monte_carlo

                    # Plot Monte Carlo expected price and bounds
                    fig.add_hline(y=mc_data.get("expected_price", 0),
                                line_dash="dash", line_color="green",
                                annotation_text="Expected Price",
                                annotation_position="top right")
                    fig.add_hline(y=mc_data.get("lower_bound_5pct", 0),
                                line_dash="dot", line_color="red",
                                annotation_text="5% Lower Bound",
                                annotation_position="bottom right")
                    fig.add_hline(y=mc_data.get("upper_bound_95pct", 0),
                                line_dash="dot", line_color="red",
                                annotation_text="95% Upper Bound",
                                annotation_position="top left")

                    # Layout
                    fig.update_layout(
                        title=f"📈 {stock_data.get('company_name', 'Stock')} Prices & Monte Carlo Prediction",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        template="plotly_white"
                    )

                    # Show chart in Streamlit
                    st.plotly_chart(fig, use_container_width=True)

            # Step 3: Generate final advice
            final_advice = generate_financial_advice(
                user_query=user_query,
                user_profile=profile,
                stock_data=stock_data,
                monte_carlo=monte_carlo,
                portfolio=portfolio
            )

            st.subheader("🧑‍💼 Final Advice")
            raw_response = final_advice.step_by_step[0]  # The big JSON string inside ```json
            print(raw_response)

            # 1. Extract JSON inside the triple backticks
            match = re.search(r"```json\n(.*?)```", raw_response, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                try:
                    parsed = json.loads(json_str)
                except json.JSONDecodeError:
                    parsed = {}
            else:
                parsed = {}

            # 2. Override fallback values if JSON parsed successfully
            if parsed:
                summary = parsed.get("summary", final_advice.summary)
                decision_validation = parsed.get("decision_validation", final_advice.decision_validation)
                trustworthiness = parsed.get("trustworthiness", final_advice.trustworthiness)
                investment_plan = parsed.get("investment_plan", final_advice.investment_plan)
                risk_analysis = parsed.get("risk_analysis", final_advice.risk_analysis)
                expected_returns = parsed.get("expected_returns", final_advice.expected_returns)
                step_by_step = parsed.get("step_by_step", final_advice.step_by_step)
                sources = parsed.get("sources", final_advice.sources)
            else:
                summary = final_advice.summary
                decision_validation = final_advice.decision_validation
                trustworthiness = final_advice.trustworthiness
                investment_plan = final_advice.investment_plan
                risk_analysis = final_advice.risk_analysis
                expected_returns = final_advice.expected_returns
                step_by_step = final_advice.step_by_step
                sources = final_advice.sources

            # 3. Streamlit display
            st.write(summary)

            st.markdown(f"**Decision Validation:** {decision_validation}")
            st.markdown(f"**Trustworthiness:** {trustworthiness}")
            st.markdown(f"**Investment Plan:** {investment_plan}")
            st.markdown(f"**Risk Analysis:** {risk_analysis}")
            st.markdown(f"**Expected Returns:** {expected_returns}")

            st.markdown("**Step-by-Step Plan:**")
            for step in step_by_step:
                st.markdown(f"- {step}")

            st.markdown("**Sources & References:**")
            for source in sources:
                st.markdown(f"- {source}")


            
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
        
        

