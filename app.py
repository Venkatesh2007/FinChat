# app.py
import streamlit as st
import pandas as pd
import re, json
import plotly.graph_objects as go

from core.intent import detect_intent
from core.userInfo import extract_user_profile
from core.portfolio import allocate_portfolio
from core.sentiment_adjust import adjust_portfolio
from core.response import generate_final_response
from core.decide_and_execute import decide_and_execute
from core.stocks import recommend_stocks
from core.company_stock import fetch_company_stock, predict_future_stock
from core.response_llm import generate_financial_advice


# --------------------------
# Chatbot Helpers
# --------------------------
def init_session_state():
    if "chats" not in st.session_state:
        st.session_state.chats = {}  # {chat_id: {"history": [], "portfolio_results": None, "summary": ""}}
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = "Chat 1"
        st.session_state.chats["Chat 1"] = {"history": [], "portfolio_results": None, "summary": ""}


def new_chat():
    chat_id = f"Chat {len(st.session_state.chats) + 1}"
    st.session_state.chats[chat_id] = {"history": [], "portfolio_results": None, "summary": ""}
    st.session_state.current_chat = chat_id


def run_portfolio_allocation(intent_obj, profile, chat_id):
    """ Portfolio allocation workflow (runs once per chat, then cached) """
    if st.session_state.chats[chat_id]["portfolio_results"]:
        return st.session_state.chats[chat_id]["portfolio_results"]

    base_alloc = allocate_portfolio(profile)
    result = adjust_portfolio(profile, base_alloc)
    adjusted_dict = result.adjusted_allocation
    final_text = generate_final_response(profile, base_alloc, result)

    results = {
        "profile": profile.model_dump(),
        "base_allocation": base_alloc,
        "adjusted_allocation": adjusted_dict,
        "final_text": final_text,
        "adjust_result": result.model_dump(),
    }
    st.session_state.chats[chat_id]["portfolio_results"] = results
    return results


def render_chat_ui():
    """ Render chat history with user + assistant messages """
    chat_id = st.session_state.current_chat
    history = st.session_state.chats[chat_id]["history"]

    for idx,msg in enumerate(history):
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                
                if isinstance(msg["content"], dict):  # bot may return structured output
                    if "text" in msg["content"]:
                        st.markdown(msg["content"]["text"])

                    # Show expanders if present
                    if "profile" in msg["content"]:
                        with st.expander("👤 Step 2: User Profile Extraction", expanded=False):
                            st.json(msg["content"]["profile"])

                    if "base_allocation" in msg["content"]:
                        with st.expander("📊 Step 3: Base Portfolio Allocation", expanded=False):
                            st.json(msg["content"]["base_allocation"])

                    if "adjust_result" in msg["content"]:
                        with st.expander("📈 Step 4: Market Sentiment Adjustment", expanded=False):
                            st.json(msg["content"]["adjust_result"])

                    if "adjusted_allocation" in msg["content"]:
                        fig = go.Figure(
                            data=[go.Pie(
                                labels=list(msg["content"]["adjusted_allocation"].keys()),
                                values=list(msg["content"]["adjusted_allocation"].values()),
                                hole=0.3,
                                pull=[0.05]*len(msg["content"]["adjusted_allocation"]),
                                marker=dict(line=dict(color='#000000', width=2))
                            )]
                        )
                        fig.update_traces(textinfo='percent+label')
                        fig.update_layout(title="📊 Adjusted Portfolio Allocation")
                        st.plotly_chart(fig, use_container_width=True,key=f"portfolio_chart_{idx}")

                    if "stock_data" in msg["content"]:
                        with st.expander("📊 Stock Data", expanded=False):
                            st.json(msg["content"]["stock_data"])

                    if "monte_carlo" in msg["content"]:
                        if msg["content"]["monte_carlo"]:
                            with st.expander("🔮 Monte Carlo Prediction", expanded=False):
                                st.json(msg["content"]["monte_carlo"])

                        # Plot historical + Monte Carlo
                        stock_data = msg["content"]["stock_data"]
                        historical_prices = stock_data.get("historical_prices", [])
                        if historical_prices:
                            df = pd.DataFrame({"price": historical_prices})
                            df['date'] = pd.date_range(end=pd.Timestamp.today(), periods=len(df))

                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df['date'], y=df['price'],
                                                    mode='lines+markers', name='Historical Price'))

                            mc_data = msg["content"]["monte_carlo"]
                            if hasattr(mc_data, "model_dump"):
                                mc_data = mc_data.model_dump()

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

                            fig.update_layout(
                                title=f"📈 {stock_data.get('company_name', 'Stock')} Prices & Monte Carlo Prediction",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                template="plotly_white"
                            )
                            st.plotly_chart(fig, use_container_width=True,key=f"stock_chart_{idx}")
                    if "decision_validation" in msg["content"]:
                        with st.expander("✅ Decision Validation", expanded=False):
                            st.markdown(msg["content"]["decision_validation"])

                    if "trustworthiness" in msg["content"]:
                        with st.expander("🔒 Trustworthiness", expanded=False):
                            st.markdown(msg["content"]["trustworthiness"])

                    if "investment_plan" in msg["content"]:
                        with st.expander("📈 Investment Plan", expanded=True):
                            st.markdown(msg["content"]["investment_plan"])

                    if "risk_analysis" in msg["content"]:
                        with st.expander("⚠️ Risk Analysis", expanded=False):
                            st.markdown(msg["content"]["risk_analysis"])

                    if "expected_returns" in msg["content"]:
                        with st.expander("💰 Expected Returns", expanded=False):
                            st.markdown(msg["content"]["expected_returns"])

                    if "step_by_step" in msg["content"]:
                        with st.expander("📝 Step-by-Step Plan", expanded=True):
                            for step in msg["content"]["step_by_step"]:
                                st.markdown(f"- {step}")

                    if "sources" in msg["content"]:
                        with st.expander("📚 Sources & References", expanded=False):
                            for src in msg["content"]["sources"]:
                                st.markdown(f"- {src}")
                else:
                    st.markdown(msg["content"])


# --------------------------
# Main App
# --------------------------
st.set_page_config(page_title="Smart Wealth Chatbot", layout="wide")
st.title("💬 Smart Wealth Chatbot")

init_session_state()

# Sidebar: Chat management
with st.sidebar:
    st.subheader("💬 Chats")
    if st.button("➕ New Chat"):
        new_chat()
    for chat_id in st.session_state.chats.keys():
        if st.button(chat_id, key=chat_id):
            st.session_state.current_chat = chat_id

st.write(f"### Current Session: {st.session_state.current_chat}")

# Render chat history
render_chat_ui()

# Input at bottom like real chat
if user_query := st.chat_input("Type your query..."):
    chat_id = st.session_state.current_chat
    st.session_state.chats[chat_id]["history"].append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("🔍 Thinking..."):
            intent_obj = detect_intent(user_query)
            profile = extract_user_profile(user_query)

            if intent_obj.intent == "Portfolio_Allocation":
                results = run_portfolio_allocation(intent_obj, profile, chat_id)
                bot_reply = {"text": results["final_text"], **results}

            elif intent_obj.intent == "Investment_Prediction":
                results = run_portfolio_allocation(intent_obj, profile, chat_id)
                adjusted_dict = results["adjusted_allocation"]

                company = decide_and_execute(user_query)
                if company["intent"] == "profile":
                    stock_info = recommend_stocks(adjusted_dict)
                    monte_carlo = None
                else:
                    if company.get("company_name"):  # ✅ ensure not None
                        stock_info = fetch_company_stock(company["company_name"])
                        monte_carlo = predict_future_stock(stock_info, days=252, simulations=500)
                    else:
                        stock_info = None
                        monte_carlo = None

                if stock_info:
                    stock_data = stock_info.model_dump()

                    bot_reply = {
                        "text": f"Here’s my investment outlook for {stock_data.get('company_name','the company')}."
                    }
                history = st.session_state.chats[chat_id]["history"]
                history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history[-3:]])
                final_advice = generate_financial_advice(
                    user_query=f"{history_text}\nUser now asks: {user_query}",
                    user_profile=profile,
                    stock_data=stock_data if stock_info else None,
                    monte_carlo=monte_carlo if stock_info else None,
                    portfolio=adjusted_dict,
                )
                raw_response = final_advice.step_by_step[0] if final_advice.step_by_step else final_advice.summary

                # 1. Extract JSON inside triple backticks
                match = re.search(r"```json\n(.*?)```", raw_response, re.DOTALL)
                if match:
                    json_str = match.group(1).strip()
                    try:
                        parsed = json.loads(json_str)
                    except json.JSONDecodeError:
                        parsed = {}
                else:
                    parsed = {}

                # 2. If JSON parsed, use structured fields; else fallback to summary
                if parsed:
                    bot_reply = {
                        "text": parsed.get("summary", final_advice.summary),
                        "decision_validation": parsed.get("decision_validation", final_advice.decision_validation),
                        "trustworthiness": parsed.get("trustworthiness", final_advice.trustworthiness),
                        "investment_plan": parsed.get("investment_plan", final_advice.investment_plan),
                        "risk_analysis": parsed.get("risk_analysis", final_advice.risk_analysis),
                        "expected_returns": parsed.get("expected_returns", final_advice.expected_returns),
                        "step_by_step": parsed.get("step_by_step", final_advice.step_by_step),
                        "sources": parsed.get("sources", final_advice.sources),
                        
                        "stock_data": stock_data if stock_info else None,
                        "monte_carlo": monte_carlo if stock_info else None,
                    }
                else:
                    bot_reply = {
                        "text": final_advice.summary,
                        "decision_validation": final_advice.decision_validation,
                        "trustworthiness": final_advice.trustworthiness,
                        "investment_plan": final_advice.investment_plan,
                        "risk_analysis": final_advice.risk_analysis,
                        "expected_returns": final_advice.expected_returns,
                        "step_by_step": final_advice.step_by_step,
                        "sources": final_advice.sources,
                        "stock_data": stock_data if stock_info else None,
                        "monte_carlo": monte_carlo if stock_info else None,
                    }


            elif intent_obj.intent == "General_Chat":
                from core.llm import get_llm
                llm = get_llm()
                history = st.session_state.chats[chat_id]["history"]

                messages = [{"role": "system", "content": "You are FinChat, a friendly and trustworthy financial mentor. Respond naturally in the same mood as the user."}]
                messages += history[-5:]  # last 5 turns
                messages.append({"role": "user", "content": user_query})
                bot_resp = llm.invoke(messages)
                bot_reply = bot_resp.content

            elif intent_obj.intent == "Knowledge":
                from vectorstores.faiss import rag_query
                history = st.session_state.chats[chat_id]["history"]
                history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history[-3:]])
                query_with_context = f"""
                Conversation so far:
                {history_text}

                Now the user asks: {user_query}
                """
                result = rag_query(query_with_context)
                bot_reply = result['answer']


            else:
                bot_reply = f"⚠️ Intent '{intent_obj.intent}' not yet implemented."

            # Show live response
            if isinstance(bot_reply, dict):
                if "text" in bot_reply:
                    st.markdown(bot_reply["text"])
            else:
                st.markdown(bot_reply)

    # Save assistant reply in history
    st.session_state.chats[chat_id]["history"].append({"role": "assistant", "content": bot_reply})
    st.rerun()
