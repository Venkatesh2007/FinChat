import os
from pathlib import Path
import asyncio
import glob
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from core.llm import get_llm

load_dotenv()

def get_embeddings():
    # Ensure a running event loop exists
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",  # or use EMBED_MODEL
        google_api_key=GEMINI_API_KEY,
        request_options={"api_endpoint": "generativelanguage.googleapis.com"}
    )


INDEX_DIR = Path("finance_faiss")

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")  
EMBED_MODEL = "models/text-embedding-004"

SYSTEM_PROMPT = """You are a helpful financial mentor for young investors.
            - Always provide a clear answer  == "_even if the context is incomplete or noisy.
            - If context text looks broken, use your own financial knowledge to answer.
            - Explain in beginner-friendly terms, no jargon.
            - Always end with one practical action in Indian rupees (₹).
            """

USER_PROMPT_TMPL = PromptTemplate.from_template(
            """Question: {question}

            Use the following context to answer:

            {context}

            Answer:""")



def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text

def rag_query(question: str, k: int = 4):
    # Load FAISS
    embeddings = get_embeddings()
    print("embeddings completed")
    INDEX_DIR = Path(__file__).resolve().parent / "finance_faiss"

    print("🔍 Looking in:", INDEX_DIR)
    print("index.faiss exists?", (INDEX_DIR / "index.faiss").exists())
    print("index.pkl exists?", (INDEX_DIR / "index.pkl").exists())

    vs = FAISS.load_local(
        folder_path=str(INDEX_DIR),
        embeddings=embeddings,
        index_name="index",
        allow_dangerous_deserialization=True
    )

    print("✅ FAISS index loaded!")
    retriever = vs.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)

    # Format context
    context = "\n\n".join([
        f"[Source: {d.metadata.get('source', 'PDF')}] {clean_text(d.page_content)}"
        for d in docs
    ])

    # Build prompt
    prompt = USER_PROMPT_TMPL.format(question=question, context=context)
    llm = get_llm()

    try:
        resp = llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ])
        answer_text = resp.content if hasattr(resp, "content") else str(resp)
        if not answer_text.strip():
            raise ValueError("Empty Groq response")
    except Exception as e:
        print(f"⚠️ Groq failed, fallback triggered: {e}")
        answer_text = (
            f"⚠️ Groq failed, fallback triggered: {e}"
        )
    
    return {"answer": answer_text, "context_used": context[:1000]}