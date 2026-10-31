
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Locate project directories
PROJECT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_DIR / "backend"

# Load backend environment variables
load_dotenv(BACKEND_DIR / ".env")

# Allow Python to import existing backend modules
sys.path.insert(0, str(BACKEND_DIR))

from app.retrieval.retriever import retrieve
from app.generation.generator import generate_answer


# Page configuration
st.set_page_config(
    page_title="Academic RAG Assistant",
    page_icon="🎓",
    layout="wide"
)

# Application heading
st.title("🎓 Academic RAG Assistant")
st.caption("Your AI-powered academic study companion.")

# Sidebar
with st.sidebar:
    st.title("📚 StudyMate AI")
    st.markdown("---")
    st.subheader("About")
    st.write(
        "Ask questions from your academic PDFs "
        "and get context-based answers."
    )
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            sources = message.get("sources", [])

            if sources:
                with st.expander("📖 View Sources"):
                    for source in sources:
                        st.write(
                            f"📄 {source['source']} | "
                            f"Page: {source['page']} | "
                            f"Score: {source['score']:.4f}"
                        )

# Chat input
question = st.chat_input(
    "Ask a question from your study material..."
)

if question:
    # Display user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve and generate answer
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your study material..."):

                results = retrieve(question, top_k=5)

                if not results:
                    answer = "No relevant information found in your uploaded study material."
                    sources = []

                else:
                    with st.spinner("Generating your answer..."):
                        answer = generate_answer(question, results)

                    sources = []

                    for result in results:
                        metadata = result.get("metadata", {})

                        sources.append({
                            "source": metadata.get("source", "Unknown"),
                            "page": metadata.get("page", "Unknown"),
                            "score": float(result.get("score", 0))
                        })

            st.markdown(answer)

            if sources:
                with st.expander("📖 View Sources"):
                    for source in sources:
                        st.write(
                            f"📄 {source['source']} | "
                            f"Page: {source['page']} | "
                            f"Score: {source['score']:.4f}"
                        )

            # Save assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        except Exception as e:
            error_message = f"Something went wrong: {str(e)}"
            st.error(error_message)

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                "sources": []
            })
