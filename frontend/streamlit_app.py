
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Academic RAG Assistant",
    page_icon="🎓",
    layout="wide"
)


# ==========================================
# PROJECT DIRECTORIES
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_DIR / "backend"


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

# Local development: load backend/.env
load_dotenv(BACKEND_DIR / ".env")


# Streamlit Cloud: load secrets
try:
    for key in (
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME",
        "OPENROUTER_API_KEY"
    ):
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])

except Exception:
    # Local development may not have Streamlit secrets
    pass


# ==========================================
# VERIFY CONFIGURATION
# ==========================================

required_keys = [
    "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME",
    "OPENROUTER_API_KEY"
]

missing_keys = [
    key for key in required_keys
    if not os.getenv(key)
]

if missing_keys:
    st.error(
        "Missing configuration: "
        + ", ".join(missing_keys)
    )
    st.stop()


# ==========================================
# BACKEND IMPORTS
# ==========================================

sys.path.insert(0, str(BACKEND_DIR))

from app.retrieval.retriever import retrieve
from app.generation.generator import generate_answer


# ==========================================
# APPLICATION HEADER
# ==========================================

st.title("🎓 Academic RAG Assistant")

st.caption(
    "Your AI-powered academic study companion."
)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("📚 StudyMate AI")

    st.markdown("---")

    st.subheader("About")

    st.write(
        "Ask questions from your academic PDFs "
        "and get context-based answers."
    )

    st.markdown("---")

    st.subheader("Technology")

    st.write(
        """
        - Retrieval-Augmented Generation
        - Sentence Transformers
        - Pinecone Vector Database
        - OpenRouter LLM
        """
    )

    st.markdown("---")

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()


# ==========================================
# INITIALIZE CHAT HISTORY
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================
# DISPLAY PREVIOUS MESSAGES
# ==========================================

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


# ==========================================
# CHAT INPUT
# ==========================================

question = st.chat_input(
    "Ask a question from your study material..."
)


# ==========================================
# PROCESS USER QUESTION
# ==========================================

if question:

    # Display user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Assistant response
    with st.chat_message("assistant"):

        try:

            # ----------------------------------
            # RETRIEVAL
            # ----------------------------------

            with st.spinner(
                "Searching your study material..."
            ):

                results = retrieve(
                    question,
                    top_k=5
                )

            if not results:

                answer = (
                    "No relevant information found "
                    "in your uploaded study material."
                )

                sources = []

            else:

                # ----------------------------------
                # GENERATE ANSWER
                # ----------------------------------

                with st.spinner(
                    "Generating your answer..."
                ):

                    answer = generate_answer(
                        question,
                        results
                    )

                # ----------------------------------
                # EXTRACT SOURCES
                # ----------------------------------

                sources = []

                for result in results:

                    metadata = result.get(
                        "metadata", {}
                    )

                    sources.append({

                        "source": metadata.get(
                            "source",
                            "Unknown"
                        ),

                        "page": metadata.get(
                            "page",
                            "Unknown"
                        ),

                        "score": float(
                            result.get("score", 0)
                        )

                    })

            # ----------------------------------
            # DISPLAY ANSWER
            # ----------------------------------

            st.markdown(answer)

            if sources:

                with st.expander("📖 View Sources"):

                    for source in sources:

                        st.write(
                            f"📄 {source['source']} | "
                            f"Page: {source['page']} | "
                            f"Score: {source['score']:.4f}"
                        )

            # ----------------------------------
            # SAVE CHAT HISTORY
            # ----------------------------------

            st.session_state.messages.append({

                "role": "assistant",

                "content": answer,

                "sources": sources

            })

        except Exception as e:

            error_message = (
                f"Something went wrong: {str(e)}"
            )

            st.error(error_message)

            st.session_state.messages.append({

                "role": "assistant",

                "content": error_message,

                "sources": []

            })
