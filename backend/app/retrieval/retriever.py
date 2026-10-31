
import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


# Load environment variables
BACKEND_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND_DIR / ".env")

API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not API_KEY:
    raise ValueError("PINECONE_API_KEY not found.")

if not INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME not found.")


# Connect to Pinecone
pc = Pinecone(api_key=API_KEY)
index = pc.Index(INDEX_NAME)

NAMESPACE = "academic-rag"


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve(query, top_k=5):

    # Generate query embedding
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    if len(query_embedding) != 384:
        raise ValueError(
            f"Expected 384 dimensions, got {len(query_embedding)}"
        )

    # Search Pinecone using Vector API
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=NAMESPACE,
        include_metadata=True
    )

    matches = []

    for match in results.matches:

        metadata = match.metadata or {}

        matches.append({
            "id": match.id,
            "score": match.score,
            "metadata": {
                "text": metadata.get("text", ""),
                "title": metadata.get("title", ""),
                "page": metadata.get("page", 0),
                "source": metadata.get("source", "dbms.pdf"),
                "chunk_id": metadata.get("chunk_id", 0)
            }
        })

    return matches
