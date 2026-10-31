
import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone, DenseVectorQuery
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

# Pinecone connection
pc = Pinecone(api_key=API_KEY)
index = pc.index(INDEX_NAME)

# Must match the ingestion namespace
NAMESPACE = "academic-rag"

# Same model used during ingestion
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

    # Search schema-based Pinecone document index
    results = index.documents.search(
        namespace=NAMESPACE,
        top_k=top_k,
        score_by=[
            DenseVectorQuery(
                field="embedding",
                values=query_embedding
            )
        ],
        include_fields=[
            "chunk_text",
            "title",
            "page",
            "source",
            "chunk_id"
        ]
    )

    # Convert Pinecone documents into the old
    # metadata-based response format for compatibility
    matches = []

    for doc in results.matches:

        matches.append({
            "id": doc.id,
            "score": doc.score,
            "metadata": {
                "text": doc.get("chunk_text", ""),
                "title": doc.get("title", ""),
                "page": doc.get("page", 0),
                "source": doc.get("source", "dbms.pdf"),
                "chunk_id": doc.get("chunk_id", 0)
            }
        })

    return matches
