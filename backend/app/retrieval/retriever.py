import os

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


# Load environment variables
load_dotenv(".env")


API_KEY = os.getenv(
    "PINECONE_API_KEY"
)

INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME"
)


if not API_KEY:

    raise ValueError(
        "PINECONE_API_KEY not found."
    )


if not INDEX_NAME:

    raise ValueError(
        "PINECONE_INDEX_NAME not found."
    )


# Pinecone connection
pc = Pinecone(
    api_key=API_KEY
)


index = pc.Index(
    INDEX_NAME
)


# Same embedding model used during ingestion
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def retrieve(
    query,
    top_k=5
):

    # Convert question into embedding
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()


    # Search Pinecone
    results = index.query(

        vector=query_embedding,

        top_k=top_k,

        include_metadata=True

    )


    return results["matches"]