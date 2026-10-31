import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone

BACKEND_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND_DIR / ".env")

API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not API_KEY:
    raise ValueError("PINECONE_API_KEY not found.")

if not INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME not found.")

pc = Pinecone(api_key=API_KEY)
index = pc.Index(INDEX_NAME)

NAMESPACE = "academic-rag"


def upload_chunks(chunks, embeddings):

    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings count mismatch.")

    vectors = []

    for chunk, embedding in zip(chunks, embeddings):

        if len(embedding) != 384:
            raise ValueError(
                f"Expected 384 dimensions, got {len(embedding)}"
            )

        vectors.append({
            "id": f"chunk-{chunk['chunk_id']}",
            "values": [float(value) for value in embedding],
            "metadata": {
                "text": chunk["text"],
                "title": "DBMS",
                "page": int(chunk["page"]),
                "source": "dbms.pdf",
                "chunk_id": int(chunk["chunk_id"])
            }
        })

    total_uploaded = 0

    for start in range(0, len(vectors), 50):

        batch = vectors[start:start + 50]

        response = index.upsert(
            vectors=batch,
            namespace=NAMESPACE
        )

        total_uploaded += response.upserted_count

    print(f"Uploaded {total_uploaded} vectors to Pinecone.")


def delete_all_vectors():

    print("Deleting all vectors...")

    index.delete(
        delete_all=True,
        namespace=NAMESPACE
    )

    print("All vectors deleted successfully.")