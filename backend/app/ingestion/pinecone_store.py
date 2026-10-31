
import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone


# Load environment variables reliably
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


def upload_chunks(chunks, embeddings):

    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings count mismatch.")

    documents = []

    for chunk, embedding in zip(chunks, embeddings):

        if len(embedding) != 384:
            raise ValueError(
                f"Expected 384 dimensions, got {len(embedding)}"
            )

        document = {
            "_id": f"chunk-{chunk['chunk_id']}",
            "embedding": [float(value) for value in embedding],
            "chunk_text": chunk["text"],
            "title": "DBMS",
            "page": int(chunk["page"]),
            "source": "dbms.pdf",
            "chunk_id": int(chunk["chunk_id"])
        }

        documents.append(document)

    # Upload in batches of 50 documents
    total_uploaded = 0

    for start in range(0, len(documents), 50):

        batch = documents[start:start + 50]

        response = index.documents.upsert(
            namespace=NAMESPACE,
            documents=batch
        )

        total_uploaded += response.upserted_count

    print(f"Uploaded {total_uploaded} documents to Pinecone.")


def delete_all_vectors():

    print("Deleting all documents...")

    index.documents.delete(
        namespace=NAMESPACE,
        delete_all=True
    )

    print("All documents deleted successfully.")
