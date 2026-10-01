import os

from dotenv import load_dotenv
from pinecone import Pinecone


# Load environment variables
load_dotenv(".env")


API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")


if not API_KEY:
    raise ValueError(
        "PINECONE_API_KEY not found in .env"
    )


if not INDEX_NAME:
    raise ValueError(
        "PINECONE_INDEX_NAME not found in .env"
    )


# Connect to Pinecone
pc = Pinecone(
    api_key=API_KEY
)


# Connect to existing index
index = pc.Index(
    INDEX_NAME
)


def upload_chunks(
    chunks,
    embeddings
):

    vectors = []

    for chunk, embedding in zip(
        chunks,
        embeddings
    ):

        vector = {

            "id": f"chunk-{chunk['chunk_id']}",

            "values": embedding,

            "metadata": {

                "text": chunk["text"],

                "page": chunk["page"],

                "source": "dbms.pdf",

                "chunk_id": chunk["chunk_id"]

            }

        }

        vectors.append(vector)


    # Upload vectors
    index.upsert(
        vectors=vectors
    )


    print(
        f"Uploaded {len(vectors)} vectors to Pinecone."
    )


def delete_all_vectors():

    print(
        "Deleting all existing vectors..."
    )

    index.delete(
        delete_all=True
    )

    print(
        "All old vectors deleted successfully."
    )