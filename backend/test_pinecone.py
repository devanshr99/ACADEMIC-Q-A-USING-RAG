import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv(".env")

api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")

if not api_key:
    raise ValueError("PINECONE_API_KEY not found")

if not index_name:
    raise ValueError("PINECONE_INDEX_NAME not found")

pc = Pinecone(api_key=api_key)

# Check if index already exists
existing_indexes = pc.list_indexes().names()

if index_name not in existing_indexes:
    print("Creating Pinecone index...")

    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    print("Index created successfully!")

else:
    print("Index already exists!")

print("Indexes:", pc.list_indexes().names())