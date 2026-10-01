import os

from dotenv import load_dotenv
from pinecone import Pinecone


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


pc = Pinecone(
    api_key=API_KEY
)


index = pc.Index(
    INDEX_NAME
)


print(
    "\nDeleting old Pinecone vectors..."
)


index.delete(
    delete_all=True
)


print(
    "Old vectors deleted successfully!"
)