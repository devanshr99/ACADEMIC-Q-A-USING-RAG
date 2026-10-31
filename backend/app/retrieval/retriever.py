
import os
import logging
import hashlib
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
logger.info("Loading Sentence Transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
logger.info("Embedding model loaded successfully.")


def retrieve(query, top_k=5):

    logger.info("========== RETRIEVAL START ==========")
    logger.info("Index: %s", INDEX_NAME)
    logger.info("Namespace: %s", NAMESPACE)
    logger.info("Query: %s", query)

    try:
        # Safe API key fingerprint (never log the actual key)
        key_fingerprint = hashlib.sha256(
            API_KEY.encode()
        ).hexdigest()[:10]

        logger.info("PINECONE KEY CHECK: %s", key_fingerprint)

        # Check actual index host during each query
        index_description = pc.describe_index(INDEX_NAME)

        logger.info(
            "CLOUD INDEX HOST: %s",
            index_description.host
        )

        # Generate query embedding
        query_embedding = model.encode(
            query,
            normalize_embeddings=True
        ).tolist()

        logger.info(
            "Embedding generated: %s dimensions",
            len(query_embedding)
        )

        if len(query_embedding) != 384:
            raise ValueError(
                f"Expected 384 dimensions, got {len(query_embedding)}"
            )

        # Check Pinecone statistics
        stats = index.describe_index_stats()

        logger.info(
            "Total Pinecone vectors: %s",
            stats.total_vector_count
        )

        namespace_stats = stats.namespaces or {}
        namespace_info = namespace_stats.get(NAMESPACE)

        if namespace_info:
            logger.info(
                "Vectors in namespace %s: %s",
                NAMESPACE,
                namespace_info.vector_count
            )
        else:
            logger.warning(
                "Namespace %s not found in index statistics",
                NAMESPACE
            )

        # Search Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=NAMESPACE,
            include_metadata=True
        )

        logger.info(
            "Retrieved matches: %s",
            len(results.matches)
        )

        matches = []

        for match in results.matches:

            metadata = match.metadata or {}

            logger.info(
                "Match ID: %s | Score: %s | Text available: %s",
                match.id,
                match.score,
                bool(metadata.get("text"))
            )

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

        logger.info("========== RETRIEVAL COMPLETE ==========")

        return matches

    except Exception:
        logger.exception("PINECONE RETRIEVAL FAILED")
        raise
