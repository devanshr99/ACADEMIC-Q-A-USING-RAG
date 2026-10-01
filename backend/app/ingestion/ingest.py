from app.ingestion.loader import load_pdf
from app.ingestion.chunker import create_chunks
from app.ingestion.embeddings import generate_embeddings
from app.ingestion.pinecone_store import upload_chunks


PDF_PATH = "data/documents/dbms.pdf"


def main():

    print("\n==============================")
    print("      DOCUMENT INGESTION")
    print("==============================")

    print("\nLoading PDF...")

    pages = load_pdf(PDF_PATH)

    print(f"Pages loaded: {len(pages)}")

    if not pages:
        print("ERROR: No text found in PDF.")
        return

    print("\nCreating chunks...")

    chunks = create_chunks(
        pages,
        chunk_size=500,
        overlap=100
    )

    print(f"Chunks created: {len(chunks)}")

    if not chunks:
        print("ERROR: No chunks created.")
        return

    print("\nGenerating embeddings...")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = generate_embeddings(texts)

    print(f"Embeddings generated: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[0])}")

    print("\nUploading to Pinecone...")

    upload_chunks(chunks, embeddings)

    print("\n==============================")
    print("  INGESTION COMPLETED")
    print("==============================")


if __name__ == "__main__":
    main()