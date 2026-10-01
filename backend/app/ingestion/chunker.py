def create_chunks(
    pages,
    chunk_size=500,
    overlap=100
):

    chunks = []

    chunk_id = 0

    for page in pages:

        text = page["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "page": page["page"]
                })

                chunk_id += 1

            start += chunk_size - overlap

    return chunks