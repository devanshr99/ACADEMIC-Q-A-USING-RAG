# Academic RAG

A starter scaffold for an academic document retrieval-augmented generation (RAG) application. The backend package is organized into ingestion, retrieval, and generation modules; implementation details such as the language-model provider and retrieval strategy are intentionally left open.

## Backend setup

From the `backend` directory, create and activate a virtual environment, then install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start the API during development with:

```powershell
uvicorn app.main:app --reload
```

Source documents belong in `backend/data/documents/`. Persistent vector-store data belongs in `backend/vectorstore/`. Configure local settings in `backend/.env`; do not commit real credentials.

## Backend layout

- `app/ingestion/`: document loading, chunking, and embeddings
- `app/retrieval/`: retrieval from the vector store
- `app/generation/`: grounded response generation
- `data/documents/`: source academic documents
- `vectorstore/`: persisted vector index
