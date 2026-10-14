# Hybrid PDF RAG Assistant

A locally hosted **Retrieval-Augmented Generation (RAG)** application
that lets you upload PDF documents, search their contents, and ask
questions about them through a FastAPI API.

The project combines **hybrid retrieval**, **semantic reranking**, and a
locally served language model. It also tracks document versions by
detecting whether a PDF's content has changed.

------------------------------------------------------------------------

## 1. What Are We Building?

We are building a document-question-answering system that grounds
answers in your own PDFs instead of relying only on the language model's
general knowledge.

### The main goals

-   Upload PDF documents through an API (for example, using Postman).
-   Extract text and split it into manageable chunks.
-   Find relevant content using both keyword-based and semantic search.
-   Combine search results and rerank them to improve relevance.
-   Send the most relevant context to a locally hosted LLM.
-   Persist the vector index and document metadata on the local machine.
-   Detect changed PDF content and increment its version.
-   Expose the system through REST API endpoints.

------------------------------------------------------------------------

## 2. High-Level Architecture

``` mermaid
flowchart TD
    U[User / Postman] -->|POST PDF| API[FastAPI Application]
    API --> VAL[Validate PDF and calculate SHA-256]
    VAL --> META[(SQLite Metadata Database)]
    VAL --> LOAD[PDF Text Extraction]
    LOAD --> CHUNK[Text Chunking]
    CHUNK --> EMB[Sentence Transformer Embeddings]
    EMB --> FAISS[(Local FAISS Vector Index)]
    CHUNK --> BM25[BM25 Keyword Index]
    FAISS --> FUSION[Reciprocal Rank Fusion]
    BM25 --> FUSION
    FUSION --> RERANK[Cross-Encoder Reranker]
    RERANK --> CTX[Relevant Context]
    Q[User Question] --> ASK[POST /ask]
    ASK --> FUSION
    CTX --> LLM[vLLM - Local LLM]
    LLM --> ANS[Answer Returned by API]
    ANS --> U
```

> **Important:** The SQLite database stores document metadata and
> version information. FAISS stores the searchable vector index. They
> serve different purposes.

------------------------------------------------------------------------

## 3. How a Question Is Answered

``` mermaid
flowchart LR
    A[Question] --> B[Create Query Embedding]
    B --> C[Semantic Search with FAISS]
    A --> D[Keyword Search with BM25]
    C --> E[Reciprocal Rank Fusion]
    D --> E
    E --> F[Cross-Encoder Reranking]
    F --> G[Select Relevant Chunks]
    G --> H[Build Context and Prompt]
    H --> I[vLLM Chat Completion]
    I --> J[Answer Returned]
```

1.  **Receive the question** through the `/ask` endpoint.
2.  **Search in parallel by meaning and keywords**:
    -   FAISS finds semantically similar chunks using embeddings.
    -   BM25 finds chunks with relevant terms.
3.  **Fuse the rankings** with Reciprocal Rank Fusion (RRF).
4.  **Rerank candidates** with a Cross-Encoder so the most relevant
    chunks are preferred.
5.  **Build the prompt** using the selected context.
6.  **Generate the response** using the model served by vLLM.
7.  **Return the answer** from the FastAPI endpoint.

This approach is called **hybrid retrieval** because it combines
semantic retrieval and keyword retrieval.

------------------------------------------------------------------------

## 4. Technology Stack

  ----------------------------------------------------------------------------------------
  Component               Technology                               Responsibility
  ----------------------- ---------------------------------------- -----------------------
  API                     FastAPI                                  Provides upload,
                                                                   question-answering,
                                                                   health, and
                                                                   document-list endpoints

  LLM serving             vLLM                                     Serves the language
                                                                   model through an
                                                                   OpenAI-compatible API

  Language model          `Qwen/Qwen2.5-7B-Instruct-AWQ`           Generates responses
                                                                   using retrieved context

  PDF parsing             PyPDFLoader                              Extracts text from PDFs

  Chunking                RecursiveCharacterTextSplitter           Splits text into chunks
                                                                   of 800 characters with
                                                                   120-character overlap

  Embeddings              Sentence Transformers,                   Converts text into
                          `all-MiniLM-L6-v2`                       vectors for semantic
                                                                   search

  Vector search           FAISS                                    Stores and searches
                                                                   vector embeddings
                                                                   locally

  Keyword search          BM25                                     Finds relevant text
                                                                   using lexical matching

  Result fusion           Reciprocal Rank Fusion (RRF)             Combines semantic and
                                                                   keyword rankings

  Reranking               `cross-encoder/ms-marco-MiniLM-L-6-v2`   Reorders retrieved
                                                                   chunks by query
                                                                   relevance

  Metadata and versions   SQLite                                   Stores file hashes,
                                                                   versions, and update
                                                                   timestamps

  Persistence             Local filesystem                         Stores PDFs, the FAISS
                                                                   index, and metadata
  ----------------------------------------------------------------------------------------

Embedding and reranking models are configured to run on the CPU in this
setup. vLLM uses the configured NVIDIA GPU for language-model inference.

------------------------------------------------------------------------

## 5. Project Layout

The application uses the following local files and directories:

``` text
hybrid-rag/
├── main.py                    # FastAPI application and RAG pipeline
├── docker-compose.yml         # vLLM service configuration, if using Compose
├── pdfs/                      # Uploaded and indexed PDF documents
├── vector_db/
│   ├── index.faiss             # Persisted FAISS vector index
│   ├── index.pkl               # FAISS document-store metadata
│   └── pdf_manifest.json       # Hash manifest used to validate the saved index
└── document_metadata.db       # SQLite database for document versions and metadata
```

Some files are created only after the application indexes its first PDF.
Your actual directory may include additional configuration or support
files.

### What is stored where?

-   **`pdfs/`** --- the current PDF files.
-   **`vector_db/`** --- the persisted semantic-search index and its
    manifest.
-   **`document_metadata.db`** --- document filename, SHA-256 hash,
    version number, and update timestamp.
-   **`main.py`** --- API routes, document processing, retrieval, and
    question-answering logic.

The metadata database is a normal local SQLite file. You can inspect it
with a SQLite browser or Python's `sqlite3` module. Avoid editing the
database manually while the API is running.

------------------------------------------------------------------------

## 6. Document Upload and Versioning

A PDF can be uploaded through the `POST /documents/upload` endpoint.

### Versioning logic

``` mermaid
flowchart TD
    A[Upload PDF] --> B{Is it a valid PDF?}
    B -- No --> X[Return validation error]
    B -- Yes --> C[Calculate SHA-256]
    C --> D{Filename exists in metadata?}
    D -- No --> E[Save as version 1]
    D -- Yes --> F{Does SHA-256 match?}
    F -- Yes --> G[Keep existing version; skip reindexing]
    F -- No --> H[Increment version]
    E --> I[Extract text and rebuild indexes]
    H --> I
    I --> J[Persist FAISS index and manifest]
    J --> K[Save metadata]
```

-   **First upload:** a new filename is registered as version `1`.
-   **Same filename and same content:** the existing version is retained
    and unnecessary reindexing is skipped.
-   **Same filename and changed content:** the version number is
    incremented and the indexes are refreshed.
-   **Different filename:** the file is treated as a separate document,
    even if its contents match another PDF.

The change check uses the SHA-256 hash of the file bytes. Consequently,
even a change to PDF metadata or internal file structure can produce a
new version if the bytes differ.

> **Versioning is filename-based.** Keep the filename the same when you
> want an uploaded replacement to be treated as a new version of the
> same document.

------------------------------------------------------------------------

## 7. Run the Application

Run these commands from the project directory. Adjust the Python
environment and Compose commands to match your existing setup.

### Step 1 --- Activate your Python environment

For example:

``` bash
cd ~/hybrid-rag
source .venv/bin/activate
```

If your environment has a different name or is already active, use your
existing environment.

### Step 2 --- Start the vLLM service

If vLLM is configured in Docker Compose:

``` bash
docker compose up -d vllm
```

Check its logs if the service does not become ready:

``` bash
docker compose logs -f vllm
```

The application configuration in this setup expects the vLLM
OpenAI-compatible API at:

``` text
http://localhost:8001/v1
```

Make sure the model name configured in `main.py` matches the model
served by vLLM.

### Step 3 --- Start FastAPI

Use the command appropriate for your project. A common command is:

``` bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

For a development machine, `--reload` restarts the app when source files
change. It is generally not recommended for production.

### Step 4 --- Check the API

Open the interactive API documentation:

``` text
http://localhost:8000/docs
```

Check the health endpoint:

``` bash
curl http://localhost:8000/health
```

A healthy response confirms that the API process is responding. It does
not necessarily prove that every model or retrieval component is ready;
verify the service logs if requests fail.

------------------------------------------------------------------------

## 8. Upload a PDF with Postman

### Request

-   **Method:** `POST`
-   **URL:** `http://localhost:8000/documents/upload`
-   Open **Body → form-data**.
-   Add a key named exactly **`file`**.
-   Change the key type from **Text** to **File**.
-   Select the PDF from your computer.
-   Send the request.

Example:

  Postman field   Value
  --------------- ---------------------------------------------
  Method          `POST`
  URL             `http://localhost:8000/documents/upload`
  Body type       `form-data`
  Key             `file`
  Key type        `File`
  Value           Select a PDF, such as `employee_policy.pdf`

**Do not manually set the `Content-Type` header.** Postman should
generate the multipart boundary automatically.

The multipart field name must be `file`, because the FastAPI route
expects an uploaded file under that parameter. The PDF itself can have
any suitable filename.

------------------------------------------------------------------------

## 9. Ask Questions About Your PDFs

### Request

-   **Method:** `POST`
-   **URL:** `http://localhost:8000/ask`
-   **Body:** select `raw` → `JSON`.
-   Send a JSON object using the question field expected by your `/ask`
    route.

Example, if your endpoint expects a `question` field:

``` json
{
  "question": "How many days of annual leave do employees get?"
}
```

The app retrieves relevant chunks from the indexed PDFs, reranks them,
and supplies the selected context to the language model.

> The exact request fields and response shape are determined by the
> Pydantic model and route implementation in `main.py`. If your route
> uses different field names, follow the schema shown at
> `http://localhost:8000/docs`.

------------------------------------------------------------------------

## 10. List Indexed Documents

Use the document-list endpoint to check which documents are registered
and their version metadata.

``` http
GET http://localhost:8000/documents
```

Depending on the current route response, the result should include
document details such as the filename, version, and update timestamp.

This endpoint is useful for checking whether a file was registered and
whether a replacement upload increased its version.

------------------------------------------------------------------------

## 11. Local Persistence and Restart Behaviour

The system saves its searchable vector index and metadata locally so it
does not have to treat every application restart as a completely new
project.

``` mermaid
flowchart LR
    A[PDFs in pdfs/] --> B[Extract and index]
    B --> C[FAISS files in vector_db/]
    A --> D[Calculate file hashes]
    D --> E[pdf_manifest.json]
    D --> F[(document_metadata.db)]
    C --> G[Application restart]
    E --> G
    F --> G
    G --> H{Saved index matches current PDFs?}
    H -- Yes --> I[Load saved FAISS index]
    H -- No --> J[Rebuild index from current PDFs]
```

-   FAISS files preserve the semantic-search index.
-   The manifest records hashes used to check whether the index
    corresponds to the current PDF files.
-   SQLite preserves document version metadata.
-   BM25 is rebuilt from the current document chunks during startup.

**Back up the PDF directory, vector index, manifest, and SQLite database
together.** They represent related parts of the same local state.

The FAISS `index.pkl` file uses Python pickle deserialization. Only load
it from a location you trust; do not load an index supplied by an
untrusted source.

------------------------------------------------------------------------

## 12. API Summary

  -----------------------------------------------------------------------
  Method                  Endpoint                Purpose
  ----------------------- ----------------------- -----------------------
  `GET`                   `/health`               Check API health

  `POST`                  `/documents/upload`     Upload a PDF and index
                                                  it, or update its
                                                  version if its content
                                                  changed

  `GET`                   `/documents`            List registered
                                                  documents and their
                                                  metadata

  `POST`                  `/ask`                  Ask a question using
                                                  the retrieval and
                                                  generation pipeline
  -----------------------------------------------------------------------

Use `http://localhost:8000/docs` to see the actual schemas and responses
exposed by the running application.

------------------------------------------------------------------------

## 13. Troubleshooting

### The upload request returns `422 Unprocessable Entity`

-   Confirm that the form-data key is named exactly `file`.
-   Confirm that its type is **File**, not Text.
-   Confirm that a file is selected.
-   Confirm that you are sending a multipart form-data request.

### The uploaded file is not being reindexed

If the filename and SHA-256 hash match the registered document, the
application intentionally skips reindexing. Modify the PDF content and
upload it again with the same filename to test version incrementing.

### The `/ask` request fails

-   Confirm that FastAPI is running on port `8000`.
-   Confirm that vLLM is available at the configured base URL, expected
    here to be `http://localhost:8001/v1`.
-   Confirm that the model name configured in the app matches the model
    served by vLLM.
-   Review the FastAPI and vLLM logs for the underlying error.
-   Check the request schema in `/docs`.

### The application cannot start without any PDFs

The current index-loading implementation may expect at least one PDF to
exist in `pdfs/`. If that directory is empty and startup fails, add an
initial PDF or update the startup logic to support an empty index before
uploading the first document.

### The saved index is out of date

If the PDF files have changed outside the upload endpoint, the saved
index may no longer represent them. Check the manifest and application
logs, and let the application rebuild the index. Avoid manually editing
the FAISS files or manifest unless you know how they are produced.

------------------------------------------------------------------------

## 14. Design Notes and Current Scope

### What this project does

-   Provides a local PDF-based RAG API.
-   Combines semantic search and keyword search.
-   Fuses and reranks retrieval results before generation.
-   Tracks document versions using filename and SHA-256 content hashes.
-   Persists FAISS index files and SQLite metadata locally.

### What it does not automatically guarantee

-   **Perfect answers:** retrieval quality depends on the PDF text,
    chunking, retrieval settings, and model.
-   **Citations to exact PDF pages:** unless page metadata is explicitly
    included in the prompt and response logic, an answer should not be
    assumed to have page-accurate citations.
-   **Multi-user access control:** no authentication or authorization is
    described by this setup.
-   **Atomic updates across every file:** the PDF, FAISS files,
    manifest, and SQLite database are separate local resources. Back up
    the project before making major changes.
-   **Scanned-PDF OCR:** PDFs that contain only scanned images may need
    an OCR step before their contents can be retrieved.
-   **Production hardening:** configure authentication, request limits,
    error handling, logging, and deployment settings before exposing the
    API beyond a trusted local network.

------------------------------------------------------------------------

## 15. End-to-End Workflow

``` mermaid
flowchart TD
    A[Start vLLM] --> B[Start FastAPI]
    B --> C[Upload PDF using Postman]
    C --> D[Validate and hash document]
    D --> E[Track document version]
    E --> F[Extract and chunk PDF text]
    F --> G[Create embeddings]
    G --> H[Build FAISS and BM25 indexes]
    H --> I[Persist index and metadata]
    I --> J[Submit a question]
    J --> K[Hybrid retrieval]
    K --> L[RRF fusion and reranking]
    L --> M[Build context]
    M --> N[Generate answer with vLLM]
    N --> O[Return answer]
```

### In one sentence

**Upload PDFs → detect document changes → build local search indexes →
retrieve and rerank relevant passages → generate answers with a locally
served LLM.**

------------------------------------------------------------------------

## License and security

Add the appropriate license before publishing or distributing this
repository. Keep credentials, private documents, and sensitive
configuration out of version control. The local metadata database and
index may reveal information about the documents you process, so protect
them accordingly.
