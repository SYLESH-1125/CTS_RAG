# CTS Graph RAG

A **Graph-first Retrieval-Augmented Generation (RAG)** system for PDF documents. It extracts multimodal content from PDFs, builds a knowledge graph in Neo4j, indexes text in FAISS for semantic search, and answers queries using graph context combined with an LLM.

![CTS Graph RAG - System Architecture](images/arch.png)

---

## Project Description

CTS Graph RAG ingests PDF documents and processes them through a multi-stage pipeline: extraction of text, tables, and images; content deduplication; semantic chunking; entity and relationship extraction into a Neo4j knowledge graph; and vector indexing in FAISS. Queries are answered by retrieving relevant chunks via FAISS, expanding context through Neo4j graph traversal, and generating answers with an LLM. Multiple documents can be uploaded cumulatively; each document is isolated in the graph by `document_id` and queries can target a single document or search across all uploaded documents.

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Frontend** | Next.js 14 (App Router) | Single-page app, upload and query UI |
| | React 18, TypeScript | Components and type safety |
| | Tailwind CSS | Styling |
| | Axios | HTTP client |
| | react-force-graph-2d | Graph visualization |
| | Zustand, Framer Motion | State and animations |
| **Backend** | FastAPI | REST API, SSE streaming |
| | Uvicorn | ASGI server |
| **PDF Processing** | PyMuPDF (fitz) | PDF parsing, text extraction |
| | pdfplumber | Table and layout extraction |
| | PaddleOCR (optional) | OCR for images |
| **Embeddings & ML** | Sentence-Transformers (MiniLM) | Text embeddings |
| | NumPy | Numerical operations |
| **Graph Database** | Neo4j (Aura) | Knowledge graph storage, Chunk/Entity/Relation nodes |
| **Vector Store** | FAISS | Semantic similarity search over chunks |
| **LLM** | Ollama (Llama 3.1 8B) | Answer generation; also supports Gemini, OpenAI |
| | LangGraph | Optional agent orchestration |
| **NLP & Translation** | langdetect | Language detection |
| | deep-translator | Query and content translation |

![End-to-End Pipeline Flow](images/flow%20dia.png)

---

## Pipeline Phases

### Phase 1: Extraction
- **Input:** PDF file
- **Output:** `text_units`, `table_units`, `image_units`
- **Process:** PyMuPDF and pdfplumber extract text and tables. Images are processed with OCR (PaddleOCR, optional) and optional vision LLM for descriptions. Content is normalized to English (when translation is enabled).
- **Deliverables:** Structured content per page with source, page number, and type (text, table, image).

![Phase 1 - Extraction Dashboard](images/Screenshot%202026-03-21%20173431.png)

### Phase 2: Chunking
- **Input:** Extraction output
- **Output:** List of chunks with `chunk_id`, `text`, `source`, `page`, `coherence_score`
- **Process:**
  1. **Content deduplication:** Overlapping content from PDF text + OCR + vision is merged (cosine similarity threshold).
  2. **Structural chunking:** Split by sections, tables, image blocks, and page boundaries.
  3. **Semantic chunking:** Further splits using sentence-level similarity (MiniLM) to keep coherent paragraphs.
  4. **Entity-aware refinement:** Splits blocks containing unrelated entity groups.
  5. **Coherence scoring:** Drops low-coherence chunks below a configurable threshold.
- **Deliverables:** Graph-aware chunks suitable for embedding and graph extraction.

![Phase 2 - Chunking Visualization](images/Screenshot%202026-03-21%20173443.png)

### Phase 3: Graph Build (Neo4j)
- **Input:** Chunks from Phase 2
- **Output:** Chunk nodes, Entity nodes, MENTIONS edges (Chunk→Entity), Entity–Entity relationships
- **Process:**
  1. Per-chunk entity extraction (NER/regex + optional LLM when confidence is low).
  2. Relationship extraction between entities (HAS_VALUE, RELATED_TO, USED_FOR, etc.).
  3. Two-stage Neo4j writes: (a) Chunk + Entity + MENTIONS, (b) batched Entity–Entity relationships.
  4. Optional community assignment (union-find) for graph structure.
- **Deliverables:** Neo4j graph with `document_id` per node for multi-document isolation; live streaming snapshots for UI.

![Phase 3 - Graph Build (Live)](images/Screenshot%202026-03-21%20173507.png)

### Phase 4: Vector Store (FAISS)
- **Input:** Chunks with `document_id`
- **Output:** FAISS index and chunk_id→metadata mapping
- **Process:** Each chunk is embedded with Sentence-Transformers and added to the FAISS index (cumulative across uploads). `document_id` is stored for scoped search.
- **Deliverables:** Semantic search returns top-k `chunk_id`s for retrieval.

### Phase 5: Query (RAG)
- **Input:** Natural-language question, optional `document_id`, optional “search all” flag
- **Output:** Answer, citations, retrieval graph, confidence
- **Process:**
  1. Query normalization (detect language, translate to English if needed).
  2. FAISS semantic search → top-k `chunk_id`s (scoped by `document_id` or all).
  3. Neo4j expansion: `MATCH (c:Chunk)-[:MENTIONS]->(e) WHERE c.id IN $chunk_ids`; optionally traverse Entity–Entity edges.
  4. Merge chunk context and graph context.
  5. LLM generates answer with citations.
- **Deliverables:** Answer text, source chunks, and graph subgraph for visualization.

![Phase 5 - Query Reasoning](images/Screenshot%202026-03-21%20173518.png)

#### LangGraph Workflow (Optional Orchestration)
When using LangGraph for agent orchestration, the state machine coordinates ingestion, chunking, vector indexing, Cypher generation, and graph/vector retrieval with fallback paths.

![LangGraph State Machine](images/langgrpah.png)

---

## Project Structure

```
rag_cts/
├── backend/
│   ├── api/endpoints/      # upload, query, status, graph, stream
│   ├── config/             # settings, logging
│   ├── services/
│   │   ├── extraction.py   # Phase 1
│   │   ├── content_deduplication.py
│   │   ├── chunking.py    # Phase 2
│   │   ├── graph_builder.py # Phase 3
│   │   ├── vector_store.py  # Phase 4
│   │   ├── query_engine.py  # Phase 5
│   │   ├── chunk_processor.py
│   │   ├── embedder.py
│   │   └── ...
│   └── main.py
├── frontend/
│   ├── app/
│   │   ├── upload/        # Classic upload + graph viz
│   │   ├── query/         # RAG query UI
│   │   ├── dashboard/     # Pipeline dashboard
│   │   └── page.tsx       # Home
│   ├── components/        # GraphVisualization, pipeline components
│   └── lib/api.ts
├── images/               # Diagrams and screenshots for documentation
└── README.md
```

---

## Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **Neo4j AuraDB** (free tier) or Neo4j Desktop
- **Ollama** (for local Llama 3.1 8B)

---

## Setup

### 1. Neo4j
Create a free database at [Neo4j Aura](https://neo4j.com/cloud/aura/). Save URI, username, and password.

### 2. Ollama
```bash
ollama pull llama3.1:8b-instruct
```

### 3. Backend
```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate  |  macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env   # Edit with Neo4j credentials
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4. Frontend
```bash
cd frontend
npm install
# Create frontend/.env.local with: NEXT_PUBLIC_API_URL=http://localhost:8000/api
npm run dev
```

Open **http://localhost:3001**.

---

## Usage

![Graph RAG - Home](images/Screenshot%202026-03-21%20173534.png)

| Route | Description |
|-------|-------------|
| `/upload` | Upload PDF → pipeline runs → graph pushed to Neo4j → force-directed visualization |
| `/query` | Ask questions; optional document scope or “Search all documents” |
| `/dashboard` | Pipeline dashboard with granular SSE events (extraction, chunking, graph, etc.) |

![Query Interface](images/Screenshot%202026-03-21%20173550.png)

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload/pdf` | POST | Upload PDF, returns `job_id` |
| `/api/upload/status/{job_id}/stream` | GET | SSE stream of job status |
| `/api/upload/push-neo4j/{job_id}` | POST | Explicit push of graph to Neo4j |
| `/api/upload/neo4j-stats` | GET | Total Chunks, Entities, Relationships in Neo4j |
| `/api/query/` | POST | RAG query with `query`, `document_id`, `search_all` |
| `/api/graph/document/{document_id}` | GET | Graph nodes and edges for visualization |

---

## Tests

```powershell
cd backend
.\venv\Scripts\python.exe scripts\test_neo4j_upload.py
.\venv\Scripts\python.exe scripts\test_neo4j_endpoints.py
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NEO4J_URI` | Yes | Neo4j connection URI |
| `NEO4J_USERNAME` | Yes | Neo4j username |
| `NEO4J_PASSWORD` | Yes | Neo4j password |
| `LLM_PROVIDER` | No | `ollama` (default) \| `gemini` \| `openai` |
| `OLLAMA_BASE_URL` | No | Default `http://localhost:11434` |
| `OLLAMA_MODEL` | No | Default `llama3.1:8b-instruct` |
| `EMBEDDING_MODEL` | No | Default `sentence-transformers/all-MiniLM-L6-v2` |
| `FRONTEND_URL` | No | CORS origin, default `http://localhost:3001` |
