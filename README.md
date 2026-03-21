# Graph RAG - Production System

Graph-first RAG with multimodal PDF ingestion, Neo4j + FAISS hybrid retrieval, and LangGraph orchestration.

---

## Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **Neo4j AuraDB** (free tier) or Neo4j Desktop
- **Ollama** (for local Llama 3.1 8B)

---

## 1. Neo4j Setup

1. Go to [Neo4j Aura](https://neo4j.com/cloud/aura/) and create a free account.
2. Create a new **Free** database.
3. After creation, copy the connection details:
   - **URI**: `neo4j+s://xxxx.databases.neo4j.io`
   - **User**: `neo4j`
   - **Password**: (shown once; download or copy it)

4. Save these for your `.env` file.

---

## 2. Ollama (Local LLM)

1. Install [Ollama](https://ollama.com) for your OS.
2. Pull the model:
   ```bash
   ollama pull llama3.1:8b-instruct
   ```
3. Ensure Ollama is running (it usually starts automatically on port `11434`).

---

## 3. Backend Setup

```bash
cd backend
python -m venv venv
```

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

Create `.env` from the example:

```bash
copy .env.example .env   # Windows
# or
cp .env.example .env     # macOS/Linux
```

Edit `backend/.env` (or project root `.env`) with your values:

```env
# Neo4j (required)
NEO4J_URI=neo4j+s://YOUR_INSTANCE.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_actual_password

# LLM (default: local Ollama, no API key)
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b-instruct

# Optional
FRONTEND_URL=http://localhost:3001
```

Start the backend:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## 4. Frontend Setup

In a new terminal:

```bash
cd frontend
npm install
```

Create `frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

Start the frontend:

```bash
npm run dev
```

---

## 5. Run the Application

**Terminal 1 – Backend:**
```powershell
cd backend
.\venv\Scripts\Activate.ps1
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 – Frontend:**
```powershell
cd frontend
npm run dev
```

Then open **http://localhost:3001** in your browser.

---

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `NEO4J_URI` | Yes | Neo4j connection URI (e.g. `neo4j+s://xxx.databases.neo4j.io`) |
| `NEO4J_USER` | Yes | Neo4j username (usually `neo4j`) |
| `NEO4J_PASSWORD` | Yes | Neo4j password |
| `LLM_PROVIDER` | No | `ollama` (default) \| `gemini` \| `openai` |
| `OLLAMA_BASE_URL` | No | Ollama URL (default: `http://localhost:11434`) |
| `OLLAMA_MODEL` | No | Model name (default: `llama3.1:8b-instruct`) |
| `FRONTEND_URL` | No | CORS origin (default: `http://localhost:3001`) |

---

## Usage

1. **Pipeline dashboard** (`/dashboard`): Upload a PDF; watch **two SSE feeds** — full job state (`/api/upload/status/{job_id}/stream`) plus **granular events** (`/api/stream/{job_id}`): `extraction_text`, `chunk_created`, `entity_created`, `relationship_created`, `graph_update`, `done`.
2. **Classic upload** (`/upload`): Upload a PDF; after completion, the graph is **explicitly pushed to Neo4j** and the graph visualization (nodes + edges) is loaded from the database.
3. **Query**: `/query` — scoped to `document_id` or **Search all** for cumulative results; response includes `retrieval_graph` (nodes + links).

---

## Test Neo4j Upload

Verify that chunks, entities, and relationships are stored in Neo4j:

```powershell
cd backend
.\venv\Scripts\python.exe scripts\test_neo4j_upload.py
```

Expected output: `PASS` with counts for chunks, entities, MENTIONS edges, and Entity-Entity relationships (e.g. HAS_VALUE, USED_FOR). The script creates a test document, verifies it in Neo4j, then cleans up.

---

## Architecture

- **Frontend**: Next.js 14 (App Router), Tailwind
- **Backend**: FastAPI, Neo4j, FAISS, SentenceTransformers, PaddleOCR, LangGraph
- **Flow**: PDF → Extract (text/table/image) → Chunk → Graph + FAISS → Query (graph-first)

---

## Project Structure

```
rag_cts/
├── backend/
│   ├── api/endpoints/     # upload, query, status
│   ├── config/            # settings, logging
│   ├── services/          # extraction, chunking, graph_builder, vector_store, query_engine, langgraph_agent
│   └── main.py
├── frontend/
│   ├── app/               # pages: upload, query
│   ├── components/        # GraphVisualization
│   └── lib/               # api client
└── README.md
```
