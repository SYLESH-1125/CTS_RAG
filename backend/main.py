"""
FastAPI application entry point.
Graph RAG API - Production grade.
"""
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import get_settings
from config.logging_config import setup_logging
from api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    settings = get_settings()
    setup_logging(settings.log_level)
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.faiss_index_dir).mkdir(parents=True, exist_ok=True)
    yield
    # Cleanup if needed


app = FastAPI(
    title="Graph RAG API",
    description="Production-grade Graph RAG with multimodal ingestion",
    version="1.0.0",
    lifespan=lifespan,
)

settings = get_settings()
Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
Path(settings.faiss_index_dir).mkdir(parents=True, exist_ok=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount uploads for serving extracted assets
app.mount("/uploads", StaticFiles(directory=settings.upload_dir), name="uploads")

app.include_router(router, prefix="/api", tags=["graph-rag"])


@app.get("/")
def root():
    """Root redirect to docs."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs", status_code=302)


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "graph-rag"}
