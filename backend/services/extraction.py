"""
Phase 1: PDF Extraction Service.
Extracts text, tables, and images (with OCR + Vision) from PDFs.
All content normalized to English.
"""
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any, Callable

StreamCallback = Callable[[str, dict[str, Any]], None]

OnLogCallback = Callable[[str, str], None]
OnProgressCallback = Callable[[str, dict], None]
import fitz  # PyMuPDF
import pdfplumber
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator

from config import get_settings

logger = logging.getLogger("graph_rag.services.extraction")


class ExtractionService:
    """
    Multimodal PDF extraction:
    1. Text via pdfplumber + PyMuPDF
    2. Tables via pdfplumber → 2D array → translate → LLM context
    3. Images via PaddleOCR + Vision LLM → merge
    """
    
    def __init__(self):
        self.settings = get_settings()
    
    def extract_pdf(
        self,
        file_path: str,
        on_log: OnLogCallback | None = None,
        on_progress: OnProgressCallback | None = None,
        on_stream: StreamCallback | None = None,
    ) -> dict[str, Any]:
        """
        Full extraction pipeline.
        Returns structured extraction with logs for UI.
        When on_log(step, message) is provided, emits logs in real-time for streaming.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        def emit(step: str, message: str):
            if on_log:
                on_log(step, message)

        def prog(step: str, payload: dict):
            if on_progress:
                on_progress(step, payload)

        result = {
            "text_units": [],
            "table_units": [],
            "image_units": [],
            "logs": [],
            "metadata": {"filename": path.name, "pages": 0},
        }

        # Get page count
        with fitz.open(file_path) as doc:
            result["metadata"]["pages"] = len(doc)

        msg = f"Processing {path.name}, {result['metadata']['pages']} pages"
        result["logs"].append({"step": "init", "message": msg})
        emit("init", msg)
        prog("extraction", {"step": "init", "total": result["metadata"]["pages"], "detail": path.name})

        # 1. Text extraction
        emit("text", "Extracting text...")
        prog("extraction", {"step": "text", "current": 0, "total": result["metadata"]["pages"], "detail": "Extracting text…"})
        text_units = self._extract_text(
            file_path,
            emit=emit,
            num_pages=result["metadata"]["pages"],
            on_page=lambda cur, tot: prog("extraction", {"step": "text", "current": cur, "total": tot, "detail": f"Page {cur}/{tot}"}),
            on_stream=on_stream,
        )
        result["text_units"] = text_units
        msg = f"Extracted {len(text_units)} text blocks"
        result["logs"].append({"step": "text", "message": msg})
        emit("text", msg)
        prog("extraction", {"step": "text_done", "current": result["metadata"]["pages"], "total": result["metadata"]["pages"], "detail": f"{len(text_units)} text blocks"})

        # 2. Table extraction
        emit("table", "Extracting tables...")
        prog("extraction", {"step": "table", "current": 0, "total": result["metadata"]["pages"], "detail": "Extracting tables…"})
        table_units = self._extract_tables(file_path, emit=emit, on_stream=on_stream)
        result["table_units"] = table_units
        msg = f"Extracted {len(table_units)} tables"
        result["logs"].append({"step": "table", "message": msg})
        emit("table", msg)
        prog("extraction", {"step": "table_done", "detail": f"{len(table_units)} tables"})

        # 3. Image extraction (PaddleOCR primary + Vision supplementary)
        emit("image", "Extracting images (OCR + Vision)...")
        prog("extraction", {"step": "image", "detail": "Images / embedded…"})
        image_units = self._extract_images(file_path, path.parent, emit=emit, on_stream=on_stream)
        result["image_units"] = image_units

        # 4. Chart extraction (OCR only - fast, no Vision)
        if getattr(self.settings, "extract_charts", True):
            emit("chart", "Extracting charts from pages...")
            n_pages = result["metadata"]["pages"]
            prog("extraction", {"step": "chart", "current": 0, "total": n_pages, "detail": "Charts (OCR)…"})
            chart_units = self._extract_charts(
                file_path,
                path.parent,
                n_pages,
                emit=emit,
                on_page=lambda cur, tot: prog("extraction", {"step": "chart", "current": cur, "total": tot, "detail": f"Chart page {cur}/{tot}"}),
                on_stream=on_stream,
            )
            result["image_units"].extend(chart_units)

        msg = f"Extracted {len(result['image_units'])} images/charts"
        result["logs"].append({"step": "image", "message": msg})
        emit("image", msg)
        prog("extraction", {"step": "done", "detail": f"{len(result['image_units'])} images/charts"})

        return result
    
    def _extract_text(
        self,
        file_path: str,
        emit: Callable[[str, str], None] | None = None,
        num_pages: int = 0,
        on_page: Callable[[int, int], None] | None = None,
        on_stream: StreamCallback | None = None,
    ) -> list[dict]:
        """Extract text using PyMuPDF (single reader, no file lock). Translate if enabled."""
        units = []

        def log(msg: str):
            if emit:
                emit("text", msg)

        # Use fitz only - avoids dual file open / Windows lock
        with fitz.open(file_path) as doc:
            total = num_pages or len(doc)
            for i in range(len(doc)):
                log(f"Page {i + 1}/{total}...")
                if on_page:
                    on_page(i + 1, total)
                page = doc[i]
                text = page.get_text()
                if not text or not text.strip():
                    continue

                original = text.strip()
                log(f"Processing page {i + 1}...")
                translated = self._translate_if_needed(original)

                unit = {
                    "type": "text",
                    "original": original,
                    "translated": translated,
                    "page": i + 1,
                    "source": f"page_{i + 1}",
                }
                units.append(unit)
                if on_stream:
                    on_stream(
                        "extraction_text",
                        {
                            "page": unit["page"],
                            "source": unit["source"],
                            "text": (translated or original)[:2000],
                            "original_preview": original[:600],
                        },
                    )

        return units
    
    def _extract_tables(
        self,
        file_path: str,
        emit: Callable[[str, str], None] | None = None,
        on_stream: StreamCallback | None = None,
    ) -> list[dict]:
        """Extract tables as 2D arrays, translate, then LLM for context."""
        units = []

        def log(msg: str):
            if emit:
                emit("table", msg)

        with pdfplumber.open(file_path) as pdf:
            pages = list(pdf.pages)
            for i, page in enumerate(pages):
                log(f"Page {i + 1}/{len(pages)}...")
                tables = page.extract_tables()
                if not tables:
                    continue

                for t_idx, table in enumerate(tables):
                    if not table:
                        continue
                    
                    # 2D array to natural language
                    table_text = self._table_to_text(table)
                    translated = self._translate_if_needed(table_text)
                    context_ready = translated if getattr(self.settings, "skip_table_llm", True) else self._table_to_context(translated)
                    
                    row = {
                        "type": "table",
                        "original": table_text,
                        "translated": translated,
                        "context_ready": context_ready,
                        "page": i + 1,
                        "source": f"page_{i + 1}_table_{t_idx}",
                        "raw_table": table,
                    }
                    units.append(row)
                    if on_stream:
                        on_stream(
                            "extraction_table",
                            {
                                "page": row["page"],
                                "source": row["source"],
                                "rows": min(12, len(table)),
                                "cols": min(12, max(len(r) for r in table) if table else 0),
                                "grid_preview": [[str(c or "")[:40] for c in (r[:8] or [])] for r in (table[:6] or [])],
                                "original": table_text[:1500],
                                "translated": (translated or table_text)[:1500],
                                "context_ready": (context_ready or translated or table_text)[:1500],
                            },
                        )

        return units
    
    def _extract_images(
        self,
        file_path: str,
        output_dir: Path,
        emit: Callable[[str, str], None] | None = None,
        on_stream: StreamCallback | None = None,
    ) -> list[dict]:
        """Extract embedded images: OCR only (no Vision - fast path)."""
        from services.image_processing import run_ocr

        units = []
        def log(msg: str):
            if emit:
                emit("image", msg)

        with fitz.open(file_path) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                images = page.get_images()
                for img_idx, img in enumerate(images):
                    log(f"Page {page_num + 1} img {img_idx + 1}...")
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    img_path = output_dir / f"page_{page_num + 1}_img_{img_idx}.{base_image['ext']}"
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                    img_path.write_bytes(base_image["image"])

                    ocr_text = run_ocr(str(img_path))
                    if ocr_text.strip():
                        translated = self._translate_if_needed(ocr_text)
                        row = {
                            "type": "image",
                            "original": ocr_text,
                            "merged_context": translated or ocr_text,
                            "ocr_translated": translated or ocr_text,
                            "page": page_num + 1,
                            "source": f"page_{page_num + 1}_img_{img_idx}",
                        }
                        units.append(row)
                        if on_stream:
                            on_stream(
                                "extraction_image",
                                {
                                    "page": row["page"],
                                    "source": row["source"],
                                    "original": ocr_text[:1200],
                                    "ocr_text": ocr_text[:1200],
                                    "merged_context": (translated or ocr_text)[:1200],
                                    "ocr_translated": (translated or ocr_text)[:1200],
                                    "path": str(img_path).replace("\\", "/"),
                                },
                            )
        return units

    def _extract_charts(
        self,
        file_path: str,
        output_dir: Path,
        num_pages: int,
        emit: Callable[[str, str], None] | None = None,
        on_page: Callable[[int, int], None] | None = None,
        on_stream: StreamCallback | None = None,
    ) -> list[dict]:
        """Extract charts from rendered pages. OCR + Vision merge, clean output."""
        from services.image_processing import extract_chart_from_page

        units = []
        translate_fn = self._translate_if_needed if getattr(self.settings, "enable_translation", True) else None
        for page_num in range(num_pages):
            if on_page:
                on_page(page_num + 1, num_pages)
            if emit:
                emit("chart", f"Page {page_num + 1}/{num_pages}...")
            result = extract_chart_from_page(file_path, page_num, output_dir, translate_fn, emit)
            if result:
                units.append(result)
                if on_stream:
                    on_stream(
                        "extraction_image",
                        {
                            "page": result.get("page", page_num + 1),
                            "source": result.get("source", f"chart_p{page_num + 1}"),
                            "ocr_text": (result.get("original") or "")[:800],
                            "merged_context": (result.get("merged_context") or "")[:1200],
                            "kind": "chart",
                        },
                    )
        return units

    def _translate_if_needed(self, text: str, timeout_sec: int = 5) -> str:
        """Translate to English if not English. Disabled by default to avoid hangs."""
        if not text or not text.strip():
            return text
        if not getattr(self.settings, "enable_translation", True):
            return text

        # Fast-path: mostly ASCII => assume English, skip translation
        ascii_count = sum(1 for c in text if ord(c) < 128)
        if ascii_count / max(1, len(text)) > 0.95:
            return text

        text = text[:3000] if len(text) > 3000 else text

        def _do_translate() -> str:
            lang = detect(text)
            if lang == "en":
                return text
            return GoogleTranslator(source=lang, target="en").translate(text)

        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_do_translate)
                return future.result(timeout=timeout_sec)
        except FuturesTimeoutError:
            logger.warning("Translation timed out, using original text")
            return text
        except (LangDetectException, Exception) as e:
            logger.warning(f"Translation skip: {e}")
            return text
    
    def _table_to_text(self, table: list[list]) -> str:
        """Convert 2D table to readable text."""
        lines = []
        for row in table:
            cells = [str(c or "").strip() for c in row]
            lines.append(" | ".join(cells))
        return "\n".join(lines)
    
    def _table_to_context(self, table_text: str) -> str:
        """Use LLM to convert table to context-ready natural language."""
        from services.llm_client import LLMClient
        client = LLMClient()
        prompt = f"""Convert this table data into clear, context-ready natural language. 
Preserve all data and relationships. Output only the converted text.

Table:
{table_text}
"""
        try:
            result = client.generate(prompt, max_tokens=1024)
            return result.strip() if result else table_text
        except Exception as e:
            logger.warning(f"Table context LLM failed: {e}")
            return table_text
    
