"""
Production-grade multimodal image/chart extraction.
- OCR (PaddleOCR): ground truth for numbers and labels
- Vision LLM: structured JSON (type, x_axis, y_axis, data_points, trend)
- Merge: validate data points with OCR, output clean natural language
- Output is ALWAYS clean text for graph builder (no raw OCR/JSON)
"""
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

from config import get_settings

logger = logging.getLogger("graph_rag.services.image_processing")


def run_ocr(image_path: str) -> str:
    """Extract raw text via PaddleOCR. Returns empty if not installed."""
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        logger.warning("PaddleOCR not installed: pip install paddlepaddle paddleocr")
        return ""

    try:
        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        result = ocr.ocr(image_path, cls=True)
        if not result or not result[0]:
            return ""
        lines = []
        for line in result[0]:
            if line and len(line) >= 2:
                lines.append(line[1][0])
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"OCR failed: {e}")
        return ""


def run_vision_structured(image_path: str) -> dict | None:
    """
    Vision LLM returns structured JSON for charts:
    {type, x_axis, y_axis, data_points, trend}
    Returns None if vision unavailable or fails.
    """
    from services.llm_client import LLMClient

    client = LLMClient()
    try:
        raw = client.vision_describe_structured(image_path)
        if not raw or not raw.strip():
            return None
        # Parse JSON from response
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            return json.loads(m.group(0))
    except Exception as e:
        logger.debug(f"Vision structured failed: {e}")
    return None


def merge_ocr_vision(ocr_text: str, vision_data: dict | None, translate_fn: Callable[[str], str] | None = None) -> str:
    """
    Merge OCR (ground truth) + Vision (structure).
    Validate data points using OCR text.
    Output: clean natural language for graph builder.
    Example: "Revenue was 100 in 2022 and increased to 150 in 2023."
    """
    ocr = (ocr_text or "").strip()
    if translate_fn and ocr:
        ocr = translate_fn(ocr)

    if not vision_data:
        return ocr if ocr else ""

    # Extract structure from vision
    v_type = vision_data.get("type", "chart")
    x_axis = vision_data.get("x_axis", "")
    y_axis = vision_data.get("y_axis", "")
    data_points = vision_data.get("data_points", [])
    trend = vision_data.get("trend", "")

    # Validate data points against OCR (numbers in OCR are ground truth)
    ocr_numbers = set(re.findall(r"\b(\d+(?:\.\d+)?)\b", ocr))
    validated = []
    for dp in data_points if isinstance(data_points, list) else []:
        if isinstance(dp, dict):
            val = dp.get("value") or dp.get("y") or dp.get("value", "")
            label = dp.get("label") or dp.get("x") or dp.get("year", "")
            if str(val) in ocr_numbers or str(label) in ocr_numbers or not ocr_numbers:
                validated.append((label, val))
        elif isinstance(dp, (list, tuple)) and len(dp) >= 2:
            validated.append((str(dp[0]), str(dp[1])))

    # Build clean natural language
    parts = []
    if x_axis or y_axis:
        parts.append(f"{y_axis or 'Value'} by {x_axis or 'category'}.")
    for label, val in validated[:10]:
        parts.append(f"{val} in {label}.")
    if trend:
        parts.append(f"Trend: {trend}.")
    if ocr and not parts:
        parts.append(ocr)

    result = " ".join(parts).strip()
    return result if result else ocr


def extract_chart_from_page(
    file_path: str,
    page_num: int,
    output_dir: Path,
    translate_fn: Callable[[str], str] | None = None,
    emit: Callable[[str, str], None] | None = None,
) -> dict | None:
    """
    Fast chart extraction: render page to image, full OCR only. No Vision.
    Fallback: page vector text if OCR empty (PaddleOCR not installed).
    """
    import fitz

    try:
        with fitz.open(file_path) as doc:
            page = doc[page_num]
            page_text = page.get_text().strip()
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            img_path = output_dir / f"page_{page_num + 1}_chart.png"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            pix.save(str(img_path))

        ocr_text = run_ocr(str(img_path))
        merged = (ocr_text or "").strip()
        if translate_fn and merged:
            merged = translate_fn(merged)
        if not merged and page_text and len(page_text) > 20:
            merged = translate_fn(page_text[:2000]) if translate_fn else page_text[:2000]
        if not merged.strip():
            return None

        return {
            "type": "image",
            "source": f"page_{page_num + 1}_chart",
            "page": page_num + 1,
            "merged_context": merged.strip(),
            "original": ocr_text or "",
            "vision_context": "",
        }
    except Exception as e:
        logger.debug(f"Chart extraction page {page_num + 1}: {e}")
        return None
