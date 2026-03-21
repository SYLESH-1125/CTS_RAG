"""
LLM client - supports Ollama (local), Gemini, OpenAI.
Ollama: resolves model names against /api/tags, works with llama3.1:8b, llama3.2-vision, etc.
"""
import logging
from pathlib import Path

import httpx

from config import get_settings

logger = logging.getLogger("graph_rag.services.llm_client")


class LLMClient:
    """Unified LLM interface for text and vision."""

    def __init__(self):
        self.settings = get_settings()
        self._client = None
        self._ollama_models_cache = None

    def _get_client(self):
        """Lazy init for API-based clients."""
        if self._client is None:
            if self.settings.llm_provider.lower() == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=self.settings.gemini_api_key)
                self._client = genai.GenerativeModel("gemini-1.5-flash")
            elif self.settings.llm_provider.lower() == "openai":
                from openai import OpenAI
                self._client = OpenAI(api_key=self.settings.openai_api_key)
        return self._client

    def _ollama_list_models(self) -> list[str]:
        """Fetch available Ollama models. Cached per instance."""
        if self._ollama_models_cache is not None:
            return self._ollama_models_cache
        try:
            r = httpx.get(
                f"{self.settings.ollama_base_url.rstrip('/')}/api/tags",
                timeout=5.0,
            )
            if r.status_code == 200:
                models = [m.get("name", "") for m in r.json().get("models", []) if m.get("name")]
                self._ollama_models_cache = models
                return models
        except Exception as e:
            logger.debug(f"Ollama tags failed: {e}")
        self._ollama_models_cache = []
        return []

    def _ollama_resolve_text_model(self) -> str | None:
        """Resolve text model: use configured or find best match from available."""
        configured = (self.settings.ollama_model or "").strip()
        available = self._ollama_list_models()
        if not available:
            return configured or "llama3.1:8b"
        if configured and configured in available:
            return configured
        base = configured.split(":")[0] if ":" in configured else configured
        base = base.replace("-instruct", "").replace("-chat", "")
        for m in available:
            if m.startswith(base) and "vision" not in m.lower():
                return m
        return available[0] if available else None

    def _ollama_resolve_vision_model(self) -> str | None:
        """Resolve vision model. Auto-detect llama3.2-vision when OLLAMA_VISION_MODEL empty."""
        configured = (self.settings.ollama_vision_model or "").strip()
        available = self._ollama_list_models()
        if configured and configured in available:
            return configured
        for m in available:
            if "vision" in m.lower() or "llava" in m.lower():
                return m
        return configured if configured else None

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate text completion. Tries Ollama first; falls back to Gemini 2.5 Flash if Ollama fails."""
        provider = self.settings.llm_provider.lower()

        if provider == "ollama":
            out = self._ollama_generate(prompt, max_tokens)
            if out:
                return out
            # Fallback: Gemini 2.5 Flash when Ollama unavailable
            return self._gemini_fallback_generate(prompt, max_tokens)
        elif provider == "gemini":
            model = self._get_client()
            response = model.generate_content(prompt)
            return response.text if response else ""
        else:
            client = self._get_client()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""

    def _gemini_fallback_generate(self, prompt: str, max_tokens: int) -> str:
        """Fallback to Gemini 2.5 Flash when Ollama is unavailable. Requires GEMINI_API_KEY."""
        if not self.settings.gemini_api_key:
            return ""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.settings.gemini_api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt, generation_config=genai.GenerationConfig(max_output_tokens=max_tokens))
            if response and response.text:
                logger.info("Used Gemini 2.5 Flash fallback (Ollama unavailable)")
                return response.text.strip()
        except Exception as e:
            logger.warning(f"Gemini fallback failed: {e}")
        return ""

    def _ollama_generate(self, prompt: str, max_tokens: int) -> str:
        """Generate via Ollama. Resolves model from /api/tags."""
        base = self.settings.ollama_base_url.rstrip("/")
        model = self._ollama_resolve_text_model()
        if not model:
            logger.warning("No Ollama text model available. Run: ollama pull llama3.1:8b")
            return ""

        for endpoint, payload in [
            ("/api/generate", {"model": model, "prompt": prompt, "stream": False, "options": {"num_predict": max_tokens}}),
            ("/api/chat", {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False, "options": {"num_predict": max_tokens}}),
        ]:
            try:
                r = httpx.post(f"{base}{endpoint}", json=payload, timeout=60.0)
                if r.status_code == 200:
                    if endpoint == "/api/generate":
                        return (r.json().get("response") or "").strip()
                    msg = r.json().get("message", {})
                    return (msg.get("content") or "").strip()
                if r.status_code == 404:
                    err = r.json().get("error", "")
                    logger.warning(f"Ollama model {model} not found: {err}")
            except Exception as e:
                logger.debug(f"Ollama {endpoint} failed: {e}")

        logger.warning("Ollama not available. Run: ollama serve && ollama pull llama3.1:8b")
        return ""

    def vision_describe(self, image_path: str) -> str:
        """Describe image. Ollama: uses vision model if configured, else empty."""
        path = Path(image_path)
        if not path.exists():
            return ""

        provider = self.settings.llm_provider.lower()

        if provider == "ollama":
            return self._ollama_vision(image_path)
        elif provider == "gemini":
            import google.generativeai as genai
            model = genai.GenerativeModel("gemini-1.5-flash")
            img = genai.upload_file(str(path))
            response = model.generate_content([
                "Describe this image in detail. Include structure, charts, graphs, labels, and key data.",
                img,
            ])
            return response.text if response else ""
        else:
            import base64
            with open(path, "rb") as f:
                b64 = base64.standard_b64encode(f.read()).decode()
            client = self._get_client()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in detail. Include structure, charts, graphs, labels, and key data."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        ],
                    }
                ],
                max_tokens=1024,
            )
            return response.choices[0].message.content or ""

    def _ollama_vision(self, image_path: str) -> str:
        """Vision via Ollama. Auto-detects llama3.2-vision when OLLAMA_VISION_MODEL not set."""
        vision_model = self._ollama_resolve_vision_model()
        if not vision_model:
            logger.debug("No Ollama vision model. Set OLLAMA_VISION_MODEL or run: ollama pull llama3.2-vision")
            return ""

        import base64
        with open(image_path, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode()

        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/generate"
        payload = {
            "model": vision_model,
            "prompt": "Describe this image in detail. Include structure, charts, graphs, labels, and key data.",
            "stream": False,
            "images": [b64],
        }
        try:
            r = httpx.post(url, json=payload, timeout=45.0)
            if r.status_code == 200:
                return (r.json().get("response") or "").strip()
            logger.warning(f"Ollama vision failed: {r.status_code} {r.text[:200]}")
        except Exception as e:
            logger.warning(f"Ollama vision failed: {e}")
        return ""

    def vision_describe_structured(self, image_path: str) -> str:
        """Vision that returns structured JSON for charts: {type, x_axis, y_axis, data_points, trend}."""
        path = Path(image_path)
        if not path.exists():
            return ""

        prompt = """Analyze this chart/graph image. Return ONLY valid JSON in this exact format:
{"type": "chart", "x_axis": "axis label", "y_axis": "axis label", "data_points": [{"label": "2022", "value": "100"}, {"label": "2023", "value": "150"}], "trend": "increasing"}
If not a chart, use type "figure" and describe in trend field. No other text."""

        provider = self.settings.llm_provider.lower()
        if provider == "ollama":
            return self._ollama_vision_structured(image_path, prompt)
        elif provider == "gemini":
            import google.generativeai as genai
            model = genai.GenerativeModel("gemini-1.5-flash")
            img = genai.upload_file(str(path))
            response = model.generate_content([prompt, img])
            return response.text if response else ""
        else:
            import base64
            with open(path, "rb") as f:
                b64 = base64.standard_b64encode(f.read()).decode()
            client = self._get_client()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ]},
                ],
                max_tokens=512,
            )
            return response.choices[0].message.content or ""

    def _ollama_vision_structured(self, image_path: str, prompt: str) -> str:
        """Vision structured JSON. Auto-detects llama3.2-vision."""
        vision_model = self._ollama_resolve_vision_model()
        if not vision_model:
            return ""
        import base64
        with open(image_path, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode()
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/generate"
        payload = {"model": vision_model, "prompt": prompt, "stream": False, "images": [b64]}
        try:
            r = httpx.post(url, json=payload, timeout=45.0)
            if r.status_code == 200:
                return (r.json().get("response") or "").strip()
        except Exception as e:
            logger.debug(f"Ollama vision structured failed: {e}")
        return ""
