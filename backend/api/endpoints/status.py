"""
Status API - Health and system status.
"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/llm")
def health_llm():
    """Test LLM connectivity. Use this to verify Ollama is running."""
    from services.llm_client import LLMClient
    llm = LLMClient()
    provider = llm.settings.llm_provider.lower()
    models = llm._ollama_list_models() if provider == "ollama" else []
    ok = False
    test_resp = ""
    if provider == "ollama" and models:
        try:
            test_resp = (llm._ollama_generate("Reply with OK only.", 10) or "").strip()
            ok = bool(test_resp and "ok" in test_resp.lower())
        except Exception as e:
            test_resp = str(e)
    elif provider == "gemini":
        test_resp = (llm.generate("Reply with OK only.", 10) or "").strip()
        ok = bool(test_resp)
    return {
        "provider": provider,
        "ollama_reachable": bool(models),
        "ollama_models": models,
        "llm_working": ok,
        "test_response": test_resp[:80],
    }
