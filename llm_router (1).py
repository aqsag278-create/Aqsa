"""
Tahqiq.ai — llm_router.py
Multi-provider LLM router with automatic fallback

Priority chain (configurable via LLM_PROVIDER env var):
  Grok  →  GPT-4o  →  Gemini 1.5 Pro  →  Claude Sonnet  →  OfflineFallback

Grok is now the PRIMARY provider (xAI API, OpenAI-compatible endpoint).

Environment variables
─────────────────────
  LLM_PROVIDER        primary choice: "grok" | "openai" | "gemini" | "anthropic" | "auto"
                      "auto" = try all in priority order  (default)
  XAI_API_KEY         required for Grok  (xAI API key)
  OPENAI_API_KEY      required for GPT-4o
  GEMINI_API_KEY      required for Gemini
  ANTHROPIC_API_KEY   required for Claude
  LLM_TIMEOUT         per-call timeout in seconds  (default: 30)
  LLM_MAX_RETRIES     retries per provider before moving on  (default: 2)

Grok vision support
───────────────────
  call_vision(image_b64, mime_type, prompt) uses grok-2-vision-1212
  Falls back to OCR-free text extraction if vision model unavailable.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger("tahqiq.llm_router")
logging.basicConfig(level=logging.INFO)

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

LLM_TIMEOUT     = int(os.environ.get("LLM_TIMEOUT",     "30"))
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "2"))
LLM_PROVIDER    = os.environ.get("LLM_PROVIDER", "auto").lower()

# ══════════════════════════════════════════════════════════════════════════════
# Provider health tracking  (in-memory, resets on restart)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _ProviderHealth:
    name:            str
    failures:        int   = 0
    last_failure_ts: float = 0.0
    cooldown_secs:   int   = 60

    def is_healthy(self) -> bool:
        if self.failures < 3:
            return True
        elapsed = time.time() - self.last_failure_ts
        if elapsed > self.cooldown_secs:
            self.failures = 0
            logger.info("Provider '%s' cooldown expired — re-enabling.", self.name)
            return True
        return False

    def record_failure(self) -> None:
        self.failures       += 1
        self.last_failure_ts = time.time()
        logger.warning("Provider '%s' failure #%d.", self.name, self.failures)

    def record_success(self) -> None:
        if self.failures:
            logger.info("Provider '%s' recovered after %d failure(s).",
                        self.name, self.failures)
        self.failures = 0


_health: dict[str, _ProviderHealth] = {}


def _get_health(name: str) -> _ProviderHealth:
    if name not in _health:
        _health[name] = _ProviderHealth(name=name)
    return _health[name]


# ══════════════════════════════════════════════════════════════════════════════
# Provider specs
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProviderSpec:
    name:        str
    env_key:     str
    full_model:  str
    fast_model:  str
    vision_model: Optional[str] = None    # model name for image tasks

    def has_key(self) -> bool:
        return bool(os.environ.get(self.env_key))

    def build(self, model: str, temperature: float) -> BaseChatModel:
        api_key = os.environ.get(self.env_key, "")

        if self.name == "grok":
            # Grok uses the OpenAI-compatible API with xAI base URL
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                openai_api_key=api_key,
                openai_api_base="https://api.x.ai/v1",
                request_timeout=LLM_TIMEOUT,
                max_retries=0,
            )

        if self.name == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                openai_api_key=api_key,
                request_timeout=LLM_TIMEOUT,
                max_retries=0,
            )

        if self.name == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                google_api_key=api_key,
                request_timeout=LLM_TIMEOUT,
                max_retries=0,
            )

        if self.name == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model,
                temperature=temperature,
                anthropic_api_key=api_key,
                timeout=LLM_TIMEOUT,
                max_retries=0,
            )

        raise ValueError(f"Unknown provider: {self.name}")


# Provider registry — priority order: Grok → GPT-4o → Gemini → Claude
_PROVIDERS: list[ProviderSpec] = [
    ProviderSpec(
        name="grok",
        env_key="XAI_API_KEY",
        full_model="grok-beta",
        fast_model="grok-beta",
        vision_model="grok-2-vision-1212",
    ),
    ProviderSpec(
        name="openai",
        env_key="OPENAI_API_KEY",
        full_model="gpt-4o",
        fast_model="gpt-4o-mini",
        vision_model="gpt-4o",
    ),
    ProviderSpec(
        name="gemini",
        env_key="GEMINI_API_KEY",
        full_model="gemini-1.5-pro",
        fast_model="gemini-1.5-flash",
        vision_model="gemini-1.5-pro",
    ),
    ProviderSpec(
        name="anthropic",
        env_key="ANTHROPIC_API_KEY",
        full_model="claude-sonnet-4-6",
        fast_model="claude-haiku-4-5-20251001",
        vision_model="claude-sonnet-4-6",
    ),
]

_PROVIDER_MAP: dict[str, ProviderSpec] = {p.name: p for p in _PROVIDERS}


# ══════════════════════════════════════════════════════════════════════════════
# Offline fallback stub
# ══════════════════════════════════════════════════════════════════════════════

class _OfflineLLM(BaseChatModel):
    """Zero-dependency stub. Returns a canned Urdish message."""

    @property
    def _llm_type(self) -> str:
        return "offline-stub"

    def _generate(self, messages: list[BaseMessage], **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_core.messages import AIMessage
        last = messages[-1].content if messages else ""
        stub = (
            "Offline mode — koi API key available nahi. "
            "Data-backed response generate nahi ho saka. "
            f"Query received: '{str(last)[:80]}'"
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=stub))])

    async def _agenerate(self, messages, **kwargs):
        return self._generate(messages, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# Core routing logic
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_provider_order() -> list[ProviderSpec]:
    if LLM_PROVIDER == "auto":
        keyed  = [p for p in _PROVIDERS if p.has_key()]
        no_key = [p for p in _PROVIDERS if not p.has_key()]
        return keyed + no_key
    primary = _PROVIDER_MAP.get(LLM_PROVIDER)
    if not primary:
        logger.warning("Unknown LLM_PROVIDER='%s' — using auto.", LLM_PROVIDER)
        return _resolve_provider_order()
    others = [p for p in _PROVIDERS if p.name != primary.name]
    return [primary] + others


def _build_llm(spec: ProviderSpec, model: str, temperature: float) -> BaseChatModel:
    health = _get_health(spec.name)
    if not health.is_healthy():
        raise RuntimeError(
            f"Provider '{spec.name}' in cooldown ({health.failures} failures)."
        )
    if not spec.has_key():
        raise RuntimeError(f"Provider '{spec.name}': {spec.env_key} not set.")
    try:
        llm = spec.build(model, temperature)
        health.record_success()
        return llm
    except Exception as exc:
        health.record_failure()
        raise


def _with_retry(llm_factory, provider_name: str) -> BaseChatModel:
    last_exc: Optional[Exception] = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            return llm_factory()
        except Exception as exc:
            last_exc = exc
            logger.warning("Provider '%s' attempt %d/%d failed: %s",
                           provider_name, attempt, LLM_MAX_RETRIES, exc)
            if attempt < LLM_MAX_RETRIES:
                time.sleep(2 ** attempt)
    raise last_exc


def _route(tier: str, temperature: float) -> BaseChatModel:
    providers = _resolve_provider_order()
    errors: list[str] = []
    for spec in providers:
        model = spec.full_model if tier == "full" else spec.fast_model
        try:
            llm = _with_retry(
                llm_factory   = lambda s=spec, m=model, t=temperature: _build_llm(s, m, t),
                provider_name = spec.name,
            )
            logger.info("LLM routed → provider='%s'  model='%s'  tier='%s'",
                        spec.name, model, tier)
            return llm
        except Exception as exc:
            errors.append(f"{spec.name}: {exc}")
            logger.warning("Skipping provider '%s': %s", spec.name, exc)

    logger.error("All LLM providers failed → offline stub.\nErrors: %s",
                 " | ".join(errors))
    return _OfflineLLM()


def get_llm(temperature: float = 0.2) -> BaseChatModel:
    """Return the best available full-capability LLM (Grok → GPT-4o → Gemini → Claude)."""
    return _route(tier="full", temperature=temperature)


def get_fast_llm(temperature: float = 0.0) -> BaseChatModel:
    """Return the best available fast/cheap LLM."""
    return _route(tier="fast", temperature=temperature)


# ══════════════════════════════════════════════════════════════════════════════
# Vision call  — for result card / marksheet OCR
# ══════════════════════════════════════════════════════════════════════════════

def call_vision(image_b64: str, mime_type: str, prompt: str) -> str:
    """
    Send an image to the best available vision-capable model.

    Tries providers in order: Grok → GPT-4o → Gemini → Claude.
    Each uses its own multimodal message format.

    Args:
        image_b64:  Base64-encoded image string (no data URI prefix needed).
        mime_type:  e.g. "image/jpeg", "image/png".
        prompt:     Instruction for the model (what to extract).

    Returns:
        Plain text response from the vision model.
    """
    for spec in _resolve_provider_order():
        if not spec.has_key() or not spec.vision_model:
            continue
        if not _get_health(spec.name).is_healthy():
            continue

        try:
            return _call_vision_provider(spec, image_b64, mime_type, prompt)
        except Exception as exc:
            _get_health(spec.name).record_failure()
            logger.warning("Vision provider '%s' failed: %s", spec.name, exc)

    logger.error("All vision providers failed — returning empty string.")
    return ""


def _call_vision_provider(
    spec: ProviderSpec,
    image_b64: str,
    mime_type: str,
    prompt: str,
) -> str:
    """Dispatch to provider-specific vision call."""
    api_key = os.environ.get(spec.env_key, "")

    if spec.name in ("grok", "openai"):
        # Both use OpenAI-style content blocks
        import openai
        base_url = "https://api.x.ai/v1" if spec.name == "grok" else None
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=spec.vision_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=512,
        )
        _get_health(spec.name).record_success()
        return resp.choices[0].message.content.strip()

    if spec.name == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        import base64, io
        from PIL import Image
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes))
        model = genai.GenerativeModel(spec.vision_model)
        resp = model.generate_content([prompt, img])
        _get_health(spec.name).record_success()
        return resp.text.strip()

    if spec.name == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=spec.vision_model,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64",
                                "media_type": mime_type,
                                "data": image_b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        _get_health(spec.name).record_success()
        return resp.content[0].text.strip()

    raise ValueError(f"No vision handler for provider: {spec.name}")


# ══════════════════════════════════════════════════════════════════════════════
# One-shot text helper
# ══════════════════════════════════════════════════════════════════════════════

def invoke(system_prompt: str, user_message: str,
           temperature: float = 0.2, fast: bool = False) -> str:
    llm    = get_fast_llm(temperature) if fast else get_llm(temperature)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ])
    chain  = prompt | llm | StrOutputParser()
    return chain.invoke({})


# ══════════════════════════════════════════════════════════════════════════════
# Status / diagnostics
# ══════════════════════════════════════════════════════════════════════════════

def get_router_status() -> dict:
    ordered     = _resolve_provider_order()
    active_name = "offline"
    for spec in ordered:
        if spec.has_key() and _get_health(spec.name).is_healthy():
            active_name = spec.name
            break
    return {
        "active_provider":  active_name,
        "provider_order":   [p.name for p in ordered],
        "llm_timeout_secs": LLM_TIMEOUT,
        "llm_max_retries":  LLM_MAX_RETRIES,
        "providers": {
            spec.name: {
                "key_present":   spec.has_key(),
                "healthy":       _get_health(spec.name).is_healthy(),
                "failures":      _get_health(spec.name).failures,
                "full_model":    spec.full_model,
                "fast_model":    spec.fast_model,
                "vision_model":  spec.vision_model,
            }
            for spec in _PROVIDERS
        },
    }


def reset_provider_health(provider_name: Optional[str] = None) -> dict:
    if provider_name:
        if provider_name in _health:
            _health[provider_name].failures         = 0
            _health[provider_name].last_failure_ts  = 0.0
        return {"reset": provider_name}
    for h in _health.values():
        h.failures         = 0
        h.last_failure_ts  = 0.0
    return {"reset": "all"}


# ══════════════════════════════════════════════════════════════════════════════
# Patch agent_logic._get_llm → router at startup
# ══════════════════════════════════════════════════════════════════════════════

def patch_agent_logic() -> None:
    """Monkey-patch agent_logic._get_llm so it uses the router automatically."""
    try:
        import agent_logic
        agent_logic._get_llm = get_llm
        logger.info("agent_logic._get_llm patched → llm_router.get_llm "
                    "(active: %s)", get_router_status()["active_provider"])
    except ImportError:
        logger.warning("agent_logic not found — patch skipped.")
