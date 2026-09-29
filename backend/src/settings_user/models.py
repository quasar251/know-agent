"""User-scoped LLM / embedding config dataclasses + resolution helpers (v2-M1).

`resolve_user_*` returns either:
  - A populated UserXxxConfig dataclass (user has explicitly configured this side)
  - None (user has not configured it → call sites fall back to env via get_settings)

Provider taxonomy (user-facing):
  LLM:        `anthropic` | `openai-compat`
  Embedding:  `openai-compat` | `ollama`

`openai-compat` covers OpenAI, DeepSeek, SiliconFlow, Together, Groq, vLLM,
LMStudio, modern Ollama (/v1 path) — anything with a Bearer-auth `/embeddings`
or `/chat/completions` endpoint.

`ollama` (embedding only) means native protocol POST /api/embeddings — kept as
a distinct provider because (a) no auth and (b) different request/response shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from src.infra.crypto import decrypt

if TYPE_CHECKING:
    from src.auth.models import User


@dataclass(frozen=True, slots=True)
class UserLLMConfig:
    provider: str          # "anthropic" | "openai-compat"
    base_url: str          # e.g. "https://api.deepseek.com"
    api_key: str           # plaintext (decrypted)
    default_model: str
    complex_model: str     # if user didn't set, mirrors default_model


@dataclass(frozen=True, slots=True)
class UserEmbeddingConfig:
    provider: str          # "openai-compat" | "ollama"
    base_url: str          # e.g. "https://api.siliconflow.cn/v1"
    api_key: str           # plaintext (decrypted; empty for ollama)
    model: str
    dim: int


@dataclass(frozen=True, slots=True)
class UserRerankerConfig:
    """v3-M4: per-user cross-encoder reranker (opt-in, default off).

    Cohere-compatible /rerank endpoint shape (SiliconFlow / Cohere / Jina /
    self-hosted TEI). The presence of this dataclass at a call site means the
    user has both saved a config AND flipped the enable toggle on — see
    `resolve_user_reranker` for the two-gate check.
    """
    provider: str          # "siliconflow" | "cohere" | "openai-compat"
    base_url: str          # e.g. "https://api.siliconflow.cn/v1"
    api_key: str           # plaintext (decrypted; empty for self-hosted)
    model: str


def _llm_is_configured(u: "User") -> bool:
    return bool(
        u.llm_provider
        and u.llm_base_url
        and u.llm_api_key_enc
        and u.llm_default_model
    )


def _embedding_is_configured(u: "User") -> bool:
    return bool(
        u.embedding_provider
        and u.embedding_base_url
        and u.embedding_model
        and u.embedding_dim
    )


def _reranker_is_configured(u: "User") -> bool:
    """Reranker requires BOTH the enable toggle AND a fully populated config.

    api_key is allowed to be empty (for self-hosted openai-compat endpoints
    that don't enforce auth) — we treat presence of provider+base_url+model
    plus the toggle as sufficient.
    """
    if not bool(getattr(u, "reranker_enabled", False)):
        return False
    return bool(
        u.reranker_provider
        and u.reranker_base_url
        and u.reranker_model
    )


def resolve_user_llm(user: "User") -> Optional[UserLLMConfig]:
    """Return user's LLM config, or None to fall back to env."""
    if not _llm_is_configured(user):
        return None
    return UserLLMConfig(
        provider=user.llm_provider or "",
        base_url=(user.llm_base_url or "").rstrip("/"),
        api_key=decrypt(user.llm_api_key_enc or ""),
        default_model=user.llm_default_model or "",
        complex_model=user.llm_complex_model or user.llm_default_model or "",
    )


def resolve_env_llm() -> UserLLMConfig:
    """Synthesize a UserLLMConfig from env so a per-request ``model`` override
    applies even when the user hasn't saved a BYOK LLM config.

    Mirrors the env-fallback branches of ``get_client(cfg=None)`` /
    ``is_anthropic`` / ``with_cache_control`` exactly: env ``llm_provider`` of
    ``"anthropic"`` maps to the anthropic surface, anything else (``"deepseek"``)
    to the openai-compat surface with the DeepSeek creds. Provider taxonomy here
    is ``anthropic | openai-compat`` (unlike env's ``anthropic | deepseek``).
    """
    from src.settings import get_settings

    s = get_settings()
    if s.llm_provider == "anthropic":
        provider = "anthropic"
        base_url = s.anthropic_base_url
        api_key = s.anthropic_api_key
    else:  # deepseek / openai-compat
        provider = "openai-compat"
        base_url = s.deepseek_base_url
        api_key = s.deepseek_api_key
    return UserLLMConfig(
        provider=provider,
        base_url=(base_url or "").rstrip("/"),
        api_key=api_key,
        default_model=s.llm_default_model,
        complex_model=s.llm_complex_model or s.llm_default_model,
    )


def resolve_user_embedding(user: "User") -> Optional[UserEmbeddingConfig]:
    """Return user's embedding config, or None to fall back to env."""
    if not _embedding_is_configured(user):
        return None
    return UserEmbeddingConfig(
        provider=user.embedding_provider or "",
        base_url=(user.embedding_base_url or "").rstrip("/"),
        api_key=decrypt(user.embedding_api_key_enc or ""),
        model=user.embedding_model or "",
        dim=int(user.embedding_dim or 0),
    )


def resolve_user_reranker(user: "User") -> Optional[UserRerankerConfig]:
    """Return user's reranker config, or None when disabled / unconfigured.

    Unlike LLM and embedding, there is no env fallback — reranker is fully
    opt-in (default off). Callers that get None must skip reranking entirely.
    """
    if not _reranker_is_configured(user):
        return None
    enc = getattr(user, "reranker_api_key_enc", None) or ""
    return UserRerankerConfig(
        provider=user.reranker_provider or "",
        base_url=(user.reranker_base_url or "").rstrip("/"),
        api_key=decrypt(enc) if enc else "",
        model=user.reranker_model or "",
    )
