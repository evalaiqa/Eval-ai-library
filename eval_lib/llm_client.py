# llm_client.py
"""
Unified LLM client for eval-ai-library.

Architecture (since 0.6.0):
    - Standard cloud providers (OpenAI, Anthropic, Google, Azure, DeepSeek, Qwen,
      Zhipu, Mistral, Groq, Grok) → routed through LiteLLM via _litellm_chat_complete().
    - Ollama → kept on its native HTTP API (supports `think=false`, no /v1 suffix).
    - MLX → kept on its custom path with <think> tag stripping.
    - Custom (CustomLLMClient) → bypasses LiteLLM completely.

The public API — chat_complete(llm, messages, temperature) → (text, cost) — is
unchanged. Metric tests that mock chat_complete on a per-module basis are
unaffected by this refactor; only tests/test_providers_and_models.py was rewritten.

Cost calculation goes through eval_lib.model_catalog.get_cost_per_million(), which
checks our optional override table in price.py first and then falls back to
LiteLLM's litellm.model_cost (~2600 models, updated each LiteLLM release).
"""
import functools
import os
import re as _re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from typing import Optional, Tuple

import aiohttp
import litellm
import openai

from .model_catalog import get_cost_per_million
from .utils import JUDGE_SEED

# Make LiteLLM behave: silence its noisy loggers, drop unknown params instead of
# erroring (different providers accept different sets of options), and disable
# its own retry/fallback machinery (we layer that ourselves where needed).
litellm.suppress_debug_info = True
litellm.drop_params = True
litellm.set_verbose = False
litellm.num_retries = 0


class CustomLLMClient(ABC):
    """
    Base class for custom LLM clients.
    Inherit from this to create your own model implementations.

    Example:
        class MyCustomLLM(CustomLLMClient):
            async def chat_complete(self, messages, temperature):
                # Your implementation
                return response_text, cost

            def get_model_name(self):
                return "my-custom-model"
    """

    @abstractmethod
    async def chat_complete(
        self, messages: list[dict[str, str]], temperature: float
    ) -> tuple[str, Optional[float]]:
        """
        Generate a response for the given messages.

        Args:
            messages: List of message dicts [{"role": "user", "content": "..."}]
            temperature: Sampling temperature

        Returns:
            Tuple of (response_text, cost_in_usd)
        """
        pass

    async def get_embeddings(
        self, texts: list[str], model: str = "text-embedding-3-small"
    ) -> tuple[list[list[float]], Optional[float]]:
        """
        Get embeddings for texts (optional implementation).

        Args:
            texts: List of texts to embed
            model: Embedding model name

        Returns:
            Tuple of (embeddings_list, cost_in_usd)

        Raises:
            NotImplementedError: If custom client doesn't support embeddings
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support embeddings. "
            "Implement get_embeddings() method or use OpenAI for embeddings."
        )

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model name for logging/tracking purposes."""
        pass


class LLMConfigurationError(Exception):
    """Raised when LLM client configuration is missing or invalid."""

    pass


class Provider(str, Enum):
    """
    Only the providers that have *native* code paths in this module —
    everything else is handled as a plain string and routed through LiteLLM
    by _to_litellm_args(). This enum is intentionally tiny: it exists so
    native helpers in _HELPERS can be keyed off enum members, and so that
    legacy code importing `Provider.OLLAMA` etc. keeps working.

    LiteLLM-backed providers (OpenAI, Anthropic, Google, Azure, Bedrock,
    Vertex AI, Cohere, Together, OpenRouter, …) are NOT members here — they
    are passed through as raw string ids to LiteLLM. Upgrade `litellm` and
    new integrations become available automatically, with zero changes to
    this file.
    """

    OLLAMA = "ollama"
    MLX = "mlx"
    ZHIPU = "zhipu"
    CUSTOM = "custom"


# Legacy eval-lib provider ids → their canonical LiteLLM provider names.
# Kept as a *string* alias map rather than enum members so that user code
# still accepting the old prefixes ("google:gemini-...", "qwen:qwen-max",
# "grok:grok-2") continues to work without reviving the old first-class
# list. New code should use the LiteLLM names directly ("gemini:", "xai:",
# "dashscope:").
_LEGACY_PROVIDER_ALIASES: dict[str, str] = {
    "google": "gemini",
    "qwen": "dashscope",
    "grok": "xai",
}


def _coerce_provider(value: "str | Provider") -> "str | Provider":
    """
    Normalise a provider id.

    - Native ids (ollama/mlx/zhipu/custom) become `Provider` enum members so
      downstream dispatch can key on them.
    - Legacy eval-lib aliases (`google`, `qwen`, `grok`) are rewritten to their
      canonical LiteLLM names (`gemini`, `dashscope`, `xai`).
    - Any other string id is returned as-is: LiteLLM is asked to handle it
      directly by _to_litellm_args().
    """
    if isinstance(value, Provider):
        return value
    if value in _LEGACY_PROVIDER_ALIASES:
        return _LEGACY_PROVIDER_ALIASES[value]
    try:
        return Provider(value)
    except ValueError:
        return value


@dataclass(frozen=True, slots=True)
class LLMDescriptor:
    """'openai:gpt-4o'  →  provider=openai, model='gpt-4o'"""

    provider: "str | Provider"
    model: str

    @classmethod
    def parse(cls, spec: "str | Tuple[str, str] | LLMDescriptor") -> "LLMDescriptor":
        """
        Accept any of:
            - LLMDescriptor instance (passthrough)
            - (provider, model) tuple
            - "provider:model" string
            - bare "model" string — defaults provider to "openai" for
              historical compatibility with `chat_complete("gpt-4o", ...)`.
              Nothing *else* in the codebase special-cases OpenAI; this is
              the one convenience we keep.
        """
        if isinstance(spec, LLMDescriptor):
            return spec
        if isinstance(spec, tuple):
            provider, model = spec
            return cls(_coerce_provider(provider), model)
        try:
            provider, model = spec.split(":", 1)
        except ValueError:
            return cls("openai", spec)
        return cls(_coerce_provider(provider), model)

    def key(self) -> str:
        """Return a unique key for the LLM descriptor."""
        return f"{self.provider}:{self.model}"


# ---------------------------------------------------------------------------
# LiteLLM provider routing
# ---------------------------------------------------------------------------
#
# There is NO hand-maintained provider→prefix map. The provider id stored on
# an LLMDescriptor is used verbatim as the LiteLLM prefix ("cohere/command-r",
# "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0", "vertex_ai/gemini-1.5-pro"
# etc.). _coerce_provider() already rewrites the three legacy eval-lib aliases
# (google → gemini, qwen → dashscope, grok → xai) so the rest of this file
# doesn't need to know about them.


def _to_litellm_args(llm: LLMDescriptor) -> dict:
    """
    Translate an LLMDescriptor into kwargs accepted by litellm.acompletion().

    For everything LiteLLM knows about natively this is just
    `{"model": "<provider>/<model>"}`. Zhipu (GLM) needs the OpenAI-compatible
    base_url override because LiteLLM has no first-class Zhipu integration.
    """
    if llm.provider == Provider.ZHIPU:
        # GLM ships an OpenAI-compatible endpoint; route via the openai/ prefix
        # plus an explicit base_url + api_key from env.
        api_key = os.getenv("ZHIPU_API_KEY")
        if not api_key:
            raise LLMConfigurationError(
                "❌ Missing Zhipu GLM configuration!\n\n"
                "Environment variable 'ZHIPU_API_KEY' is not set.\n\n"
                "To fix this, set the environment variable:\n"
                "  export ZHIPU_API_KEY='your-api-key-here'"
            )
        return {
            "model": f"openai/{llm.model}",
            "api_key": api_key,
            "api_base": "https://open.bigmodel.cn/api/paas/v4",
        }

    # Everything else: treat the provider id (string or enum value) as the
    # LiteLLM prefix directly. Unknown strings work automatically — whatever
    # LiteLLM supports, we support.
    prefix = llm.provider.value if isinstance(llm.provider, Provider) else str(llm.provider)
    return {"model": f"{prefix}/{llm.model}"}


def _check_env_var(var_name: str, provider: str, required: bool = True) -> Optional[str]:
    """
    Check if environment variable is set and return its value.

    Used by the surviving non-LiteLLM paths (Ollama, MLX) to give friendly
    error messages when expected env vars are missing.

    Raises:
        LLMConfigurationError: If required variable is missing
    """
    value = os.getenv(var_name)
    if required and not value:
        raise LLMConfigurationError(
            f"❌ Missing {provider} configuration!\n\n"
            f"Environment variable '{var_name}' is not set.\n\n"
            f"To fix this, set the environment variable:\n"
            f"  export {var_name}='your-api-key-here'\n\n"
            f"Or add it to your .env file:\n"
            f"  {var_name}=your-api-key-here\n\n"
            f"📖 Documentation: https://github.com/meshkovQA/Eval-ai-library#environment-variables"
        )
    return value


@functools.cache
def _get_client(provider: Provider):
    """
    Get or create an HTTP client for providers that bypass LiteLLM.

    Only Ollama and MLX use this — every other provider goes through LiteLLM,
    which manages its own clients internally.

    Raises:
        LLMConfigurationError: If required configuration is missing
        ValueError: If provider is not supported here
    """
    if provider == Provider.OLLAMA:
        api_key = _check_env_var("OLLAMA_API_KEY", "Ollama", required=False) or "ollama"
        base_url = (
            _check_env_var("OLLAMA_API_BASE_URL", "Ollama", required=False)
            or "http://localhost:11434/v1"
        )

        return openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    if provider == Provider.MLX:
        base_url = os.getenv("MLX_API_BASE_URL", "http://localhost:8899/v1")
        return openai.AsyncOpenAI(
            api_key="mlx",
            base_url=base_url,
        )

    raise ValueError(
        f"Provider {provider.value} is routed through LiteLLM and does not have "
        "a dedicated client. Use chat_complete() instead of _get_client()."
    )


# ---------------------------------------------------------------------------
# LiteLLM-backed helper (handles all standard cloud providers)
# ---------------------------------------------------------------------------


async def _litellm_chat_complete(
    client,  # unused — kept for _HELPERS signature parity
    llm: LLMDescriptor,
    messages: list[dict[str, str]],
    temperature: float,
    *,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    extra_kwargs: Optional[dict] = None,
):
    """Universal chat completion via LiteLLM.

    Optional kwargs `api_key`, `api_base`, and `extra_kwargs` are forwarded
    directly to litellm.acompletion(). When set, they override anything that
    would otherwise be picked up from environment variables — this is the
    mechanism that lets multi-tenant callers (e.g. the eval-ai-platform
    llm-gateway service) inject per-user credentials without touching
    process-wide os.environ.

    Precedence rules:
        - explicit kwargs > _to_litellm_args(llm) (e.g. Zhipu's api_key)
        - explicit kwargs > LiteLLM env-var lookup
        - explicit extra_kwargs["seed"] > the default JUDGE_SEED applied at temperature 0
    """
    args = _to_litellm_args(llm)
    if api_key is not None:
        args["api_key"] = api_key
    if api_base is not None:
        args["api_base"] = api_base
    # Deterministic judge calls: every call made at temperature 0 gets a fixed
    # seed, so identical inputs reproduce the same output as far as the
    # provider allows (OpenAI: best-effort). Calls that sample on purpose
    # (temperature > 0: G-Eval, CustomEval with n_runs > 1, the data generator)
    # are left alone, and an explicit extra_kwargs["seed"] still wins via the
    # update() below. LiteLLM drops the param for providers that lack it.
    if temperature == 0:
        args.setdefault("seed", JUDGE_SEED)
    if extra_kwargs:
        args.update(extra_kwargs)

    try:
        response = await litellm.acompletion(
            messages=messages,
            temperature=temperature,
            **args,
        )
    except litellm.AuthenticationError as e:
        provider_label = (
            llm.provider.value if isinstance(llm.provider, Provider) else str(llm.provider)
        )
        raise LLMConfigurationError(
            f"❌ {provider_label} authentication failed!\n\n"
            f"Error: {str(e)}\n\n"
            f"Please check the relevant API key environment variable for your provider."
        )

    text = response.choices[0].message.content
    if text is None:
        text = ""
    text = text.strip()

    # response.usage from LiteLLM is normalized to OpenAI-style fields
    # (prompt_tokens, completion_tokens) regardless of the underlying provider,
    # so our existing _calculate_cost() works without changes.
    cost = _calculate_cost(llm, response.usage)
    return text, cost


# ---------------------------------------------------------------------------
# Native helpers (kept because LiteLLM doesn't cover their special cases)
# ---------------------------------------------------------------------------


async def _ollama_chat_complete(
    client,
    llm: LLMDescriptor,
    messages: list[dict[str, str]],
    temperature: float,
):
    """Ollama (local) chat completion via native API (supports think=false)."""
    base_url = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434")
    # Strip /v1 suffix if present (native API doesn't use it)
    base_url = base_url.rstrip("/").removesuffix("/v1")

    payload = {
        "model": llm.model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": temperature},
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                data = await resp.json()

        text = data.get("message", {}).get("content", "").strip()
        return text, 0.0
    except (aiohttp.ClientConnectorError, aiohttp.ClientError, ConnectionError) as e:
        raise LLMConfigurationError(
            f"❌ Cannot connect to Ollama server!\n\n"
            f"Error: {str(e)}\n\n"
            f"Make sure Ollama is running:\n"
            f"  1. Install Ollama: https://ollama.ai/download\n"
            f"  2. Start Ollama: ollama serve\n"
            f"  3. Pull model: ollama pull {llm.model}\n\n"
            f"Or set OLLAMA_API_BASE_URL to your Ollama server:\n"
            f"  export OLLAMA_API_BASE_URL='http://localhost:11434'"
        )


_THINK_RE = _re.compile(r"<think>[\s\S]*?</think>\s*", _re.DOTALL)


async def _mlx_chat_complete(
    client,
    llm: LLMDescriptor,
    messages: list[dict[str, str]],
    temperature: float,
):
    """MLX local server chat completion. Prepends /no_think and strips thinking tags."""
    # Prepend /no_think to the last user message to disable thinking
    patched = []
    for msg in messages:
        if msg["role"] == "user":
            patched.append({"role": msg["role"], "content": "/no_think\n" + msg["content"]})
        else:
            patched.append(msg)

    try:
        response = await client.chat.completions.create(
            model=llm.model,
            messages=patched,
            temperature=temperature,
        )
        text = response.choices[0].message.content.strip()
        # Strip residual <think>...</think> tags
        text = _THINK_RE.sub("", text).strip()
        return text, 0.0
    except Exception as e:
        if "Connection" in str(e) or "refused" in str(e).lower():
            raise LLMConfigurationError(
                f"❌ Cannot connect to MLX server!\n\n"
                f"Error: {str(e)}\n\n"
                f"Make sure MLX server is running:\n"
                f"  mlx_lm.server --model <model-name> --port 8899\n\n"
                f"Or set MLX_API_BASE_URL to your MLX server:\n"
                f"  export MLX_API_BASE_URL='http://localhost:8899/v1'"
            )
        raise


# ---------------------------------------------------------------------------
# Provider → helper dispatch
# ---------------------------------------------------------------------------
#
# Only native (non-LiteLLM) providers need an entry here. Anything else —
# OpenAI, Anthropic, Google, Azure, Bedrock, Vertex AI, Cohere, OpenRouter,
# Together AI, and every other provider LiteLLM supports — falls through to
# _litellm_chat_complete via the default in chat_complete().

_HELPERS: dict[Provider, "callable"] = {
    Provider.OLLAMA: _ollama_chat_complete,
    Provider.MLX: _mlx_chat_complete,
    # Zhipu still goes through LiteLLM, but with a special base_url injection
    # in _to_litellm_args(). It's kept out of this dispatch so chat_complete()
    # falls through to _litellm_chat_complete for it.
}


async def chat_complete(
    llm: "str | tuple[str, str] | LLMDescriptor | CustomLLMClient",
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    *,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    extra_kwargs: Optional[dict] = None,
):
    """
    Complete a chat conversation using the specified LLM.

    Args:
        llm: LLM specification (e.g., "gpt-4o-mini", "openai:gpt-4o", or LLMDescriptor)
        messages: List of message dicts with "role" and "content"
        temperature: Sampling temperature (0.0-2.0)
        api_key: Optional override for the provider API key. When set, it is
            forwarded directly to LiteLLM and supersedes the env var.
            Used by multi-tenant hosts that store credentials per user.
        api_base: Optional override for the provider base URL (e.g. for
            self-hosted OpenAI-compatible servers or Azure deployments).
        extra_kwargs: Optional dict of additional kwargs forwarded to
            litellm.acompletion() — for niche cases like
            `aws_access_key_id` for Bedrock or `vertex_project` for Vertex AI.

    The optional kwargs are passed through to the LiteLLM-backed helper.
    They are ignored for native helpers (Ollama, MLX) and CustomLLMClient,
    where credentials are configured at construction time instead.

    Returns:
        Tuple of (response_text, cost_in_usd)

    Raises:
        LLMConfigurationError: If required API keys or configuration are missing
        ValueError: If provider is not supported
    """
    # Handle custom LLM clients
    if isinstance(llm, CustomLLMClient):
        return await llm.chat_complete(messages, temperature)

    # Standard providers
    llm = LLMDescriptor.parse(llm)
    # Known providers have a dedicated helper in _HELPERS. Anything else is
    # an auto-discovered LiteLLM provider — route it through the LiteLLM
    # passthrough helper, which uses the provider id as the LiteLLM prefix.
    helper = _HELPERS.get(llm.provider, _litellm_chat_complete)

    # Native paths need an HTTP client; LiteLLM-backed paths don't.
    if llm.provider in (Provider.OLLAMA, Provider.MLX):
        client = _get_client(llm.provider)
    else:
        client = None

    # Only the LiteLLM helper accepts the credential kwargs. Native helpers
    # (Ollama, MLX) ignore them — they configure connections via env vars at
    # client construction time.
    if helper is _litellm_chat_complete:
        return await helper(
            client,
            llm,
            messages,
            temperature,
            api_key=api_key,
            api_base=api_base,
            extra_kwargs=extra_kwargs,
        )
    return await helper(client, llm, messages, temperature)


def _usage_attr(usage, *names) -> int:
    """First non-null integer among ``names`` on a usage object or dict."""
    for name in names:
        value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def _usage_sub(usage, name):
    """Nested usage detail block (``prompt_tokens_details``), or None."""
    value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
    return value


def _calculate_cost(llm: LLMDescriptor, usage) -> Optional[float]:
    """Calculate the cost of the LLM usage based on the model and usage data.

    Cached prompt tokens are billed at the provider's (much cheaper) cache-read
    rate. In a multi-agent loop the conversation history is resent every turn,
    so pricing cache hits at the full input rate overstates spend several-fold.

    The two providers report caching differently and both are handled:

    * OpenAI — ``prompt_tokens`` *includes* the cached tokens, which are
      detailed in ``prompt_tokens_details.cached_tokens``; the cached part is
      subtracted from the full-rate input before being re-priced.
    * Anthropic — ``input_tokens`` *excludes* cache reads, reported separately
      as ``cache_read_input_tokens``; those are added on at the cache rate.

    Reasoning tokens need no adjustment: providers already count them inside
    ``completion_tokens`` and bill them at the output rate.
    """
    if llm.provider == Provider.OLLAMA:
        return 0.0
    if not usage:
        return None

    price = get_cost_per_million(llm.model)
    if not price:
        return None

    prompt = _usage_attr(usage, "prompt_tokens", "input_tokens")
    completion = _usage_attr(usage, "completion_tokens", "output_tokens")

    # OpenAI style: cached tokens are a subset of `prompt`.
    details = _usage_sub(usage, "prompt_tokens_details")
    cached_inclusive = _usage_attr(details, "cached_tokens") if details else 0
    # Anthropic style: cache reads are reported outside `prompt`.
    cached_exclusive = _usage_attr(usage, "cache_read_input_tokens")
    cache_write = _usage_attr(usage, "cache_creation_input_tokens")

    # Fall back to the full input rate when the model has no cache price.
    cache_read_rate = price.get("cache_read")
    if cache_read_rate is None:
        cache_read_rate = price["input"]
    cache_write_rate = price.get("cache_write")
    if cache_write_rate is None:
        cache_write_rate = price["input"]

    billable_prompt = max(prompt - cached_inclusive, 0)
    cached_total = cached_inclusive + cached_exclusive

    cost = (
        billable_prompt * price["input"]
        + cached_total * cache_read_rate
        + cache_write * cache_write_rate
        + completion * price["output"]
    ) / 1_000_000

    return round(cost, 6)


# ---------------------------------------------------------------------------
# Embeddings — still uses the OpenAI SDK directly (separate slice, untouched)
# ---------------------------------------------------------------------------


async def get_embeddings(
    model: "str | tuple[str, str] | LLMDescriptor | CustomLLMClient",
    texts: list[str],
) -> tuple[list[list[float]], Optional[float]]:
    """
    Get embeddings for a list of texts.

    Args:
        model: Model specification or CustomLLMClient instance
        texts: List of texts to embed

    Returns:
        Tuple of (embeddings_list, total_cost)

    Raises:
        LLMConfigurationError: If required API keys are missing
        ValueError: If provider doesn't support embeddings
        NotImplementedError: If CustomLLMClient doesn't implement get_embeddings
    """
    # Handle custom LLM clients
    if isinstance(model, CustomLLMClient):
        return await model.get_embeddings(texts)

    llm = LLMDescriptor.parse(model)

    if llm.provider != "openai":
        raise ValueError(f"Only OpenAI embedding models are supported, got {llm.provider}")

    _check_env_var("OPENAI_API_KEY", "OpenAI")
    client = openai.AsyncOpenAI()
    return await _openai_get_embeddings(client, llm, texts)


async def _openai_get_embeddings(
    client,
    llm: LLMDescriptor,
    texts: list[str],
) -> tuple[list[list[float]], Optional[float]]:
    """OpenAI embeddings implementation."""
    try:
        response = await client.embeddings.create(
            model=llm.model, input=texts, encoding_format="float"
        )

        embeddings = [data.embedding for data in response.data]
        cost = _calculate_embedding_cost(llm, response.usage)

        return embeddings, cost
    except Exception as e:
        if "API key" in str(e) or "authentication" in str(e).lower():
            raise LLMConfigurationError(
                f"❌ OpenAI API authentication failed for embeddings!\n\n"
                f"Error: {str(e)}\n\n"
                f"Please check that your OPENAI_API_KEY is valid.\n"
                f"Get your API key at: https://platform.openai.com/api-keys"
            )
        raise


def _calculate_embedding_cost(llm: LLMDescriptor, usage) -> Optional[float]:
    """Calculate the cost of embedding usage for OpenAI models."""
    if not usage:
        return None

    price = get_cost_per_million(llm.model)
    if not price:
        return None

    total_tokens = getattr(usage, "total_tokens", 0)
    input_price = price.get("input", 0)

    return round(total_tokens * input_price / 1_000_000, 6)
