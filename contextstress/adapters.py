"""
Model adapter interface for ContextStress.

Provides an abstract base class that users implement for their model,
plus reference implementations for common APIs.
"""

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


SYSTEM_PROMPT = (
    "You are a factual reasoning assistant. Answer the question based "
    "solely on the provided context. Do not use any prior knowledge. "
    "After your answer, state your confidence (0-100%) in your response."
)

USER_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Question: {query}\n\n"
    "Provide your answer, then state \"Confidence: X%\".\n"
    "For multi-step questions, show your reasoning step by step."
)


@dataclass
class ModelResponse:
    """Response from a model adapter."""
    raw_text: str
    answer: str
    confidence: Optional[float]  # 0-100, or None if not parseable
    latency_ms: float
    tokens_used: int


class ModelAdapter(ABC):
    """Abstract base class for model adapters."""

    @abstractmethod
    def generate(self, query: str, context: str) -> ModelResponse:
        """
        Generate a response for the given query and context.

        Args:
            query: The evaluation question.
            context: The assembled context string.

        Returns:
            ModelResponse with answer, confidence, and metadata.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
        ...

    @property
    @abstractmethod
    def context_window(self) -> int:
        """Maximum context window in tokens."""
        ...

    def format_prompt(self, query: str, context: str) -> tuple[str, str]:
        """Format the system and user prompts. Override for custom formatting."""
        user_msg = USER_TEMPLATE.format(context=context, query=query)
        return SYSTEM_PROMPT, user_msg


def parse_answer_and_confidence(text: str) -> tuple[str, Optional[float]]:
    """Extract the answer and confidence from model output."""
    # Try to find confidence pattern
    conf_match = re.search(r"[Cc]onfidence:\s*(\d+(?:\.\d+)?)\s*%", text)
    confidence = float(conf_match.group(1)) if conf_match else None

    # Answer is everything before the confidence statement
    if conf_match:
        answer_text = text[:conf_match.start()].strip()
    else:
        answer_text = text.strip()

    # Clean up common prefixes
    for prefix in ["Answer:", "The answer is", "Final answer:"]:
        if answer_text.lower().startswith(prefix.lower()):
            answer_text = answer_text[len(prefix):].strip()

    # Look for explicit conclusion patterns anywhere in the text (before confidence)
    # These indicate the model stated a final answer explicitly
    conclusion_patterns = [
        r"[Ff]inal [Aa]nswer:\s*(.+)",
        r"[Tt]he (?:highest|lowest|best|answer) is\s+(.+?)\.?\s*$",
        r"[Aa]nswer:\s*(.+)",
        r"[Tt]herefore,?\s+(.+?)\.?\s*$",
        r"[Tt]hus,?\s+(.+?)\.?\s*$",
        r"[Ss]o,?\s+(.+?)\.?\s*$",
    ]
    for pat in conclusion_patterns:
        m = re.search(pat, answer_text, re.MULTILINE)
        if m:
            concluded = m.group(1).strip().rstrip(".")
            # Only use if it looks like a short answer (not another long sentence)
            if len(concluded) < 200:
                return concluded, confidence

    # Take last non-empty line before confidence
    lines = [l.strip() for l in answer_text.split("\n") if l.strip()]
    answer = lines[-1] if lines else answer_text  # Last non-empty line before confidence

    return answer.strip().rstrip("."), confidence


# ── Reference Implementations ─────────────────────────────────────────


class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI API models (GPT-4o, etc.)."""

    def __init__(
        self,
        model_id: str = "gpt-4o-2024-08-06",
        max_context: int = 128_000,
        max_retries: int = 10,
    ):
        self._model_id = model_id
        self._max_context = max_context
        self._max_retries = max_retries

    @property
    def name(self) -> str:
        return self._model_id

    @property
    def context_window(self) -> int:
        return self._max_context

    def generate(self, query: str, context: str) -> ModelResponse:
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai")

        system_prompt, user_msg = self.format_prompt(query, context)
        client = openai.OpenAI()

        for attempt in range(self._max_retries):
            try:
                t0 = time.time()
                response = client.chat.completions.create(
                    model=self._model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0,
                    top_p=1.0,
                    max_tokens=256,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                latency = (time.time() - t0) * 1000
                raw = response.choices[0].message.content or ""
                answer, confidence = parse_answer_and_confidence(raw)

                return ModelResponse(
                    raw_text=raw,
                    answer=answer,
                    confidence=confidence,
                    latency_ms=latency,
                    tokens_used=response.usage.total_tokens if response.usage else 0,
                )
            except Exception as e:
                if attempt == self._max_retries - 1:
                    raise
                wait = 2 ** attempt
                err_str = str(e)
                if "rate" in err_str.lower() or "429" in err_str:
                    # Parse wait time from error if available
                    import re as _re
                    m = _re.search(r"try again in (\d+(?:\.\d+)?)s", err_str)
                    wait = float(m.group(1)) + 5 if m else max(wait, 45)
                    print(f"    [rate limited, waiting {wait:.0f}s...]", flush=True)
                time.sleep(wait)

        raise RuntimeError("Unreachable")


class AnthropicAdapter(ModelAdapter):
    """Adapter for Anthropic API models (Claude)."""

    def __init__(
        self,
        model_id: str = "claude-3-5-sonnet-20241022",
        max_context: int = 200_000,
        max_retries: int = 10,
    ):
        self._model_id = model_id
        self._max_context = max_context
        self._max_retries = max_retries

    @property
    def name(self) -> str:
        return self._model_id

    @property
    def context_window(self) -> int:
        return self._max_context

    def generate(self, query: str, context: str) -> ModelResponse:
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")

        system_prompt, user_msg = self.format_prompt(query, context)
        client = anthropic.Anthropic()

        for attempt in range(self._max_retries):
            try:
                t0 = time.time()
                response = client.messages.create(
                    model=self._model_id,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_msg}],
                    temperature=0,
                    top_p=1.0,
                    max_tokens=256,
                )
                latency = (time.time() - t0) * 1000
                raw = response.content[0].text if response.content else ""
                answer, confidence = parse_answer_and_confidence(raw)

                return ModelResponse(
                    raw_text=raw,
                    answer=answer,
                    confidence=confidence,
                    latency_ms=latency,
                    tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                )
            except Exception as e:
                if attempt == self._max_retries - 1:
                    raise
                wait = 2 ** attempt
                err_str = str(e).lower()
                if "rate" in err_str or "429" in err_str:
                    wait = max(wait, 20)  # Rate limit: wait longer
                    print(f"    [rate limited, waiting {wait}s...]", flush=True)
                time.sleep(wait)

        raise RuntimeError("Unreachable")
