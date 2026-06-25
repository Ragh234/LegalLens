import logging
from time import perf_counter

import google.generativeai as genai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from utils.models import RAGAnswer, RetrievedChunk
from utils.logging import log_event

logger = logging.getLogger(__name__)


class GeminiClient:
    """Google Gemini text generation client used after retrieval and prompt construction."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: int,
        max_retries: int,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required to generate answers.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def generate(self, prompt: str, citations: list[RetrievedChunk]) -> RAGAnswer:
        start = perf_counter()
        try:
            response_text = self._generate_with_retry(prompt)
            latency_ms = (perf_counter() - start) * 1000
            log_event(logger, "llm_completed", latency_ms=round(latency_ms, 2))
            return RAGAnswer(
                answer=response_text or "I could not find that information in the uploaded contracts.",
                citations=citations,
                prompt=prompt,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            latency_ms = (perf_counter() - start) * 1000
            message = "Gemini could not generate an answer right now. Please check your API key or try again."
            log_event(logger, "llm_failed", latency_ms=round(latency_ms, 2), error=str(exc))
            return RAGAnswer(answer=message, citations=citations, prompt=prompt, latency_ms=latency_ms, error=message)

    def _generate_with_retry(self, prompt: str) -> str:
        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        def call() -> str:
            log_event(logger, "llm_attempt")
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                },
                request_options={"timeout": self.timeout_seconds},
            )
            return response.text or ""

        return call()
