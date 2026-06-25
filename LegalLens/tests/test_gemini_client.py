from llm.gemini_client import GeminiClient


def test_gemini_returns_graceful_error_when_api_fails(monkeypatch) -> None:
    client = GeminiClient(
        api_key="test-key",
        model_name="gemini-test",
        temperature=0.1,
        max_tokens=128,
        timeout_seconds=1,
        max_retries=1,
    )

    def fail(_: str) -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr(client, "_generate_with_retry", fail)

    answer = client.generate("prompt", [])

    assert answer.error is not None
    assert "Gemini could not generate" in answer.answer

