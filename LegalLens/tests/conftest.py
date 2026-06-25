from pathlib import Path

import pytest

from config.settings import Settings


@pytest.fixture
def settings_override(tmp_path: Path) -> Settings:
    return Settings(
        contracts_dir=tmp_path,
        qdrant_url="http://localhost:6333",
        gemini_api_key="test-key",
    )

