import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXAMPLES = ROOT / "examples"


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    """Point every storage call (including the UI's) to a temporary folder."""
    monkeypatch.setenv("MLPERFORMANCE_RUNS_DIR", str(tmp_path))
    return tmp_path
