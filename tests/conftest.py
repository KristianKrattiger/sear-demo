"""Run the tests without the GPU stack: stub transformers when it isn't installed."""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    import transformers  # noqa: F401
except ImportError:
    sys.modules["transformers"] = types.SimpleNamespace(pipeline=None)
