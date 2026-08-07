import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import resolve_start_epoch


def test_resolve_start_epoch_from_checkpoint() -> None:
    assert resolve_start_epoch(10, 135) == 11
    assert resolve_start_epoch(135, 135) == 136
    assert resolve_start_epoch(None, 135) == 1
