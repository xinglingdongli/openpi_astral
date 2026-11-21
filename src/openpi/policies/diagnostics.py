"""Optional CPU-only policy observers; no JAX/PyTorch imports or device reads."""
from __future__ import annotations

import numpy as np


def notify(hook, event: str, data: dict) -> None:
    """An observer gets owned CPU snapshots and cannot mutate model inputs."""
    if hook is None or not getattr(hook, "enabled", True):
        return

    def snapshot(value):
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, dict):
            return {key: snapshot(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [snapshot(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool, np.generic)):
            return value
        raise TypeError("Only CPU NumPy data may enter diagnostic observers")

    try:
        hook(event, snapshot(data))
    except Exception:
        # Diagnostics never change model execution or error handling.
        pass
