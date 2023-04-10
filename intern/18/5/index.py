from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Node:
    """Decision tree node."""
    feature: int = field(default=None)
    threshold: float = field(default=None)
    n_samples: int = field(default=None)
    value: int = field(default=None)
    mse: float = field(default=None)

    left: Node = None
    right: Node = None
