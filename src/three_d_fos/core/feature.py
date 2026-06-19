from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Feature:
    name: str
    size: int
    scale: float = 1.0

    def normalize(self, data: np.ndarray) -> np.ndarray:
        if self.scale <= 0:
            raise ValueError(f"Scale must be positive, got {self.scale}")
        return np.clip(data / self.scale, 0.0, 1.0)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Feature):
            return NotImplemented
        return self.name == other.name and self.scale == other.scale


# Predefined features
Z0 = Feature("Z0", 1, 30.0)
DIST_AXES = Feature("dist_axes", 1, 15.0)
