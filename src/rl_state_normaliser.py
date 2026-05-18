"""Welford-running state normaliser for PPO.

Extracted as a module so the math can be tested without TensorFlow. The
notebook defines an inline copy of this class so the cell can be re-executed
without the import path being on sys.path; the two should stay numerically
identical.
"""
from __future__ import annotations
import numpy as np


class StateNormaliser:
    def __init__(self, dim: int, eps: float = 1e-5):
        self.dim = dim
        self.mean = np.zeros(dim, dtype=np.float32)
        self.M2 = np.zeros(dim, dtype=np.float32)
        self.count = 0
        self.eps = eps
        self.frozen = False

    @property
    def var(self) -> np.ndarray:
        return self.M2 / max(self.count, 1)

    def update(self, x: np.ndarray) -> None:
        if self.frozen:
            return
        x = np.asarray(x, dtype=np.float32)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def normalise(self, x: np.ndarray, update: bool = True) -> np.ndarray:
        if update:
            self.update(x)
        return ((x - self.mean) / (np.sqrt(self.var) + self.eps)).astype(np.float32)
