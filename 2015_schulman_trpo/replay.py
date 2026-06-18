from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: np.ndarray
    mask: np.ndarray
    gae_mask: np.ndarray
    next_state: np.ndarray
    reward: np.ndarray


class Memory:
    def __init__(self) -> None:
        self._s = []
        self._a = []
        self._m = []
        self._gm = []
        self._ns = []
        self._r = []

    def push(
        self,
        state: np.ndarray,
        action: int | np.ndarray,
        mask: float,
        next_state: np.ndarray,
        reward: float,
        gae_mask: float,
    ) -> None:
        self._s.append(np.asarray(state, dtype=np.float64))
        self._a.append(np.asarray(action))
        self._m.append(float(mask))
        self._gm.append(float(gae_mask))
        self._ns.append(np.asarray(next_state, dtype=np.float64))
        self._r.append(float(reward))

    def get_all(self) -> Transition:
        return Transition(
            state=np.stack(self._s),
            action=np.stack(self._a),
            mask=np.asarray(self._m, dtype=np.float64).reshape(-1, 1),
            gae_mask=np.asarray(self._gm, dtype=np.float64).reshape(-1, 1),
            next_state=np.stack(self._ns),
            reward=np.asarray(self._r, dtype=np.float64).reshape(-1, 1),
        )
