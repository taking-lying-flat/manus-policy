from __future__ import annotations

import numpy as np


class RunningStat:
    def __init__(self, shape):
        self._count = 0
        self._mean = np.zeros(shape)
        self._M2 = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        self._count += 1
        delta = x - self._mean
        self._mean += delta / self._count
        self._M2 += delta * (x - self._mean)

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self._M2 / (self._count - 1) if self._count > 1 else np.square(self._mean)

    @property
    def std(self):
        return np.sqrt(self.var)


class ZFilter:
    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip
        self._stat = RunningStat(shape)

    def __call__(self, x, update=True):
        x = np.asarray(x, dtype=np.float64)
        if update:
            self._stat.push(x)
        if self.demean:
            x = x - self._stat.mean
        if self.destd:
            x = x / (self._stat.std + 1e-8)
        if self.clip is not None:
            x = np.clip(x, -self.clip, self.clip)
        return x
