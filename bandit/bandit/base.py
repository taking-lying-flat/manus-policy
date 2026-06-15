from abc import ABC, abstractmethod
from numbers import Integral

import numpy as np


class BernoulliBandit:
    def __init__(self, K):
        self.K = self.validate_arm_count(K)
        self.probs = np.random.uniform(size=K)
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]

    @staticmethod
    def validate_arm_count(K):
        if not isinstance(K, Integral):
            raise TypeError("K must be an integer")
        if K <= 0:
            raise ValueError("K must be positive")
        return int(K)

    @staticmethod
    def validate_arm(k, K):
        if not isinstance(k, Integral):
            raise TypeError("arm index must be an integer")
        if k < 0 or k >= K:
            raise ValueError(f"arm index must be in [0, {K})")
        return int(k)

    def step(self, k):
        k = self.validate_arm(k, self.K)
        if np.random.rand() < self.probs[k]:
            return 1
        return 0


class Solver(ABC):
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)
        self.regret = 0
        self.actions = []
        self.regrets = []

    @staticmethod
    def update_estimate(old_estimate, count, reward):
        return old_estimate + 1. / count * (reward - old_estimate)

    def update_regret(self, k):
        k = self.bandit.validate_arm(k, self.bandit.K)
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    @abstractmethod
    def run_one_step(self):
        pass

    def run(self, num_steps):
        if not isinstance(num_steps, Integral):
            raise TypeError("num_steps must be an integer")
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative")

        for _ in range(num_steps):
            k = self.run_one_step()
            k = self.bandit.validate_arm(k, self.bandit.K)
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)
