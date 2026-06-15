import numpy as np

from .base import Solver


class UCB(Solver):
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.counts + 1))
        )
        k = np.argmax(ucb)
        r = self.bandit.step(k)
        self.estimates[k] = self.update_estimate(
            self.estimates[k], self.counts[k] + 1, r
        )
        return k
