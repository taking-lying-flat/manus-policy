from collections import namedtuple
import numpy as np

Transition = namedtuple("Transition", ["state", "action", "mask", "next_state", "reward"])


class Memory:
    def __init__(self):
        self._s = []
        self._a = []
        self._m = []
        self._ns = []
        self._r = []

    def push(self, state, action, mask, next_state, reward):
        self._s.append(np.asarray(state))
        self._a.append(np.asarray(action))
        self._m.append(mask)
        self._ns.append(np.asarray(next_state))
        self._r.append(reward)

    def get_all(self):
        return Transition(
            state=np.stack(self._s),
            action=np.concatenate(self._a, axis=0),
            mask=np.array(self._m, dtype=np.float64),
            next_state=np.stack(self._ns),
            reward=np.array(self._r, dtype=np.float64),
        )

    def __len__(self):
        return len(self._r)
