from collections import defaultdict
import numpy as np


class QLearning:
    def __init__(self, actions):
        self.q_values = defaultdict(lambda: [0 for _ in range(len(actions))])
        self.alpha = 0.1
        self.discount_factor = 0.9

    def learn(self, state, action, next_state, next_action, reward):
        td_error = self.discount_factor * self.q_values[next_state][next_action] - self.q_values[state][action]
        self.q_values[state][action] += self.alpha * td_error

    def get_action(self, state):
        return np.argmax(self.q_values[state])