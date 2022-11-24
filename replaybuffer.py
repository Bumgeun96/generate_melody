import random
from collections import namedtuple
import numpy as np
Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'done'))

class Memory(object):
    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, next_state, action, reward, done):
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(state, next_state, action, reward, done))
        self.memory[self.position] = Transition(state, next_state, action, reward, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size-1)
        transitions.append(self.memory[-1])
        batch = Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)