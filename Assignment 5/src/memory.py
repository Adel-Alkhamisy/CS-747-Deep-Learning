from config import *
from collections import deque
import numpy as np
import random
import torch


class ReplayMemory(object):
    def __init__(self):
        self.memory = deque(maxlen=Memory_capacity)
    
    def push(self, history, action, reward, done):
        if torch.is_tensor(action):
            action = action.cpu()
        self.memory.append((history, action, reward, done))

    def sample_mini_batch(self, frame):
        mini_batch = []
        if frame >= Memory_capacity:
            sample_range = Memory_capacity
        else:
            sample_range = frame

        # history size
        sample_range -= (HISTORY_SIZE + 1)

        idx_sample = random.sample(range(sample_range), batch_size)
        for i in idx_sample:
            sample = []
            for j in range(HISTORY_SIZE + 1):
                sample.append(self.memory[i + j])

            sample_history = np.stack([s[0] for s in sample], axis=0)
            sample_action = sample[-1][1]
            sample_reward = sample[-1][2]
            sample_done = sample[-1][3]

            mini_batch.append((sample_history, sample_action, sample_reward, sample_done))

        return mini_batch


    def __len__(self):
        return len(self.memory)