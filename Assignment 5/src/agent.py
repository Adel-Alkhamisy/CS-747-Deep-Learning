import random
import torch
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory import ReplayMemory
from model import DQN
from utils import find_max_lives, check_live, get_frame, get_init_state
from config import *
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, num_actions):
        self.num_actions = num_actions

        # DQN hyperparameters
        self.discount_rate = 0.99
        self.init_epsilon = 1.0
        self.min_epsilon = 0.01
        self.epsilon_decay_steps = 500000
        self.epsilon_decay = (self.init_epsilon - self.min_epsilon) / self.epsilon_decay_steps
        self.training_start_step = 100000
        self.target_update_freq = 1000

        # Initialize the replay memory
        self.replay_memory = ReplayMemory()

        # Create the policy network
        self.policy_network = DQN(num_actions)
        self.policy_network.to(device)

        self.optim = optim.Adam(params=self.policy_network.parameters(), lr=learning_rate)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optim, step_size=scheduler_step_size, gamma=scheduler_gamma)

    def load_pretrained_policy_net(self, model_path):
        self.policy_network = torch.load(model_path)

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return torch.tensor([[random.randrange(self.num_actions)]], device=device, dtype=torch.long)

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).cuda()
            return self.policy_network(state).max(1)[1].view(1, 1).detach()

    def train_policy_network(self, current_frame):
        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.epsilon_decay

        minibatch = self.replay_memory.sample_mini_batch(current_frame)
        minibatch = np.array(minibatch, dtype=object).transpose()

        history_stack = np.stack(minibatch[0], axis=0)
        states_batch = np.float32(history_stack[:, :4, :, :]) / 255.
        states_batch = torch.from_numpy(states_batch).cuda()
        actions_batch = list(minibatch[1])
        actions_batch = torch.LongTensor(actions_batch).cuda()
        rewards_batch = list(minibatch[2])
        rewards_batch = torch.FloatTensor(rewards_batch).cuda()
        next_states_batch = np.float32(history_stack[:, 1:, :, :]) / 255.
        next_states_batch = torch.from_numpy(next_states_batch).cuda()
        terminal_flags = minibatch[3]
        not_terminal_mask = torch.tensor(list(map(int, terminal_flags==False)),dtype=torch.bool)

        q_values = torch.gather(self.policy_network(states_batch), 1, actions_batch.view(-1, 1))

        next_q_values = torch.zeros(len(minibatch[0]), device=device)
        next_q_values[not_terminal_mask] = self.policy_network(next_states_batch[not_terminal_mask]).max(1)[0].detach()

        target_q_values = (next_q_values * self.discount_rate) + rewards_batch

        loss = F.smooth_l1_loss(q_values, target_q_values.unsqueeze(1))

        self.optim.zero_grad()
        loss.backward()
        for param in self.policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim.step()
        self.lr_scheduler.step()