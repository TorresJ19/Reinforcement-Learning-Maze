from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.full = False
        self.obs = torch.zeros((capacity,) + obs_shape, dtype=torch.float32, device=device)
        self.next_obs = torch.zeros_like(self.obs)
        self.actions = torch.zeros((capacity, 1), dtype=torch.int64, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

    def push(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = torch.as_tensor(obs, device=self.device)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = torch.as_tensor(next_obs, device=self.device)
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.full = self.full or self.ptr == 0

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        max_idx = self.capacity if self.full else self.ptr
        idx = torch.randint(0, max_idx, (batch_size,), device=self.device)
        return (
            self.obs[idx],
            self.actions[idx].squeeze(-1),
            self.rewards[idx].squeeze(-1),
            self.next_obs[idx],
            self.dones[idx].squeeze(-1),
        )

    def __len__(self):
        return self.capacity if self.full else self.ptr


