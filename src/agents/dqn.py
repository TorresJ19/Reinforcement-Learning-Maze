from dataclasses import dataclass
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.networks import DuelingMLP


@dataclass
class DQNConfig:
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 20000
    target_update: int = 1000
    batch_size: int = 128
    replay_capacity: int = 50000


class DQNAgent:
    def __init__(self, obs_shape, n_actions: int, device: torch.device, config: DQNConfig = DQNConfig()):
        self.device = device
        self.config = config
        obs_dim = int(torch.prod(torch.tensor(obs_shape)))
        self.net = DuelingMLP(obs_dim, n_actions).to(device)
        self.target = DuelingMLP(obs_dim, n_actions).to(device)
        self.target.load_state_dict(self.net.state_dict())
        self.n_actions = n_actions
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.lr)
        self.steps = 0

    def act(self, obs: torch.Tensor) -> int:
        eps = self._epsilon()
        self.steps += 1
        if torch.rand(1).item() < eps:
            return int(torch.randint(0, self.n_actions, (1,)).item())
        with torch.no_grad():
            q = self.net(obs.to(self.device))
            return int(q.argmax(dim=1).item())

    def _epsilon(self) -> float:
        cfg = self.config
        return max(cfg.epsilon_end, cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * max(0, (cfg.epsilon_decay - self.steps) / cfg.epsilon_decay))

    def update(self, batch) -> float:
        obs, actions, rewards, next_obs, dones = batch
        q_values = self.net(obs).gather(1, actions.long().unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target(next_obs).max(dim=1)[0]
            target_q = rewards + self.config.gamma * next_q * (1 - dones)
        loss = F.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.steps % self.config.target_update == 0:
            self.target.load_state_dict(self.net.state_dict())
        return loss.item()

