from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBackbone(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int], hidden_size: int = 128):
        super().__init__()
        c, h, w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out = self.conv(dummy).shape[1]
        self.head = nn.Sequential(nn.Linear(conv_out, hidden_size), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x))


class MLPBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, obs_shape: Tuple[int, ...], n_actions: int, use_cnn: bool):
        super().__init__()
        if use_cnn:
            # obs expected as (B,H,W,C) -> permute in forward
            c, h, w = obs_shape[2], obs_shape[0], obs_shape[1]
            self.backbone = CNNBackbone((c, h, w))
        else:
            self.backbone = MLPBackbone(int(torch.prod(torch.tensor(obs_shape))))
        hidden_size = 128
        self.policy = nn.Linear(hidden_size, n_actions)
        self.value = nn.Linear(hidden_size, 1)
        self.use_cnn = use_cnn

    def forward(self, obs: torch.Tensor):
        if self.use_cnn:
            x = obs.permute(0, 3, 1, 2)  # (B,H,W,C) -> (B,C,H,W)
        else:
            x = obs.view(obs.size(0), -1)
        features = self.backbone(x)
        logits = self.policy(features)
        value = self.value(features)
        return logits, value.squeeze(-1)


class DuelingMLP(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden, 1)
        self.adv_head = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature(x.view(x.size(0), -1))
        value = self.value_head(h)
        adv = self.adv_head(h)
        q = value + adv - adv.mean(dim=1, keepdim=True)
        return q


