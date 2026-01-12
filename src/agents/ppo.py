from dataclasses import dataclass
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.models.networks import ActorCritic
from src.utils.rollout import RolloutBuffer


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    update_epochs: int = 2
    mini_batch_size: int = 14  # Optimized for rollout_size=64: gives 4 mini-batches per epoch
    entropy_coef: float = 0.01  # Higher entropy for more exploration
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


class PPOAgent:
    def __init__(self, obs_shape, n_actions: int, device: torch.device, use_cnn: bool, config: PPOConfig = PPOConfig()):
        self.device = device
        self.config = config
        self.net = ActorCritic(obs_shape, n_actions, use_cnn).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.lr)
        self.buffer = RolloutBuffer()

    def act(self, obs: torch.Tensor):
        obs = obs.to(self.device)
        with torch.no_grad():
            logits, value = self.net(obs)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)
        return action, logprob, value

    def update(self, next_value: torch.Tensor) -> Dict[str, Any]:
        cfg = self.config
        buf = self.buffer
        batch_size = len(buf.rewards)
        
        # Skip update if buffer is too small
        if batch_size == 0:
            buf.clear()
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        
        tensors = buf.to_tensors()
        advantages, returns = buf.compute_advantages(cfg.gamma, cfg.gae_lambda, next_value)
        # Normalize advantages only if we have enough samples (std requires at least 2 samples)
        if len(advantages) > 32:       
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages - advantages.mean()
        # If only 1 sample, just center it (can't normalize)
        inds = torch.randperm(batch_size)
        policy_losses = []
        value_losses = []
        entropy_losses = []

        for _ in range(cfg.update_epochs):
            inds = torch.randperm(batch_size)
            for start in range(0, batch_size, cfg.mini_batch_size):
                end = start + cfg.mini_batch_size
                mb_inds = inds[start:end]

                mb_obs = tensors["obs"][mb_inds].to(self.device)
                mb_actions = tensors["actions"][mb_inds].to(self.device)
                mb_old_logprobs = tensors["logprobs"][mb_inds].to(self.device)
                mb_adv = advantages[mb_inds].to(self.device)
                mb_returns = returns[mb_inds].to(self.device)

                logits, values = self.net(mb_obs)
                dist = Categorical(logits=logits)
                new_logprob = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                log_ratio = new_logprob - mb_old_logprobs
                ratio = log_ratio.exp()
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Network already returns squeezed values (1D), and returns from compute_advantages is also 1D
                if values.dim() > 1:
                    values = values.squeeze(-1)
                if mb_returns.dim() > 1:
                    mb_returns = mb_returns.squeeze(-1)
                
                value_loss = 0.5 * (mb_returns - values).pow(2).mean()

                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())

        buf.clear()
        return {
            "policy_loss": sum(policy_losses) / max(len(policy_losses), 1),
            "value_loss": sum(value_losses) / max(len(value_losses), 1),
            "entropy": sum(entropy_losses) / max(len(entropy_losses), 1),
        }

