from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class RolloutBuffer:
    obs: List[torch.Tensor]
    actions: List[torch.Tensor]
    logprobs: List[torch.Tensor]
    rewards: List[float]
    dones: List[float]
    values: List[torch.Tensor]

    def __init__(self):
        self.obs, self.actions, self.logprobs, self.rewards, self.dones, self.values = [], [], [], [], [], []

    def add(self, obs, action, logprob, reward, done, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_advantages(self, gamma: float, lam: float, next_value: torch.Tensor):
        size = len(self.rewards)
        advantages = torch.zeros(size, device=next_value.device)
        lastgaelam = 0
        
        # Ensure next_value is a scalar tensor
        next_value = next_value.squeeze()
        
        for t in reversed(range(size)):
            next_nonterminal = 1.0 - self.dones[t]
            # Get next value (scalar)
            if t == size - 1:
                next_values = next_value
            else:
                next_values = self.values[t + 1].squeeze()
            
            # Get current value (scalar)
            curr_val = self.values[t].squeeze()
            
            delta = self.rewards[t] + gamma * next_values * next_nonterminal - curr_val
            lastgaelam = delta + gamma * lam * next_nonterminal * lastgaelam
            advantages[t] = lastgaelam
        
        # Stack values and ensure they're 1D (squeeze any extra dimensions)
        stacked_values = torch.stack([v.squeeze() for v in self.values])
        returns = advantages + stacked_values
        return advantages, returns

    def to_tensors(self) -> Dict[str, torch.Tensor]:
        return {
            "obs": torch.stack(self.obs),
            "actions": torch.stack(self.actions),
            "logprobs": torch.stack(self.logprobs),
            "values": torch.stack(self.values),
            "rewards": torch.tensor(self.rewards),
            "dones": torch.tensor(self.dones),
        }

    def clear(self):
        self.__init__()

