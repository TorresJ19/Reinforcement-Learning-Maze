import argparse
import random
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch

from src.agents.dqn import DQNAgent, DQNConfig
from src.envs.maze_env import MazeEnv, MazeConfig
from src.utils.logger import Logger, save_config
from src.utils.replay_buffer import ReplayBuffer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_env(args) -> MazeEnv:
    cfg = MazeConfig(
        grid_size=(args.size, args.size),
        wall_fraction=args.wall_fraction,
        obstacle_fraction=args.obstacle_fraction,
        dynamic_obstacles=args.dynamic_obstacles,
        max_steps=args.max_steps,
        shaping=not args.no_shaping,
        flatten=True,  # DQN uses flat vector
        render_mode=args.render_mode,
        seed=args.seed,
    )
    return MazeEnv(cfg)


def main():
    parser = argparse.ArgumentParser(description="Train DQN on the maze environment")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--wall-fraction", type=float, default=0.15)
    parser.add_argument("--obstacle-fraction", type=float, default=0.10)
    parser.add_argument("--dynamic-obstacles", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--no-shaping", action="store_true")
    parser.add_argument("--render-mode", choices=["none", "human", "rgb_array"], default="none")
    parser.add_argument("--log-dir", type=Path, default=Path("runs/dqn"))
    args = parser.parse_args()

    set_seed(args.seed)
    env = make_env(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_shape = env.observation_space.shape
    if obs_shape is None:
        raise ValueError("Observation space shape is None")
    action_space_n = env.action_space.n  # type: ignore
    agent = DQNAgent(obs_shape, action_space_n, device)
    buffer = ReplayBuffer(50000, obs_shape, device)

    log_dir = args.log_dir / f"seed{args.seed}"
    logger = Logger(log_dir)
    save_config(log_dir, vars(args))

    obs, _ = env.reset(seed=args.seed)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    ep_return = 0.0
    ep_len = 0

    for step in range(1, args.total_steps + 1):
        action = agent.act(obs_t)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        buffer.push(obs, action, reward, next_obs, float(done))

        ep_return += reward
        ep_len += 1

        obs = next_obs
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        if len(buffer) >= agent.config.batch_size:
            batch = buffer.sample(agent.config.batch_size)
            loss = agent.update(batch)
            logger.log(step, {"dqn_loss": loss})

        if done:
            logger.log(step, {"return": ep_return, "length": ep_len})
            obs, _ = env.reset()
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            ep_return = 0.0
            ep_len = 0

    logger.flush()
    logger.plot()
    torch.save(agent.net.state_dict(), log_dir / "dqn_agent.pt")
    env.close()


if __name__ == "__main__":
    main()

