import argparse
import random
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch

from src.agents.ppo import PPOAgent, PPOConfig
from src.envs.maze_env import MazeEnv, MazeConfig
from src.utils.logger import Logger, save_config


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
        flatten=args.flatten,
        render_mode=args.render_mode,
        seed=args.seed,
    )
    return MazeEnv(cfg)


def main():
    parser = argparse.ArgumentParser(description="Train PPO on the maze environment")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--wall-fraction", type=float, default=0.15)
    parser.add_argument("--obstacle-fraction", type=float, default=0.10)
    parser.add_argument("--dynamic-obstacles", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--rollout-horizon", type=int, default=128)
    parser.add_argument("--flatten", action="store_true")
    parser.add_argument("--no-shaping", action="store_true")
    parser.add_argument("--render-mode", choices=["none", "human", "rgb_array"], default="none")
    parser.add_argument("--log-dir", type=Path, default=Path("runs/ppo"))
    args = parser.parse_args()

    set_seed(args.seed)
    env = make_env(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_shape = env.observation_space.shape
    if obs_shape is None:
        raise ValueError("Observation space shape is None")
    use_cnn = len(obs_shape) == 3 and not args.flatten
    action_space_n = env.action_space.n  # type: ignore
    agent = PPOAgent(obs_shape, action_space_n, device, use_cnn)

    log_dir = args.log_dir / f"seed{args.seed}"
    logger = Logger(log_dir)
    save_config(log_dir, vars(args))

    obs, info = env.reset(seed=args.seed)
    obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    ep_return = 0.0
    ep_len = 0
    successes = 0
    episodes = 0

    for step in range(1, args.total_steps + 1):
        with torch.no_grad():
            action, logprob, value = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(int(action.item()))
        ep_return += reward
        ep_len += 1
        done = terminated or truncated
        agent.buffer.add(obs.squeeze(0).cpu(), action.cpu(), logprob.cpu(), reward, float(done), value.cpu())

        obs = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
        if done or len(agent.buffer.rewards) >= args.rollout_horizon:
            with torch.no_grad():
                _, next_value = agent.net(obs)
            logs = agent.update(next_value)
            logger.log(step, {"return": ep_return, "length": ep_len, **logs})
            if done:
                episodes += 1
                if terminated and info.get("distance", 1) == 0:
                    successes += 1
            ep_return = 0.0
            ep_len = 0
            if done:
                obs, _ = env.reset()
                obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        if episodes > 0 and episodes % 20 == 0:
            logger.log(step, {"success_rate": successes / episodes})

    logger.flush()
    logger.plot()
    torch.save(agent.net.state_dict(), log_dir / "ppo_agent.pt")
    env.close()


if __name__ == "__main__":
    main()

