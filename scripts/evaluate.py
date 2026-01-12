import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch

from src.agents.dqn import DQNAgent
from src.agents.ppo import PPOAgent
from src.envs.maze_env import MazeEnv, MazeConfig


def make_env(args) -> MazeEnv:
    cfg = MazeConfig(
        grid_size=(args.size, args.size),
        wall_fraction=args.wall_fraction,
        obstacle_fraction=args.obstacle_fraction,
        dynamic_obstacles=args.dynamic_obstacles,
        max_steps=args.max_steps,
        shaping=not args.no_shaping,
        flatten=args.algorithm == "dqn",
        render_mode=args.render_mode,
        seed=args.seed,
    )
    return MazeEnv(cfg)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agent on maze")
    parser.add_argument("--algorithm", choices=["ppo", "dqn"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--wall-fraction", type=float, default=0.15)
    parser.add_argument("--obstacle-fraction", type=float, default=0.10)
    parser.add_argument("--dynamic-obstacles", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--no-shaping", action="store_true")
    parser.add_argument("--render-mode", choices=["none", "human", "rgb_array"], default="human")
    args = parser.parse_args()

    env = make_env(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_shape = env.observation_space.shape
    if obs_shape is None:
        raise ValueError("Observation space shape is None")
    action_space_n = env.action_space.n  # type: ignore
    if args.algorithm == "ppo":
        use_cnn = len(obs_shape) == 3 and not env.config.flatten
        agent = PPOAgent(obs_shape, action_space_n, device, use_cnn=use_cnn)
        agent.net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        agent = DQNAgent(obs_shape, action_space_n, device)
        agent.net.load_state_dict(torch.load(args.checkpoint, map_location=device))

    returns = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        ep_return = 0.0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            if args.algorithm == "ppo":
                with torch.no_grad():
                    action_tuple = agent.act(obs_t)
                    if isinstance(action_tuple, tuple) and len(action_tuple) >= 1:
                        action_tensor = action_tuple[0]
                        if isinstance(action_tensor, torch.Tensor):
                            action = int(action_tensor.item())
                        else:
                            action = int(action_tensor)
                    else:
                        # Handle non-tuple return
                        if isinstance(action_tuple, torch.Tensor):
                            action = int(action_tuple.item())
                        elif isinstance(action_tuple, int):
                            action = action_tuple
                        else:
                            action = int(action_tuple)
            else:
                action_result = agent.act(obs_t)
                if isinstance(action_result, int):
                    action = action_result
                elif isinstance(action_result, torch.Tensor):
                    action = int(action_result.item())
                elif isinstance(action_result, tuple):
                    # Handle unexpected tuple return
                    if len(action_result) > 0 and isinstance(action_result[0], torch.Tensor):
                        action = int(action_result[0].item())
                    elif len(action_result) > 0:
                        action = int(action_result[0])
                    else:
                        raise ValueError(f"Unexpected empty tuple from DQN agent.act()")
                else:
                    action = int(action_result)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated
            env.render()
        returns.append(ep_return)
        print(f"Episode {ep+1}: return={ep_return:.2f}")
    print(f"Mean return over {args.episodes} episodes: {np.mean(returns):.2f}")
    env.close()


if __name__ == "__main__":
    main()


