import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional, Tuple, cast

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch

from src.agents.dqn import DQNAgent
from src.agents.ppo import PPOAgent
from src.envs.maze_env import MazeEnv, MazeConfig
from src.utils.logger import Logger, save_config
from src.utils.replay_buffer import ReplayBuffer


def load_maze(maze_path: Path) -> dict:
    """Load maze configuration from JSON file."""
    with open(maze_path, "r") as f:
        return json.load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_env_from_maze(maze_data: dict, render_mode: str = "none", max_steps: int = 500, flatten: bool = False) -> MazeEnv:
    """Create environment from custom maze data."""
    grid_size = maze_data["grid_size"]
    walls = np.array(maze_data["walls"], dtype=np.bool_)
    obstacles = np.array(maze_data["obstacles"], dtype=np.bool_)
    start = tuple(maze_data["start"])
    goal = tuple(maze_data["goal"])
    
    cfg = MazeConfig(
        grid_size=(grid_size, grid_size),
        wall_fraction=0.0,  # Not used when custom_walls is provided
        obstacle_fraction=0.0,  # Not used when custom_obstacles is provided
        dynamic_obstacles=0,
        max_steps=max_steps,
        shaping=True,
        flatten=flatten,
        render_mode=render_mode,
        seed=None,
        custom_walls=walls,
        custom_obstacles=obstacles,
        custom_start=start,
        custom_goal=goal,
    )
    return MazeEnv(cfg)


def train_and_test(algorithm: str, maze_data: dict, total_steps: int, seed: int, log_dir: Path, render_mode: str, max_steps: int):
    """Train an algorithm on the maze and return efficiency metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create environment with correct flatten setting
    flatten = (algorithm == "dqn")
    env = make_env_from_maze(maze_data, render_mode, max_steps, flatten=flatten)
    obs_shape = env.observation_space.shape
    if obs_shape is None:
        raise ValueError("Observation space shape is None")
    action_space_n = env.action_space.n  # type: ignore
    
    if algorithm == "dqn":
        agent = DQNAgent(obs_shape, action_space_n, device)
        buffer = ReplayBuffer(50000, obs_shape, device)
    else:  # ppo
        use_cnn = len(obs_shape) == 3
        agent = PPOAgent(obs_shape, action_space_n, device, use_cnn)
    
    logger = Logger(log_dir)
    
    obs, _ = env.reset(seed=seed)
    if algorithm == "dqn":
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    
    ep_return = 0.0
    ep_len = 0
    episodes = 0
    successes = 0
    total_steps_taken = 0
    
    for step in range(1, total_steps + 1):
        if algorithm == "dqn":
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
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.push(obs, action, reward, next_obs, float(done))
            
            obs = next_obs
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            if len(buffer) >= agent.config.batch_size:  # type: ignore
                batch = buffer.sample(agent.config.batch_size)  # type: ignore
                loss = agent.update(batch)  # type: ignore[arg-type]  # DQN update takes batch, not next_value
                logger.log(step, {"dqn_loss": loss})
        else:  # ppo
            # Ensure obs is a Tensor before calling agent.act()
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action_tuple = agent.act(obs)
                if isinstance(action_tuple, tuple) and len(action_tuple) == 3:
                    action, logprob, value = action_tuple
                else:
                    raise ValueError(f"Expected tuple of 3, got {action_tuple}")
            next_obs, reward, terminated, truncated, info = env.step(int(action.item()))
            done = terminated or truncated
            # Type check: agent is PPOAgent in this branch, which has buffer attribute
            if hasattr(agent, 'buffer'):
                agent.buffer.add(obs.squeeze(0).cpu(), action.cpu(), logprob.cpu(), reward, float(done), value.cpu())  # type: ignore[attr-defined]
            
            obs = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Type check: agent is PPOAgent in this branch
            if done or (hasattr(agent, 'buffer') and len(agent.buffer.rewards) >= 128):  # rollout_horizon  # type: ignore[attr-defined]
                with torch.no_grad():
                    net_output = agent.net(obs)
                    # Extract next_value from network output
                    if isinstance(net_output, tuple):
                        if len(net_output) >= 2:
                            _, next_value_raw = net_output
                        elif len(net_output) == 1:
                            next_value_raw = net_output[0]
                        else:
                            next_value_raw = torch.tensor(0.0, device=obs.device)
                    else:
                        next_value_raw = net_output
                    
                    # Ensure next_value is definitely a Tensor
                    if isinstance(next_value_raw, tuple):
                        if len(next_value_raw) > 0:
                            first_item = next_value_raw[0]
                            if isinstance(first_item, torch.Tensor):
                                next_value = first_item
                            elif isinstance(first_item, tuple):
                                # Handle nested tuple (shouldn't happen, but be safe)
                                next_value = torch.tensor(0.0, device=obs.device)
                            else:
                                next_value = torch.tensor(float(first_item), device=obs.device)
                        else:
                            next_value = torch.tensor(0.0, device=obs.device)
                    elif isinstance(next_value_raw, torch.Tensor):
                        next_value = next_value_raw
                    else:
                        next_value = torch.tensor(float(next_value_raw), device=obs.device)
                # Type assertion: next_value is now definitely a Tensor
                assert isinstance(next_value, torch.Tensor), "next_value must be a Tensor"
                # Type cast to satisfy type checker (we've verified it's a Tensor above)
                next_value_tensor = cast(torch.Tensor, next_value)
                logs = agent.update(next_value_tensor)
                if isinstance(logs, dict):
                    logger.log(step, logs)
                else:
                    logger.log(step, {"update": float(logs)})
        
        ep_return += reward
        ep_len += 1
        
        if done:
            episodes += 1
            total_steps_taken += ep_len
            if terminated and info.get("distance", 1) == 0:
                successes += 1
            logger.log(step, {"return": ep_return, "length": ep_len})
            obs, _ = env.reset()
            if algorithm == "dqn":
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            ep_return = 0.0
            ep_len = 0
            
            if episodes > 0 and episodes % 10 == 0:
                success_rate = successes / episodes
                avg_steps = total_steps_taken / episodes
                logger.log(step, {"success_rate": success_rate, "avg_steps": avg_steps})
    
    logger.flush()
    logger.plot()
    
    # Save agent
    if algorithm == "dqn":
        torch.save(agent.net.state_dict(), log_dir / "dqn_agent.pt")
    else:
        torch.save(agent.net.state_dict(), log_dir / "ppo_agent.pt")
    
    # Final metrics
    success_rate = successes / episodes if episodes > 0 else 0.0
    avg_steps = total_steps_taken / episodes if episodes > 0 else 0.0
    
    env.close()
    
    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "total_training_steps": total_steps,
    }


def evaluate_agent(algorithm: str, maze_data: dict, checkpoint_path: Path, num_episodes: int = 10, render_mode: str = "none", max_steps: int = 500):
    """Evaluate a trained agent on the maze."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create environment with correct flatten setting
    flatten = (algorithm == "dqn")
    env = make_env_from_maze(maze_data, render_mode, max_steps, flatten=flatten)
    obs_shape = env.observation_space.shape
    if obs_shape is None:
        raise ValueError("Observation space shape is None")
    action_space_n = env.action_space.n  # type: ignore
    
    if algorithm == "dqn":
        agent = DQNAgent(obs_shape, action_space_n, device)
        agent.net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:  # ppo
        use_cnn = len(obs_shape) == 3
        agent = PPOAgent(obs_shape, action_space_n, device, use_cnn)
        agent.net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    successes = 0
    total_steps = 0
    returns = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_steps = 0
        
        while not done:
            if algorithm == "dqn":
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
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
            else:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
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
            
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            ep_steps += 1
            done = terminated or truncated
        
        returns.append(ep_return)
        total_steps += ep_steps
        if terminated and info.get("distance", 1) == 0:
            successes += 1
    
    env.close()
    
    return {
        "episodes": num_episodes,
        "successes": successes,
        "success_rate": successes / num_episodes,
        "avg_steps": total_steps / num_episodes,
        "avg_return": np.mean(returns),
    }


def main():
    parser = argparse.ArgumentParser(description="Test RL algorithms on custom mazes")
    parser.add_argument("--maze", type=Path, required=True, help="Path to maze JSON file")
    parser.add_argument("--algorithm", choices=["ppo", "dqn"], required=True, help="Algorithm to use")
    parser.add_argument("--mode", choices=["train", "evaluate"], default="train", 
                       help="Train new agent or evaluate existing checkpoint")
    parser.add_argument("--checkpoint", type=Path, help="Path to checkpoint (for evaluate mode)")
    parser.add_argument("--total-steps", type=int, default=50000, help="Training steps (for train mode). Use smaller values (1000-5000) for quick testing.")
    parser.add_argument("--episodes", type=int, default=10, help="Evaluation episodes (for evaluate mode)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--render-mode", choices=["none", "human", "rgb_array"], default="none")
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()
    
    # Load maze
    if not args.maze.exists():
        print(f"Error: Maze file {args.maze} not found")
        sys.exit(1)
    
    maze_data = load_maze(args.maze)
    print(f"Loaded maze: {maze_data['grid_size']}x{maze_data['grid_size']}")
    print(f"Start: {maze_data['start']}, Goal: {maze_data['goal']}")
    print(f"Walls: {np.sum(maze_data['walls'])}, Obstacles: {np.sum(maze_data['obstacles'])}")
    
    set_seed(args.seed)
    
    if args.mode == "train":
        log_dir = args.log_dir / args.algorithm / f"maze_{args.maze.stem}" / f"seed{args.seed}"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nTraining {args.algorithm.upper()} on custom maze...")
        metrics = train_and_test(args.algorithm, maze_data, args.total_steps, args.seed, log_dir, args.render_mode, args.max_steps)
        
        print("\n" + "="*50)
        print("Training Results:")
        print("="*50)
        print(f"Episodes: {metrics['episodes']}")
        print(f"Successes: {metrics['successes']}")
        print(f"Success Rate: {metrics['success_rate']:.2%}")
        print(f"Average Steps to Goal: {metrics['avg_steps']:.2f}")
        print(f"Total Training Steps: {metrics['total_training_steps']}")
        print(f"\nCheckpoint saved to: {log_dir}")
        
    else:  # evaluate
        if args.checkpoint is None or not args.checkpoint.exists():
            print(f"Error: Checkpoint file {args.checkpoint} not found")
            sys.exit(1)
        
        print(f"\nEvaluating {args.algorithm.upper()} on custom maze...")
        metrics = evaluate_agent(args.algorithm, maze_data, args.checkpoint, args.episodes, args.render_mode, args.max_steps)
        
        print("\n" + "="*50)
        print("Evaluation Results:")
        print("="*50)
        print(f"Episodes: {metrics['episodes']}")
        print(f"Successes: {metrics['successes']}")
        print(f"Success Rate: {metrics['success_rate']:.2%}")
        print(f"Average Steps to Goal: {metrics['avg_steps']:.2f}")
        print(f"Average Return: {metrics['avg_return']:.2f}")


if __name__ == "__main__":
    main()

