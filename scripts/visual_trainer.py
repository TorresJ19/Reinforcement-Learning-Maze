import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, cast

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pygame
import torch

from src.agents.dqn import DQNAgent
from src.agents.ppo import PPOAgent
from src.envs.maze_env import MazeEnv, MazeConfig
from src.utils.logger import Logger, save_config
from src.utils.replay_buffer import ReplayBuffer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_maze(maze_path: Path) -> dict:
    """Load maze configuration from JSON file."""
    with open(maze_path, "r") as f:
        return json.load(f)


def make_env_from_maze(maze_data: dict, render_mode: str = "human", max_steps: int = 500, flatten: bool = False) -> MazeEnv:
    """Create environment from custom maze data."""
    grid_size = maze_data["grid_size"]
    walls = np.array(maze_data["walls"], dtype=np.bool_)
    obstacles = np.array(maze_data["obstacles"], dtype=np.bool_)
    start = tuple(maze_data["start"])
    goal = tuple(maze_data["goal"])
    
    cfg = MazeConfig(
        grid_size=(grid_size, grid_size),
        wall_fraction=0.0,
        obstacle_fraction=0.0,
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


class VisualTrainer:
    def __init__(self, algorithm: str, maze_data: dict, total_steps: int, seed: int, max_steps: int = 500):
        self.algorithm = algorithm
        self.maze_data = maze_data
        self.total_steps = total_steps
        self.seed = seed
        self.max_steps = max_steps
        
        # Statistics
        self.episodes = 0
        self.successes = 0
        self.total_steps_taken = 0
        self.current_episode_steps = 0
        self.current_episode_return = 0.0
        self.episode_returns = []
        self.episode_lengths = []
        self.best_path_length = None  # Track best path length across episodes
        
        # Initialize pygame first (before creating environment)
        pygame.init()
        pygame.font.init()
        
        # Training state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.flatten = (algorithm == "dqn")
        self.env = make_env_from_maze(maze_data, "human", max_steps, flatten=self.flatten)
        obs_shape = self.env.observation_space.shape
        
        if algorithm == "dqn":
            action_space_n = self.env.action_space.n  # type: ignore[attr-defined]
            self.agent = DQNAgent(obs_shape, action_space_n, self.device)
            self.buffer = ReplayBuffer(50000, obs_shape, self.device)
        else:  # ppo
            # Ensure obs_shape is not None before checking length
            if obs_shape is None:
                raise ValueError("Observation space shape is None")
            use_cnn = len(obs_shape) == 3
            # Use better hyperparameters for maze learning
            from src.agents.ppo import PPOConfig
            # Optimize mini_batch_size for rollout_horizon=64:
            # - rollout=64, mini_batch=16 → 4 mini-batches per epoch → 16 total gradient updates
            # This gives better learning than 2 mini-batches (with mini_batch=32)
            # Use adaptive entropy that starts high and decays slowly to prevent getting stuck
            ppo_config = PPOConfig(
                lr=3e-4,
                entropy_coef=0.3,  # Start with high entropy for strong exploration
                mini_batch_size=16,  # Optimized: 64/16 = 4 mini-batches per epoch
                update_epochs=4,  # Keep standard update epochs
            )
            action_space_n = self.env.action_space.n  # type: ignore[attr-defined]
            self.agent = PPOAgent(obs_shape, action_space_n, self.device, use_cnn, config=ppo_config)
            self.initial_entropy = 0.3
            self.min_entropy = 0.05  # Minimum entropy to maintain some exploration
            self.entropy_decay_steps = 10000  # Decay over 10k steps
        
        # Observation state
        self.obs = None
        self.obs_t = None
        
        # Pygame UI setup (after pygame is initialized)
        self.font = pygame.font.SysFont("Segoe UI", 18)
        self.title_font = pygame.font.SysFont("Segoe UI", 24, bold=True)
        self.small_font = pygame.font.SysFont("Segoe UI", 14)
        
        # Animation settings
        self.step_delay = 0.05  # Delay between steps in seconds
        self.paused = False
        self.speed_multiplier = 1.0
        
        # Path tracking
        self.current_path = []
        self.all_paths = []
        
        # Stuck detection
        self.recent_positions = []  # Track recent positions to detect loops
        self.stuck_threshold = 20  # If same position repeated 20 times, consider stuck
        self.stuck_penalty_count = 0
        
    def draw_statistics(self, surface):
        """Draw enterprise-grade statistics dashboard."""
        width, height = surface.get_size()
        
        # Modern glass-morphism overlay
        overlay_height = 220
        overlay = pygame.Surface((width, overlay_height), pygame.SRCALPHA)
        # Gradient background
        for y in range(overlay_height):
            alpha = int(220 - (y / overlay_height) * 40)
            overlay.fill((10, 15, 25, alpha), (0, y, width, 1))
        surface.blit(overlay, (0, 0))
        
        # Top border accent
        pygame.draw.rect(surface, (99, 102, 241), (0, 0, width, 3))
        
        # Title with badge
        title_bg = pygame.Surface((400, 40), pygame.SRCALPHA)
        pygame.draw.rect(title_bg, (99, 102, 241, 100), title_bg.get_rect(), border_radius=8)
        surface.blit(title_bg, (20, 15))
        
        title = self.title_font.render(f"{self.algorithm.upper()} Training", True, (255, 255, 255))
        surface.blit(title, (30, 20))
        
        episode_badge = self.font.render(f"Episode {self.episodes + 1}", True, (255, 255, 255))
        badge_bg = pygame.Surface((episode_badge.get_width() + 20, 25), pygame.SRCALPHA)
        pygame.draw.rect(badge_bg, (139, 92, 246, 150), badge_bg.get_rect(), border_radius=12)
        surface.blit(badge_bg, (440, 18))
        surface.blit(episode_badge, (450, 21))
        
        # Statistics cards in grid layout
        stats_y = 70
        card_width = 280
        card_height = 60
        card_spacing = 20
        cards_per_row = 3
        
        success_rate = (self.successes / self.episodes * 100) if self.episodes > 0 else 0.0
        avg_steps = self.total_steps_taken / self.episodes if self.episodes > 0 else 0.0
        avg_return = np.mean(self.episode_returns) if self.episode_returns else 0.0
        best_text = f"{self.best_path_length}" if self.best_path_length is not None else "N/A"
        
        # Get entropy or epsilon
        if self.algorithm == "ppo" and hasattr(self.agent, 'config') and hasattr(self.agent.config, 'entropy_coef'):
            exploration_val = self.agent.config.entropy_coef  # type: ignore
            exploration_label = "Entropy"
        elif self.algorithm == "dqn" and hasattr(self.agent, 'config'):
            exploration_val = self.agent._epsilon() if hasattr(self.agent, '_epsilon') else 0.0  # type: ignore
            exploration_label = "Epsilon"
        else:
            exploration_val = 0.0
            exploration_label = "Exploration"
        
        stats_cards = [
            {"label": "Success Rate", "value": f"{success_rate:.1f}%", "color": (34, 197, 94)},
            {"label": "Avg Steps", "value": f"{avg_steps:.1f}", "color": (99, 102, 241)},
            {"label": "Best Path", "value": f"{best_text} steps", "color": (139, 92, 246)},
            {"label": "Current Steps", "value": f"{self.current_episode_steps}/{self.max_steps}", "color": (251, 191, 36)},
            {"label": exploration_label, "value": f"{exploration_val:.3f}", "color": (59, 130, 246)},
            {"label": "Return", "value": f"{self.current_episode_return:.2f}", "color": (236, 72, 153)},
        ]
        
        for i, card in enumerate(stats_cards):
            row = i // cards_per_row
            col = i % cards_per_row
            x = 20 + col * (card_width + card_spacing)
            y = stats_y + row * (card_height + 15)
            
            # Card background
            card_surf = pygame.Surface((card_width, card_height), pygame.SRCALPHA)
            pygame.draw.rect(card_surf, (20, 25, 35, 200), card_surf.get_rect(), border_radius=10)
            pygame.draw.rect(card_surf, (card["color"][0], card["color"][1], card["color"][2], 100), 
                           card_surf.get_rect(), width=2, border_radius=10)
            surface.blit(card_surf, (x, y))
            
            # Left accent bar
            pygame.draw.rect(surface, card["color"], (x, y, 4, card_height), border_radius=2)
            
            # Label
            label = self.small_font.render(card["label"], True, (156, 163, 175))
            surface.blit(label, (x + 12, y + 8))
            
            # Value
            value = self.font.render(card["value"], True, (255, 255, 255))
            surface.blit(value, (x + 12, y + 28))
        
        # Controls bar at bottom
        controls_y = height - 50
        controls_bg = pygame.Surface((width, 50), pygame.SRCALPHA)
        controls_bg.fill((10, 15, 25, 200))
        surface.blit(controls_bg, (0, controls_y))
        pygame.draw.rect(surface, (99, 102, 241), (0, controls_y, width, 2))
        
        controls = "SPACE: Pause/Resume  |  +/-: Speed  |  R: Reset  |  ESC: Quit"
        control_text = self.small_font.render(controls, True, (156, 163, 175))
        surface.blit(control_text, (20, controls_y + 15))
        
        # Progress bar
        progress = min(1.0, self.current_episode_steps / self.max_steps)
        bar_width = width - 40
        bar_x, bar_y = 20, height - 30
        bar_height = 6
        
        # Background
        pygame.draw.rect(surface, (30, 41, 59), (bar_x, bar_y, bar_width, bar_height), border_radius=3)
        
        # Progress fill with gradient effect
        if progress > 0:
            progress_width = int(bar_width * progress)
            # Color based on progress
            if progress < 0.5:
                color = (34, 197, 94)  # Green
            elif progress < 0.8:
                color = (251, 191, 36)  # Amber
            else:
                color = (239, 68, 68)  # Red
            
            pygame.draw.rect(surface, color, (bar_x, bar_y, progress_width, bar_height), border_radius=3)
            
            # Glow effect
            glow_surf = pygame.Surface((progress_width, bar_height), pygame.SRCALPHA)
            glow_surf.fill((*color, 50))
            surface.blit(glow_surf, (bar_x, bar_y))
    
    def draw_path(self, surface):
        """Draw the agent's path on the maze."""
        if not self.current_path:
            return
        
        mx, my, mw, mh = self.env.maze_viewport
        cell = self.env.cell_px
        
        # Draw path as a trail
        for i, (r, c) in enumerate(self.current_path):
            if i == 0:
                continue  # Skip start position
            
            prev_r, prev_c = self.current_path[i - 1]
            x1 = mx + prev_c * cell + cell // 2
            y1 = my + prev_r * cell + cell // 2
            x2 = mx + c * cell + cell // 2
            y2 = my + r * cell + cell // 2
            
            # Fade trail based on position
            alpha = int(255 * (1 - i / len(self.current_path) * 0.5))
            color = (99, 223, 156, alpha)
            
            # Draw line
            if len(self.current_path) > 1:
                pygame.draw.line(surface, (99, 223, 156), (x1, y1), (x2, y2), width=2)
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.speed_multiplier = min(5.0, self.speed_multiplier + 0.5)
                elif event.key == pygame.K_MINUS:
                    self.speed_multiplier = max(0.1, self.speed_multiplier - 0.5)
                elif event.key == pygame.K_r:
                    # Reset current episode
                    obs, _ = self.env.reset()
                    if self.algorithm == "dqn":
                        self.obs = obs
                        self.obs_t = torch.tensor(self.obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    else:
                        self.obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    self.current_path = []
                    self.current_episode_steps = 0
                    self.current_episode_return = 0.0
        return True
    
    def train(self):
        """Main training loop with visualization."""
        set_seed(self.seed)
        
        obs, _ = self.env.reset(seed=self.seed)
        if self.algorithm == "dqn":
            self.obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.obs = obs
        else:
            self.obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        running = True
        step_count = 0
        
        while running and step_count < self.total_steps:
            # Handle events
            running = self.handle_events()
            if not running:
                break
            
            # Skip if paused
            if self.paused:
                self.env.render()
                self.draw_path(self.env.screen)
                self.draw_statistics(self.env.screen)
                pygame.display.flip()
                time.sleep(0.1)
                continue
            
            # Get action
            if self.algorithm == "dqn":
                if self.obs_t is None:
                    continue
                action_result = self.agent.act(self.obs_t)
                if isinstance(action_result, int):
                    action = action_result
                elif isinstance(action_result, torch.Tensor):
                    action = int(action_result.item())
                else:
                    # Handle unexpected tuple return (shouldn't happen for DQN)
                    if isinstance(action_result, tuple):
                        if len(action_result) > 0 and isinstance(action_result[0], torch.Tensor):
                            action = int(action_result[0].item())
                        elif len(action_result) > 0:
                            action = int(action_result[0])
                        else:
                            raise ValueError(f"Unexpected empty tuple from DQN agent.act()")
                    else:
                        action = int(action_result)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                self.buffer.push(self.obs, action, reward, next_obs, float(done))
                
                self.obs = next_obs
                self.obs_t = torch.tensor(self.obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                if len(self.buffer) >= self.agent.config.batch_size:  # type: ignore
                    batch = self.buffer.sample(self.agent.config.batch_size)  # type: ignore
                    self.agent.update(batch)  # type: ignore[arg-type]  # DQN update takes batch, not next_value
            else:  # ppo
                if self.obs is None:
                    continue
                # Ensure obs is a Tensor before calling agent.act()
                if not isinstance(self.obs, torch.Tensor):
                    self.obs = torch.tensor(self.obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action_tuple = self.agent.act(self.obs)
                    if isinstance(action_tuple, tuple) and len(action_tuple) == 3:
                        action, logprob, value = action_tuple
                    else:
                        raise ValueError(f"Expected tuple of 3, got {action_tuple}")
                next_obs, reward, terminated, truncated, info = self.env.step(int(action.item()))
                done = terminated or truncated
                # Type check: agent is PPOAgent in this branch, which has buffer attribute
                if hasattr(self.agent, 'buffer'):
                    self.agent.buffer.add(self.obs.squeeze(0).cpu(), action.cpu(), logprob.cpu(), reward, float(done), value.cpu())  # type: ignore[attr-defined]
                
                self.obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                # Update more frequently for better learning (shorter rollouts)
                rollout_horizon = 64  # Shorter rollouts = more frequent updates
                # Type check: agent is PPOAgent in this branch
                if done or (hasattr(self.agent, 'buffer') and len(self.agent.buffer.rewards) >= rollout_horizon):  # type: ignore[attr-defined]
                    with torch.no_grad():
                        output = self.agent.net(self.obs)
                        # Extract next_value from network output
                        if isinstance(output, tuple):
                            if len(output) >= 2:
                                _, next_value_raw = output
                            elif len(output) == 1:
                                next_value_raw = output[0]
                            else:
                                next_value_raw = torch.tensor(0.0, device=self.obs.device)
                        else:
                            next_value_raw = output
                        
                        # Ensure next_value is definitely a Tensor
                        if isinstance(next_value_raw, tuple):
                            if len(next_value_raw) > 0:
                                first_item = next_value_raw[0]
                                if isinstance(first_item, torch.Tensor):
                                    next_value = first_item
                                elif isinstance(first_item, tuple):
                                    # Handle nested tuple 
                                    next_value = torch.tensor(0.0, device=self.obs.device)
                                else:
                                    next_value = torch.tensor(float(first_item), device=self.obs.device)
                            else:
                                next_value = torch.tensor(0.0, device=self.obs.device)
                        elif isinstance(next_value_raw, torch.Tensor):
                            next_value = next_value_raw
                        else:
                            next_value = torch.tensor(float(next_value_raw), device=self.obs.device)
                    # Type assertion: next_value is now definitely a Tensor
                    assert isinstance(next_value, torch.Tensor), "next_value must be a Tensor"
                    # Type cast to satisfy type checker (we've verified it's a Tensor above)
                    next_value_tensor = cast(torch.Tensor, next_value)
                    update_info = self.agent.update(next_value_tensor)
                    # Print update info occasionally for debugging
                    if isinstance(update_info, dict) and self.episodes % 5 == 0 and done:
                        print(f"Episode {self.episodes}: Policy Loss: {update_info.get('policy_loss', 0):.4f}, "
                              f"Value Loss: {update_info.get('value_loss', 0):.4f}, "
                              f"Entropy: {update_info.get('entropy', 0):.4f}")
            
            # Update statistics
            self.current_episode_steps += 1
            self.current_episode_return += reward
            step_count += 1
            
            # Track path
            ar, ac = self.env.agent_pos
            if not self.current_path or self.current_path[-1] != (ar, ac):
                self.current_path.append((ar, ac))
            
            # Detect if stuck (repeating same position)
            self.recent_positions.append((ar, ac))
            if len(self.recent_positions) > self.stuck_threshold:
                self.recent_positions.pop(0)
            
            # If stuck in same position, add exploration boost
            if len(self.recent_positions) >= self.stuck_threshold:
                if len(set(self.recent_positions[-self.stuck_threshold:])) <= 3:  # Only 3 unique positions
                    # Boost exploration temporarily to escape
                    if self.algorithm == "ppo" and hasattr(self.agent.config, 'entropy_coef'):
                        original_entropy = self.agent.config.entropy_coef  # type: ignore[attr-defined]
                        self.agent.config.entropy_coef = min(0.5, original_entropy * 1.5)  # type: ignore[attr-defined]
                    # For DQN, epsilon is handled internally, so we just track stuck detections
                    self.stuck_penalty_count += 1
                else:
                    # Gradually reduce exploration as training progresses (only for PPO)
                    if self.algorithm == "ppo" and hasattr(self, 'initial_entropy') and hasattr(self.agent.config, 'entropy_coef'):
                        progress = min(1.0, step_count / self.entropy_decay_steps)
                        self.agent.config.entropy_coef = self.initial_entropy * (1 - progress) + self.min_entropy * progress  # type: ignore[attr-defined]
            
            # Render
            self.env.render()
            self.draw_path(self.env.screen)
            self.draw_statistics(self.env.screen)
            pygame.display.flip()
            
            # Delay for visualization
            time.sleep(self.step_delay / self.speed_multiplier)
            
            # Handle episode end
            if done:
                self.episodes += 1
                self.total_steps_taken += self.current_episode_steps
                
                if terminated and info.get("distance", 1) == 0:
                    self.successes += 1
                    
                    # Track best path
                    if self.best_path_length is None or self.current_episode_steps < self.best_path_length:
                        self.best_path_length = self.current_episode_steps
                        # Show improved path message
                    success_text = self.title_font.render(
                        f"SUCCESS! New Best: {self.current_episode_steps} steps!", 
                        True, (99, 223, 156)
                    )
                else:
                    success_text = self.title_font.render(
                        f"SUCCESS! ({self.current_episode_steps} steps, Best: {self.best_path_length})", 
                        True, (99, 223, 156)
                    )
                
                if self.env.screen is not None:
                    if self.env.screen is not None:
                        text_rect = success_text.get_rect(center=(self.env.screen.get_width() // 2, 250))
                        self.env.screen.blit(success_text, text_rect)
                    pygame.display.flip()
                    time.sleep(0.5)  # Shorter delay to keep training moving
                
                self.episode_returns.append(self.current_episode_return)
                self.episode_lengths.append(self.current_episode_steps)
                self.all_paths.append(self.current_path.copy())
                
                # Reset for next episode
                obs, _ = self.env.reset()
                if self.algorithm == "dqn":
                    self.obs = obs
                    self.obs_t = torch.tensor(self.obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                else:
                    self.obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                self.current_path = []
                self.current_episode_steps = 0
                self.current_episode_return = 0.0
                self.recent_positions = []  # Reset stuck detection
        
        self.env.close()
        
        # Print final statistics
        print("\n" + "="*50)
        print("Training Complete!")
        print("="*50)
        print(f"Total Episodes: {self.episodes}")
        print(f"Successes: {self.successes}")
        print(f"Success Rate: {(self.successes / self.episodes * 100) if self.episodes > 0 else 0:.1f}%")
        print(f"Average Steps: {self.total_steps_taken / self.episodes if self.episodes > 0 else 0:.1f}")
        print(f"Average Return: {np.mean(self.episode_returns) if self.episode_returns else 0:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Visual training interface for RL algorithms on custom mazes")
    parser.add_argument("--maze", type=Path, required=True, help="Path to maze JSON file")
    parser.add_argument("--algorithm", choices=["ppo", "dqn"], required=True, help="Algorithm to use")
    parser.add_argument("--total-steps", type=int, default=5000, help="Training steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    args = parser.parse_args()
    
    # Load maze
    if not args.maze.exists():
        print(f"Error: Maze file {args.maze} not found")
        sys.exit(1)
    
    maze_data = load_maze(args.maze)
    print(f"Loaded maze: {maze_data['grid_size']}x{maze_data['grid_size']}")
    print(f"Start: {maze_data['start']}, Goal: {maze_data['goal']}")
    print(f"Walls: {np.sum(maze_data['walls'])}, Obstacles: {np.sum(maze_data['obstacles'])}")
    
    # Create visual trainer
    trainer = VisualTrainer(args.algorithm, maze_data, args.total_steps, args.seed, args.max_steps)
    trainer.train()


if __name__ == "__main__":
    main()

