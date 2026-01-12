import os
import random
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


Move = Tuple[int, int]


@dataclass
class MazeConfig:
    grid_size: Tuple[int, int] = (10, 10)
    wall_fraction: float = 0.15
    obstacle_fraction: float = 0.10
    dynamic_obstacles: int = 0
    max_steps: int = 200
    reward_goal: float = 10.0
    reward_step: float = -0.1  # Reduced step penalty to encourage exploration
    reward_collision: float = -2.0  # Reduced collision penalty
    reward_efficiency: float = 5.0  # Bonus for efficient paths (steps saved)
    shaping: bool = True
    flatten: bool = False
    render_mode: str = "none"  # "none" | "human" | "rgb_array"
    seed: Optional[int] = None
    # Custom maze layout
    custom_walls: Optional[np.ndarray] = None  # Boolean array of wall positions
    custom_obstacles: Optional[np.ndarray] = None  # Boolean array of obstacle positions
    custom_start: Optional[Tuple[int, int]] = None  # Custom start position
    custom_goal: Optional[Tuple[int, int]] = None  # Custom goal position


class MazeEnv(gym.Env):
    metadata = {"render_modes": ["none", "human", "rgb_array"], "render_fps": 15}

    def __init__(self, config: MazeConfig = MazeConfig()):
        super().__init__()
        self.config = config
        self.rows, self.cols = config.grid_size
        self.rng = np.random.default_rng(config.seed)
        self.action_space = spaces.Discrete(4)
        obs_shape = (self.rows, self.cols, 4)
        if config.flatten:
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(int(np.prod(obs_shape)),), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=obs_shape, dtype=np.float32)
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.goal_pos: Tuple[int, int] = (self.rows - 1, self.cols - 1)
        self.walls: np.ndarray = np.zeros((self.rows, self.cols), dtype=np.bool_)
        self.static_obstacles: np.ndarray = np.zeros((self.rows, self.cols), dtype=np.bool_)
        self.obstacles: np.ndarray = np.zeros((self.rows, self.cols), dtype=np.bool_)  # static + dynamic
        self.dynamic_paths: List[List[Tuple[int, int]]] = []
        self.dynamic_indices: List[int] = []
        self.steps = 0
        self.collided_dynamic = False
        self.best_path_length: Optional[int] = None  # Track best path length for optimization
        self.visited_positions: set = set()  # Track visited positions for exploration bonus
        self.last_position: Optional[Tuple[int, int]] = None
        self.position_repeat_count = 0
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.font: Optional[pygame.font.Font] = None
        # Fixed window size - doesn't change with grid size
        self.window_size: Tuple[int, int] = (1024, 768)
        self.cell_px: int = 44
        self.maze_viewport: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x, y, width, height)
        self.theme = {
            "bg_top": (18, 27, 38),
            "bg_bottom": (10, 15, 25),
            "grid": (45, 58, 74),
            "wall": (82, 97, 115),
            "obstacle": (235, 111, 104),
            "obstacle_dyn": (255, 165, 122),
            "agent": (99, 223, 156),
            "agent_outline": (46, 179, 113),
            "goal": (255, 214, 102),
            "goal_glow": (255, 239, 191),
            "text": (220, 230, 242),
        }
        self._compute_layout()
        if self.config.render_mode == "none":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        self._make_board()

    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            self.config.seed = seed
        self.rng = np.random.default_rng(self.config.seed)
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

    def _make_board(self):
        self.walls[:] = False
        self.obstacles[:] = False
        self.static_obstacles[:] = False
        
        # Use custom layout if provided, otherwise generate randomly
        if self.config.custom_walls is not None:
            assert self.config.custom_walls.shape == (self.rows, self.cols), \
                f"Custom walls shape {self.config.custom_walls.shape} doesn't match grid size {(self.rows, self.cols)}"
            self.walls = self.config.custom_walls.copy()
        else:
            wall_count = int(self.rows * self.cols * self.config.wall_fraction)
            obstacle_count = int(self.rows * self.cols * self.config.obstacle_fraction)
            free_cells = [(r, c) for r in range(self.rows) for c in range(self.cols) if (r, c) not in [self.agent_pos, self.goal_pos]]
            self.rng.shuffle(free_cells)
            for r, c in free_cells[:wall_count]:
                if (r, c) in [self.agent_pos, self.goal_pos]:
                    continue
                self.walls[r, c] = True
        
        if self.config.custom_obstacles is not None:
            assert self.config.custom_obstacles.shape == (self.rows, self.cols), \
                f"Custom obstacles shape {self.config.custom_obstacles.shape} doesn't match grid size {(self.rows, self.cols)}"
            self.static_obstacles = self.config.custom_obstacles.copy()
        elif self.config.custom_walls is None:  # Only generate random obstacles if not using custom walls
            remaining = [(r, c) for r in range(self.rows) for c in range(self.cols) 
                        if not self.walls[r, c] and (r, c) not in [self.agent_pos, self.goal_pos]]
            self.rng.shuffle(remaining)
            obstacle_count = int(self.rows * self.cols * self.config.obstacle_fraction)
            for r, c in remaining[:obstacle_count]:
                if (r, c) in [self.agent_pos, self.goal_pos]:
                    continue
                self.static_obstacles[r, c] = True
        
        # Ensure walls/obstacles don't block start/goal
        if self.agent_pos:
            self.walls[self.agent_pos[0], self.agent_pos[1]] = False
            self.static_obstacles[self.agent_pos[0], self.agent_pos[1]] = False
        if self.goal_pos:
            self.walls[self.goal_pos[0], self.goal_pos[1]] = False
            self.static_obstacles[self.goal_pos[0], self.goal_pos[1]] = False
        
        self._init_dynamic_obstacles()
        self._update_obstacle_map()

    def _init_dynamic_obstacles(self):
        self.dynamic_paths.clear()
        self.dynamic_indices.clear()
        if self.config.dynamic_obstacles <= 0:
            return
        self.collided_dynamic = False
        empties = [(r, c) for r in range(self.rows) for c in range(self.cols) if not self.walls[r, c] and (r, c) not in [self.agent_pos, self.goal_pos]]
        self.rng.shuffle(empties)
        for start in empties[: self.config.dynamic_obstacles]:
            path = self._generate_loop_path(start)
            if path:
                self.dynamic_paths.append(path)
                self.dynamic_indices.append(0)

    def _generate_loop_path(self, start: Tuple[int, int]) -> List[Tuple[int, int]]:
        # Simple deterministic loop: right -> down -> left -> up
        r, c = start
        path = []
        directions: List[Move] = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and not self.walls[nr, nc]:
                path.append((nr, nc))
        if not path:
            return []
        return [start] + path

    def reset(self, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.seed(seed)
        # Use custom positions if provided, otherwise use defaults
        if self.config.custom_start is not None:
            self.agent_pos = self.config.custom_start
        else:
            self.agent_pos = (0, 0)
        if self.config.custom_goal is not None:
            self.goal_pos = self.config.custom_goal
        else:
            self.goal_pos = (self.rows - 1, self.cols - 1)
        self.steps = 0
        self.best_path_length = None  # Reset best path when starting new episode
        self.visited_positions = set()
        self.last_position = None
        self.position_repeat_count = 0
        self._make_board()
        
        # Validate that start and goal positions are valid (not blocked)
        if not self._is_valid(self.agent_pos):
            # If start is blocked, find nearest valid position
            for r in range(self.rows):
                for c in range(self.cols):
                    if self._is_valid((r, c)) and (r, c) != self.goal_pos:
                        self.agent_pos = (r, c)
                        break
                if self._is_valid(self.agent_pos):
                    break
        
        if not self._is_valid(self.goal_pos):
            # If goal is blocked, find nearest valid position
            for r in range(self.rows - 1, -1, -1):
                for c in range(self.cols - 1, -1, -1):
                    if self._is_valid((r, c)) and (r, c) != self.agent_pos:
                        self.goal_pos = (r, c)
                        break
                if self._is_valid(self.goal_pos):
                    break
        
        # Ensure start/goal are not blocked
        self.walls[self.agent_pos[0], self.agent_pos[1]] = False
        self.obstacles[self.agent_pos[0], self.agent_pos[1]] = False
        self.walls[self.goal_pos[0], self.goal_pos[1]] = False
        self.obstacles[self.goal_pos[0], self.goal_pos[1]] = False
        self._update_obstacle_map()
        
        self._maybe_init_pygame()
        obs = self._get_obs()
        info = {"distance": self._manhattan(self.agent_pos, self.goal_pos)}
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action)
        self.steps += 1
        reward = self.config.reward_step
        terminated = False
        truncated = False
        prev_dist = self._manhattan(self.agent_pos, self.goal_pos)
        move = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]  # Up, Down, Left, Right
        new_pos = (self.agent_pos[0] + move[0], self.agent_pos[1] + move[1])
        
        # Check if move is valid (not out of bounds, not a wall, not an obstacle)
        if self._is_valid(new_pos):
            self.agent_pos = new_pos
            
            # Exploration bonus: reward for visiting new positions
            if self.agent_pos not in self.visited_positions:
                reward += 0.1  # Small bonus for exploring new areas
                self.visited_positions.add(self.agent_pos)
            
            # Penalty for getting stuck (repeating same position)
            if self.agent_pos == self.last_position:
                self.position_repeat_count += 1
                if self.position_repeat_count > 5:  # Stuck in same spot
                    reward -= 0.5 * (self.position_repeat_count - 5)  # Increasing penalty
            else:
                self.position_repeat_count = 0
            
            self.last_position = self.agent_pos
        else:
            # Invalid move: give penalty but don't terminate (allows learning from mistakes)
            reward += self.config.reward_collision
            # Don't terminate on collision - let agent learn to avoid walls
        
        self.collided_dynamic = False
        self._advance_dynamic_obstacles()
        if self.collided_dynamic:
            reward += self.config.reward_collision
            # Don't terminate on dynamic obstacle collision either
        
        # Check if reached goal
        if self.agent_pos == self.goal_pos:
            reward += self.config.reward_goal
            
            # Bonus reward for efficient paths (encourage optimization)
            # Give bonus if this is the best path so far, or if it's shorter than previous best
            if self.best_path_length is None or self.steps < self.best_path_length:
                steps_saved = (self.best_path_length - self.steps) if self.best_path_length is not None else 0
                efficiency_bonus = self.config.reward_efficiency * (1.0 + steps_saved * 0.1)
                reward += efficiency_bonus
                self.best_path_length = self.steps
            
            terminated = True
        
        # Reward shaping: encourage moving closer to goal
        if self.config.shaping:
            curr_dist = self._manhattan(self.agent_pos, self.goal_pos)
            reward += (prev_dist - curr_dist) * 0.1
        
        # Timeout after max steps
        if self.steps >= self.config.max_steps:
            truncated = True
        
        obs = self._get_obs()
        info = {"distance": self._manhattan(self.agent_pos, self.goal_pos)}
        return obs, reward, terminated, truncated, info

    def _advance_dynamic_obstacles(self):
        self._update_obstacle_map()
        if not self.dynamic_paths:
            return
        for idx, path in enumerate(self.dynamic_paths):
            self.dynamic_indices[idx] = (self.dynamic_indices[idx] + 1) % len(path)
        self._update_obstacle_map()
        for path, cur_idx in zip(self.dynamic_paths, self.dynamic_indices):
            r, c = path[cur_idx]
            if (r, c) == self.agent_pos:
                self.collided_dynamic = True
                break

    def _update_obstacle_map(self):
        self.obstacles = self.static_obstacles.copy()
        for path, cur_idx in zip(self.dynamic_paths, self.dynamic_indices):
            r, c = path[cur_idx]
            self.obstacles[r, c] = True

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is valid (within bounds and not blocked)."""
        r, c = pos
        # Check bounds (edges are handled correctly: 0 <= r < rows means r can be 0 to rows-1)
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False
        # Check if it's a wall
        if self.walls[r, c]:
            return False
        # Check if it's an obstacle
        if self.obstacles[r, c]:
            return False
        return True

    def _get_obs(self):
        grid = np.zeros((self.rows, self.cols, 4), dtype=np.float32)
        grid[self.agent_pos[0], self.agent_pos[1], 0] = 1.0
        grid[self.walls, 1] = 1.0
        grid[self.obstacles, 2] = 1.0
        goal_r, goal_c = self.goal_pos
        grid[goal_r, goal_c, 3] = 1.0
        if self.config.flatten:
            return grid.flatten()
        return grid

    def render(self):
        if self.config.render_mode == "none":
            return
        self._maybe_init_pygame()
        assert self.screen is not None, "Screen should be initialized after _maybe_init_pygame()"
        surface = self.screen
        # Fill entire window with background
        self._draw_gradient(surface, self.theme["bg_top"], self.theme["bg_bottom"])
        # Draw maze in viewport
        maze_x, maze_y, maze_w, maze_h = self.maze_viewport
        maze_surface = pygame.Surface((maze_w, maze_h))
        self._draw_gradient(maze_surface, self.theme["bg_top"], self.theme["bg_bottom"])
        self._draw_grid(maze_surface)
        self._draw_cells(maze_surface)
        surface.blit(maze_surface, (maze_x, maze_y))
        self._draw_overlay(surface)
        if self.config.render_mode == "human":
            pygame.display.flip()
            assert self.clock is not None, "Clock should be initialized after _maybe_init_pygame()"
            self.clock.tick(self.metadata["render_fps"])
        elif self.config.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(surface)), (1, 0, 2))

    def _maybe_init_pygame(self):
        if self.config.render_mode == "none":
            return
        if self.screen is not None:
            # Recompute layout if grid size changed
            self._compute_layout()
            return
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("Maze RL — Modern View")
        self.font = pygame.font.SysFont("Segoe UI", 18)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # --- Modern rendering helpers ---
    def _compute_layout(self):
        # Fixed window size; calculate maze viewport and cell size to fit grid
        win_w, win_h = self.window_size
        # Reserve space for UI overlay at top (100px)
        maze_h = win_h - 100
        maze_w = win_w - 40  # margins
        
        # Calculate cell size to fit grid in viewport
        cell_from_w = max(16, maze_w // max(1, self.cols))
        cell_from_h = max(16, maze_h // max(1, self.rows))
        self.cell_px = max(24, min(64, cell_from_w, cell_from_h))
        
        # Calculate maze viewport (centered)
        maze_view_w = self.cols * self.cell_px
        maze_view_h = self.rows * self.cell_px
        maze_x = (win_w - maze_view_w) // 2
        maze_y = 100 + (maze_h - maze_view_h) // 2
        self.maze_viewport = (maze_x, maze_y, maze_view_w, maze_view_h)

    def _draw_gradient(self, surface: pygame.Surface, top_color, bottom_color):
        width, height = surface.get_size()
        for y in range(height):
            ratio = y / max(1, height - 1)
            r = int(top_color[0] + (bottom_color[0] - top_color[0]) * ratio)
            g = int(top_color[1] + (bottom_color[1] - top_color[1]) * ratio)
            b = int(top_color[2] + (bottom_color[2] - top_color[2]) * ratio)
            pygame.draw.line(surface, (r, g, b), (0, y), (width, y))

    def _draw_grid(self, surface: pygame.Surface):
        color = self.theme["grid"]
        cell = self.cell_px
        width, height = surface.get_size()
        for c in range(self.cols + 1):
            x = c * cell
            pygame.draw.line(surface, color, (x, 0), (x, height), width=1)
        for r in range(self.rows + 1):
            y = r * cell
            pygame.draw.line(surface, color, (0, y), (width, y), width=1)

    def _draw_cells(self, surface: pygame.Surface):
        cell = self.cell_px
        radius = 8
        ticks = pygame.time.get_ticks()
        
        for r in range(self.rows):
            for c in range(self.cols):
                x, y = c * cell, r * cell
                rect = pygame.Rect(x + 3, y + 3, cell - 6, cell - 6)
                if self.walls[r, c]:
                    # Animated wall with subtle pulse
                    pulse = 0.95 + 0.05 * math.sin(ticks / 800 + r + c)
                    wall_color = tuple(int(c * pulse) for c in self.theme["wall"])
                    pygame.draw.rect(surface, wall_color, rect, border_radius=radius)
                    # Add texture lines
                    pygame.draw.line(surface, tuple(int(c * 0.7) for c in wall_color), 
                                   (x + 5, y + cell // 2), (x + cell - 5, y + cell // 2), width=1)
                elif self.obstacles[r, c]:
                    is_dyn = self._is_dynamic_cell(r, c)
                    base_color = self.theme["obstacle_dyn"] if is_dyn else self.theme["obstacle"]
                    # Pulsing animation for dynamic obstacles
                    if is_dyn:
                        pulse = 0.9 + 0.1 * math.sin(ticks / 300)
                        color = tuple(int(c * pulse) for c in base_color)
                    else:
                        pulse = 0.95 + 0.05 * math.sin(ticks / 1000)
                        color = tuple(int(c * pulse) for c in base_color)
                    pygame.draw.rect(surface, color, rect, border_radius=radius)
                    # Warning symbol for obstacles
                    center_x, center_y = x + cell // 2, y + cell // 2
                    pygame.draw.circle(surface, tuple(int(c * 0.6) for c in color), 
                                     (center_x, center_y), cell // 6, width=2)
                elif (r, c) == self.goal_pos:
                    # Enhanced pulsing goal with stars
                    pulse = 0.35 + 0.25 * (1 + math.sin(ticks / 350))
                    glow_color = tuple(
                        min(255, int(self.theme["goal_glow"][i] * pulse + self.theme["goal"][i] * (1 - pulse)))
                        for i in range(3)
                    )
                    pygame.draw.rect(surface, glow_color, rect.inflate(8, 8), border_radius=radius + 4)
                    pygame.draw.rect(surface, self.theme["goal"], rect, border_radius=radius)
                    # Star effect
                    center_x, center_y = x + cell // 2, y + cell // 2
                    star_size = int(4 + 2 * math.sin(ticks / 200))
                    for angle in [0, 72, 144, 216, 288]:
                        rad = math.radians(angle + ticks / 10)
                        px = center_x + int(star_size * math.cos(rad))
                        py = center_y + int(star_size * math.sin(rad))
                        pygame.draw.circle(surface, glow_color, (px, py), 2)
        
        # Enhanced agent with direction indicator and bounce
        ar, ac = self.agent_pos
        bounce = 2 * math.sin(ticks / 200)
        agent_rect = pygame.Rect(ac * cell + 4, ar * cell + 4 + int(bounce), cell - 8, cell - 8)
        # Outer glow
        glow_alpha = int(100 + 50 * math.sin(ticks / 150))
        glow_surf = pygame.Surface((cell + 12, cell + 12), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.theme["agent_outline"], glow_alpha), 
                        pygame.Rect(0, 0, cell + 12, cell + 12), border_radius=radius + 6)
        surface.blit(glow_surf, (ac * cell - 2, ar * cell - 2 + int(bounce)), special_flags=pygame.BLEND_ALPHA_SDL2)
        # Agent body
        pygame.draw.rect(surface, self.theme["agent_outline"], agent_rect.inflate(6, 6), width=2, border_radius=radius + 2)
        pygame.draw.rect(surface, self.theme["agent"], agent_rect, border_radius=radius)
        # Direction indicator (small triangle)
        center_x, center_y = ac * cell + cell // 2, ar * cell + cell // 2 + int(bounce)
        indicator_size = cell // 4
        points = [
            (center_x, center_y - indicator_size),
            (center_x - indicator_size // 2, center_y),
            (center_x + indicator_size // 2, center_y)
        ]
        pygame.draw.polygon(surface, self.theme["agent_outline"], points)

    def _is_dynamic_cell(self, r: int, c: int) -> bool:
        for path, idx in zip(self.dynamic_paths, self.dynamic_indices):
            pr, pc = path[idx]
            if (pr, pc) == (r, c):
                return True
        return False

    def _draw_overlay(self, surface: pygame.Surface):
        if self.font is None:
            return
        info_text = f"Steps {self.steps}/{self.config.max_steps}  |  Distance {self._manhattan(self.agent_pos, self.goal_pos)}"
        text_surf = self.font.render(info_text, True, self.theme["text"])
        surface.blit(text_surf, (12, 10))

