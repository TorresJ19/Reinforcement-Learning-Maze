import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pygame


class MazeEditor:
    def __init__(self, grid_size: int = 10, maze_path: Optional[Path] = None):
        self.grid_size = grid_size
        self.rows = grid_size
        self.cols = grid_size
        
        # Maze state
        self.walls = np.zeros((self.rows, self.cols), dtype=np.bool_)
        self.obstacles = np.zeros((self.rows, self.cols), dtype=np.bool_)
        self.start_pos: Optional[Tuple[int, int]] = (0, 0)
        self.goal_pos: Optional[Tuple[int, int]] = (self.rows - 1, self.cols - 1)
        
        # Editor state
        self.mode = "wall"  # "wall", "obstacle", "start", "goal", "erase"
        self.dragging = False
        
        # Pygame setup
        pygame.init()
        self.window_size = (1024, 768)
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Maze Editor")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Segoe UI", 18)
        self.title_font = pygame.font.SysFont("Segoe UI", 24, bold=True)
        
        # Load maze if path provided (after pygame is initialized)
        if maze_path and maze_path.exists():
            self.load_maze(maze_path)
        
        # Calculate cell size and viewport
        self._compute_layout()
        
        # Color scheme
        self.theme = {
            "bg_top": (10, 15, 25),
            "bg_bottom": (5, 8, 15),
            "grid": (30, 41, 59),
            "wall": (71, 85, 105),
            "obstacle": (239, 68, 68),
            "start": (34, 197, 94),
            "goal": (251, 191, 36),
            "text": (255, 255, 255),
            "text_dim": (156, 163, 175),
            "selected": (99, 102, 241),
            "accent": (139, 92, 246),
        }
    
    def _compute_layout(self):
        win_w, win_h = self.window_size
        maze_h = win_h - 150  # Reserve space for UI
        maze_w = win_w - 40
        
        cell_from_w = max(16, maze_w // max(1, self.cols))
        cell_from_h = max(16, maze_h // max(1, self.rows))
        self.cell_px = max(24, min(64, cell_from_w, cell_from_h))
        
        maze_view_w = self.cols * self.cell_px
        maze_view_h = self.rows * self.cell_px
        maze_x = (win_w - maze_view_w) // 2
        maze_y = 120 + (maze_h - maze_view_h) // 2
        self.maze_viewport = (maze_x, maze_y, maze_view_w, maze_view_h)
    
    def load_maze(self, maze_path: Path):
        """Load a maze from a JSON file."""
        try:
            with open(maze_path, "r") as f:
                data = json.load(f)
            
            # Update grid size
            grid_size = data.get("grid_size", self.grid_size)
            self.grid_size = grid_size
            self.rows = grid_size
            self.cols = grid_size
            
            # Load walls and obstacles
            walls_data = data.get("walls", [])
            obstacles_data = data.get("obstacles", [])
            
            # Convert lists to numpy arrays
            if isinstance(walls_data, list):
                if len(walls_data) > 0 and isinstance(walls_data[0], list):
                    self.walls = np.array(walls_data, dtype=np.bool_)
                else:
                    # Flattened list, reshape it
                    self.walls = np.array(walls_data, dtype=np.bool_).reshape((self.rows, self.cols))
            else:
                self.walls = np.zeros((self.rows, self.cols), dtype=np.bool_)
            
            if isinstance(obstacles_data, list):
                if len(obstacles_data) > 0 and isinstance(obstacles_data[0], list):
                    self.obstacles = np.array(obstacles_data, dtype=np.bool_)
                else:
                    # Flattened list, reshape it
                    self.obstacles = np.array(obstacles_data, dtype=np.bool_).reshape((self.rows, self.cols))
            else:
                self.obstacles = np.zeros((self.rows, self.cols), dtype=np.bool_)
            
            # Load start and goal positions
            start = data.get("start", [0, 0])
            goal = data.get("goal", [self.rows - 1, self.cols - 1])
            self.start_pos = tuple(start) if isinstance(start, list) else start
            self.goal_pos = tuple(goal) if isinstance(goal, list) else goal
            
            # Recompute layout for new size (after pygame is initialized)
            if hasattr(self, 'window_size'):
                self._compute_layout()
            
            print(f"Loaded maze from {maze_path}")
        except Exception as e:
            print(f"Error loading maze: {e}")
            # Keep default empty maze
    
    def _get_cell_from_pos(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Convert screen position to grid cell coordinates."""
        mx, my, mw, mh = self.maze_viewport
        x, y = pos
        if not (mx <= x < mx + mw and my <= y < my + mh):
            return None
        c = (x - mx) // self.cell_px
        r = (y - my) // self.cell_px
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return (r, c)
        return None
    
    def _handle_click(self, pos: Tuple[int, int], button: int):
        """Handle mouse click on the maze."""
        cell = self._get_cell_from_pos(pos)
        if cell is None:
            return
        
        r, c = cell
        
        if self.mode == "wall":
            if button == 1:  # Left click - place wall
                if cell != self.start_pos and cell != self.goal_pos:
                    self.walls[r, c] = True
                    self.obstacles[r, c] = False
            elif button == 3:  # Right click - remove wall
                self.walls[r, c] = False
        elif self.mode == "obstacle":
            if button == 1:  # Left click - place obstacle
                if cell != self.start_pos and cell != self.goal_pos:
                    self.obstacles[r, c] = True
                    self.walls[r, c] = False
            elif button == 3:  # Right click - remove obstacle
                self.obstacles[r, c] = False
        elif self.mode == "start":
            if button == 1:  # Left click - set start
                if not self.walls[r, c] and not self.obstacles[r, c]:
                    self.start_pos = (r, c)
        elif self.mode == "goal":
            if button == 1:  # Left click - set goal
                if not self.walls[r, c] and not self.obstacles[r, c]:
                    self.goal_pos = (r, c)
        elif self.mode == "erase":
            if button == 1:  # Left click - erase
                self.walls[r, c] = False
                self.obstacles[r, c] = False
    
    def _handle_drag(self, pos: Tuple[int, int], button: int):
        """Handle mouse drag for continuous drawing."""
        if not self.dragging:
            return
        self._handle_click(pos, button)
    
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_w:
                        self.mode = "wall"
                    elif event.key == pygame.K_o:
                        self.mode = "obstacle"
                    elif event.key == pygame.K_s:
                        self.mode = "start"
                    elif event.key == pygame.K_g:
                        self.mode = "goal"
                    elif event.key == pygame.K_e:
                        self.mode = "erase"
                    elif event.key == pygame.K_c:
                        # Clear all
                        self.walls[:] = False
                        self.obstacles[:] = False
                    elif event.key == pygame.K_1:
                        # Decrease grid size
                        if self.grid_size > 5:
                            self.grid_size -= 1
                            self.rows = self.cols = self.grid_size
                            self.walls = np.zeros((self.rows, self.cols), dtype=np.bool_)
                            self.obstacles = np.zeros((self.rows, self.cols), dtype=np.bool_)
                            self.start_pos = (0, 0)
                            self.goal_pos = (self.rows - 1, self.cols - 1)
                            self._compute_layout()
                    elif event.key == pygame.K_2:
                        # Increase grid size
                        if self.grid_size < 30:
                            self.grid_size += 1
                            self.rows = self.cols = self.grid_size
                            old_walls = self.walls.copy()
                            old_obstacles = self.obstacles.copy()
                            # Safely extract shape dimensions
                            old_shape = old_walls.shape
                            if len(old_shape) >= 2:
                                old_rows, old_cols = int(old_shape[0]), int(old_shape[1])
                            else:
                                old_rows, old_cols = 0, 0
                            self.walls = np.zeros((self.rows, self.cols), dtype=np.bool_)
                            self.obstacles = np.zeros((self.rows, self.cols), dtype=np.bool_)
                            # Copy old layout if possible
                            if old_rows > 0 and old_cols > 0:
                                min_r = min(self.rows, old_rows)
                                min_c = min(self.cols, old_cols)
                                self.walls[:min_r, :min_c] = old_walls[:min_r, :min_c]
                                self.obstacles[:min_r, :min_c] = old_obstacles[:min_r, :min_c]
                            # Ensure start and goal are within bounds
                            if self.start_pos is not None and len(self.start_pos) >= 2 and (self.start_pos[0] >= self.rows or self.start_pos[1] >= self.cols):
                                self.start_pos = (0, 0)
                            if self.goal_pos is not None and len(self.goal_pos) >= 2 and (self.goal_pos[0] >= self.rows or self.goal_pos[1] >= self.cols):
                                self.goal_pos = (self.rows - 1, self.cols - 1)
                            self._compute_layout()
                    elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                        # Save maze
                        self._save_maze()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.dragging = True
                    self._handle_click(event.pos, event.button)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.dragging = False
                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging:
                        self._handle_drag(event.pos, 1)  # Assume left button when dragging
            
            self._render()
            self.clock.tick(60)
        
        pygame.quit()
    
    def _save_maze(self):
        """Save the current maze to a JSON file."""
        maze_data = {
            "grid_size": self.grid_size,
            "walls": self.walls.tolist(),
            "obstacles": self.obstacles.tolist(),
            "start": self.start_pos,
            "goal": self.goal_pos,
        }
        
        # Create mazes directory if it doesn't exist
        maze_dir = Path("mazes")
        maze_dir.mkdir(exist_ok=True)
        
        # Save with timestamp or ask for filename
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = maze_dir / f"maze_{timestamp}.json"
        
        with open(filename, "w") as f:
            json.dump(maze_data, f, indent=2)
        
        print(f"Maze saved to {filename}")
        return filename
    
    def _render(self):
        """Render the editor interface."""
        self.screen.fill((12, 18, 28))
        
        # Draw gradient background
        self._draw_gradient(self.screen, self.theme["bg_top"], self.theme["bg_bottom"])
        
        # Draw maze
        mx, my, mw, mh = self.maze_viewport
        maze_surface = pygame.Surface((mw, mh))
        self._draw_gradient(maze_surface, self.theme["bg_top"], self.theme["bg_bottom"])
        self._draw_grid(maze_surface)
        self._draw_cells(maze_surface)
        self.screen.blit(maze_surface, (mx, my))
        
        # Draw UI overlay
        self._draw_ui()
        
        pygame.display.flip()
    
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
        radius = 6
        
        for r in range(self.rows):
            for c in range(self.cols):
                x, y = c * cell, r * cell
                rect = pygame.Rect(x + 2, y + 2, cell - 4, cell - 4)
                
                if self.walls[r, c]:
                    pygame.draw.rect(surface, self.theme["wall"], rect, border_radius=radius)
                elif self.obstacles[r, c]:
                    pygame.draw.rect(surface, self.theme["obstacle"], rect, border_radius=radius)
                
                if (r, c) == self.start_pos:
                    pygame.draw.rect(surface, self.theme["start"], rect, border_radius=radius)
                    # Draw "S" label
                    text = self.font.render("S", True, (0, 0, 0))
                    text_rect = text.get_rect(center=(x + cell // 2, y + cell // 2))
                    surface.blit(text, text_rect)
                elif (r, c) == self.goal_pos:
                    pygame.draw.rect(surface, self.theme["goal"], rect, border_radius=radius)
                    # Draw "G" label
                    text = self.font.render("G", True, (0, 0, 0))
                    text_rect = text.get_rect(center=(x + cell // 2, y + cell // 2))
                    surface.blit(text, text_rect)
    
    def _draw_ui(self):
        """Draw enterprise-grade UI controls and instructions."""
        width, height = self.screen.get_size()
        
        # Top bar with gradient
        top_bar = pygame.Surface((width, 100), pygame.SRCALPHA)
        for y in range(100):
            alpha = int(220 - (y / 100) * 40)
            top_bar.fill((10, 15, 25, alpha), (0, y, width, 1))
        self.screen.blit(top_bar, (0, 0))
        pygame.draw.rect(self.screen, self.theme["accent"], (0, 0, width, 3))
        
        # Title with badge
        title_bg = pygame.Surface((250, 40), pygame.SRCALPHA)
        pygame.draw.rect(title_bg, (99, 102, 241, 100), title_bg.get_rect(), border_radius=8)
        self.screen.blit(title_bg, (20, 15))
        title = self.title_font.render("Maze Editor", True, self.theme["text"])
        self.screen.blit(title, (30, 20))
        
        # Mode indicator with card design
        mode_colors = {
            "wall": self.theme["wall"],
            "obstacle": self.theme["obstacle"],
            "start": self.theme["start"],
            "goal": self.theme["goal"],
            "erase": (156, 163, 175),
        }
        mode_bg = pygame.Surface((200, 30), pygame.SRCALPHA)
        pygame.draw.rect(mode_bg, (20, 25, 35, 200), mode_bg.get_rect(), border_radius=6)
        pygame.draw.rect(mode_bg, (mode_colors.get(self.mode, self.theme["text"])[0], 
                                   mode_colors.get(self.mode, self.theme["text"])[1],
                                   mode_colors.get(self.mode, self.theme["text"])[2], 150), 
                        mode_bg.get_rect(), width=2, border_radius=6)
        self.screen.blit(mode_bg, (20, 60))
        mode_text = f"Mode: {self.mode.upper()}"
        mode_surf = self.font.render(mode_text, True, mode_colors.get(self.mode, self.theme["text"]))
        self.screen.blit(mode_surf, (30, 65))
        
        # Instructions
        instructions = [
            ("W", "Wall"), ("O", "Obstacle"), ("S", "Start"), ("G", "Goal"), ("E", "Erase"),
        ]
        inst_x = 280
        for i, (key, label) in enumerate(instructions):
            card = pygame.Surface((80, 25), pygame.SRCALPHA)
            pygame.draw.rect(card, (20, 25, 35, 150), card.get_rect(), border_radius=4)
            self.screen.blit(card, (inst_x + i * 90, 60))
            key_text = self.font.render(key, True, self.theme["accent"])
            label_text = self.font.render(label, True, self.theme["text_dim"])
            self.screen.blit(key_text, (inst_x + i * 90 + 5, 63))
            self.screen.blit(label_text, (inst_x + i * 90 + 25, 63))
        
        # Stats
        wall_count = np.sum(self.walls)
        obstacle_count = np.sum(self.obstacles)
        stats_bg = pygame.Surface((250, 50), pygame.SRCALPHA)
        pygame.draw.rect(stats_bg, (20, 25, 35, 200), stats_bg.get_rect(), border_radius=8)
        pygame.draw.rect(stats_bg, (99, 102, 241, 100), stats_bg.get_rect(), width=2, border_radius=8)
        self.screen.blit(stats_bg, (width - 270, 15))
        
        stats_label = self.font.render("Statistics", True, self.theme["text_dim"])
        self.screen.blit(stats_label, (width - 260, 20))
        stats_text = f"Walls: {wall_count}  •  Obstacles: {obstacle_count}  •  Size: {self.grid_size}x{self.grid_size}"
        stats_surf = self.font.render(stats_text, True, self.theme["text"])
        self.screen.blit(stats_surf, (width - 260, 40))
        
        # Bottom controls bar
        controls_y = height - 40
        controls_bg = pygame.Surface((width, 40), pygame.SRCALPHA)
        controls_bg.fill((10, 15, 25, 200))
        self.screen.blit(controls_bg, (0, controls_y))
        pygame.draw.rect(self.screen, self.theme["accent"], (0, controls_y, width, 2))
        
        controls = "Left Click: Place  |  Right Click: Remove  |  Drag: Draw  |  C: Clear  |  Enter: Save  |  Esc: Quit"
        control_text = self.font.render(controls, True, self.theme["text_dim"])
        self.screen.blit(control_text, (20, controls_y + 10))


def main():
    parser = argparse.ArgumentParser(description="Interactive maze editor")
    parser.add_argument("--size", type=int, default=10, help="Initial grid size")
    parser.add_argument("--load", type=Path, help="Path to maze JSON file to load")
    args = parser.parse_args()
    
    if args.load:
        # Load maze from file
        editor = MazeEditor(grid_size=10, maze_path=args.load)
    else:
        editor = MazeEditor(grid_size=args.size)
    editor.run()


if __name__ == "__main__":
    main()

