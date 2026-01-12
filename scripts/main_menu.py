import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pygame
import numpy as np


class MainMenu:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        
        # Enable high DPI scaling for crisp rendering
        try:
            import os
            os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
        except:
            pass
        
        self.window_size = (1280, 720)  # 16:9 aspect ratio

        # Use hardware acceleration and vsync for smooth rendering
        self.screen = pygame.display.set_mode(self.window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Maze Maker & RL Algorithm Tester - Enterprise Edition")
        self.clock = pygame.time.Clock()
        
        # Fonts with anti-aliasing enabled - use Font() for better quality
        pygame.font.init()
       
        try:
            # Try to load Segoe UI font directly
            font_paths = [
                "C:/Windows/Fonts/segoeui.ttf",
                "C:/Windows/Fonts/segoeuib.ttf",  # Bold
                pygame.font.match_font("segoeui"),
            ]
            
            title_font_path = None
            for path in font_paths:
                if path and Path(path).exists():
                    title_font_path = path
                    break
            
            if title_font_path:
                self.title_font = pygame.font.Font(title_font_path, 52)
                self.heading_font = pygame.font.Font(title_font_path, 32)
                self.menu_font = pygame.font.Font(title_font_path, 24)
                self.info_font = pygame.font.Font(title_font_path, 18)
            else:
                # Fallback to system fonts with anti-aliasing
                self.title_font = pygame.font.SysFont("Segoe UI", 52, bold=True)
                self.heading_font = pygame.font.SysFont("Segoe UI", 32, bold=True)
                self.menu_font = pygame.font.SysFont("Segoe UI", 24)
                self.info_font = pygame.font.SysFont("Segoe UI", 18)
        except Exception as e:
            # Final fallback
            self.title_font = pygame.font.SysFont("Arial", 52, bold=True)
            self.heading_font = pygame.font.SysFont("Arial", 32, bold=True)
            self.menu_font = pygame.font.SysFont("Arial", 24)
            self.info_font = pygame.font.SysFont("Arial", 18)
        
        # Menu state
        self.mode = "main"  # "main", "select_maze_edit", "select_maze_test", "select_algorithm", "rename_maze"
        self.selected_index = 0
        self.scroll_offset = 0
        self.mazes_dir = Path("mazes")
        self.mazes_dir.mkdir(exist_ok=True)
        self.selected_maze = None
        self.animation_time = 0
        self.renaming_maze = None
        self.rename_text = ""
        self.rename_active = False
        
        # Color scheme
        self.colors = {
            "bg_top": (10, 15, 25),  # Deep navy
            "bg_bottom": (5, 8, 15),  # Almost black
            "accent": (99, 102, 241),  # Indigo (professional, modern)
            "accent_secondary": (139, 92, 246),  # Purple accent
            "accent_hover": (129, 140, 248),  # Lighter indigo
            "text": (255, 255, 255),  # Pure white
            "text_dim": (156, 163, 175),  # Cool gray
            "text_muted": (107, 114, 128),  # Muted gray
            "selected": (99, 102, 241),  # Indigo highlight
            "selected_glow": (139, 92, 246),  # Purple glow
            "border": (30, 41, 59),  # Dark border
            "success": (34, 197, 94),  # Emerald green
            "warning": (251, 191, 36),  # Amber
            "error": (239, 68, 68),  # Red
            "card_bg": (20, 25, 35, 220),  # Semi-transparent card
            "glass": (255, 255, 255, 10),  # Glass morphism effect
        }
        
    def get_maze_files(self) -> List[Path]:
        """Get list of maze JSON files."""
        if not self.mazes_dir.exists():
            return []
        return sorted(self.mazes_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    def load_maze_info(self, maze_path: Path) -> dict:
        """Load basic info about a maze."""
        try:
            with open(maze_path, "r") as f:
                data = json.load(f)
            return {
                "name": maze_path.stem,
                "size": data.get("grid_size", "?"),
                "walls": np.sum(data.get("walls", [])),
                "obstacles": np.sum(data.get("obstacles", [])),
                "path": maze_path,
            }
        except:
            return {"name": maze_path.stem, "size": "?", "walls": 0, "obstacles": 0, "path": maze_path}
    
    def run(self):
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0  # Delta time in seconds
            self.animation_time += dt
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.mode == "main":
                            running = False
                        elif self.mode == "rename_maze":
                            self.mode = "select_maze_edit"
                            self.rename_active = False
                            self.rename_text = ""
                        else:
                            self.mode = "main"
                            self.selected_index = 0
                            self.scroll_offset = 0
                    elif event.key == pygame.K_UP:
                        if self.mode != "rename_maze":
                            self.selected_index = max(0, self.selected_index - 1)
                            self._adjust_scroll()
                    elif event.key == pygame.K_DOWN:
                        if self.mode != "rename_maze":
                            if self.mode == "main":
                                self.selected_index = min(2, self.selected_index + 1)
                            elif self.mode == "select_algorithm":
                                self.selected_index = min(1, self.selected_index + 1)  # Only 2 algorithms
                            else:
                                items = self._get_menu_items()
                                self.selected_index = min(len(items) - 1, self.selected_index + 1)
                                self._adjust_scroll()
                    elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        if self.mode == "rename_maze" and self.rename_active:
                            self._finish_rename()
                        else:
                            self._handle_selection()
                    elif event.key == pygame.K_r and self.mode in ("select_maze_edit", "select_maze_test"):
                        # Rename maze
                        items = self._get_menu_items()
                        if self.selected_index < len(items):
                            maze_info = items[self.selected_index]
                            self.renaming_maze = maze_info["path"]
                            self.rename_text = maze_info["name"]
                            self.rename_active = True
                            self.mode = "rename_maze"
                            self.selected_index = 0
                    elif event.key == pygame.K_BACKSPACE:
                        if self.mode == "rename_maze" and self.rename_active:
                            self.rename_text = self.rename_text[:-1]
                    elif event.key == pygame.K_DELETE and self.mode in ("select_maze_edit", "select_maze_test"):
                        # Delete maze
                        items = self._get_menu_items()
                        if self.selected_index < len(items):
                            maze_info = items[self.selected_index]
                            self._delete_maze(maze_info["path"])
                elif event.type == pygame.TEXTINPUT:
                    if self.mode == "rename_maze" and self.rename_active:
                        if len(self.rename_text) < 50:  # Limit length
                            self.rename_text += event.text
            
            self._render()
    
    def _get_menu_items(self) -> List:
        """Get items for current menu mode."""
        if self.mode == "main":
            return ["Create New Maze", "Edit Existing Maze", "Test Algorithm on Maze"]
        elif self.mode == "select_maze_edit":
            mazes = self.get_maze_files()
            return [self.load_maze_info(m) for m in mazes]
        elif self.mode == "select_maze_test":
            mazes = self.get_maze_files()
            return [self.load_maze_info(m) for m in mazes]
        return []
    
    def _adjust_scroll(self):
        """Adjust scroll offset to keep selected item visible."""
        items = self._get_menu_items()
        if len(items) == 0:
            return
        
        items_per_page = 8
        if self.selected_index < self.scroll_offset:
            self.scroll_offset = self.selected_index
        elif self.selected_index >= self.scroll_offset + items_per_page:
            self.scroll_offset = self.selected_index - items_per_page + 1
    
    def _handle_selection(self):
        """Handle menu selection."""
        if self.mode == "main":
            if self.selected_index == 0:
                # Create new maze
                self._launch_editor()
            elif self.selected_index == 1:
                # Edit existing maze
                mazes = self.get_maze_files()
                if len(mazes) == 0:
                    # No mazes, create new one
                    self._launch_editor()
                else:
                    self.mode = "select_maze_edit"
                    self.selected_index = 0
                    self.scroll_offset = 0
            elif self.selected_index == 2:
                # Test algorithm
                mazes = self.get_maze_files()
                if len(mazes) == 0:
                    # No mazes, show message
                    pass
                else:
                    self.mode = "select_maze_test"
                    self.selected_index = 0
                    self.scroll_offset = 0
        
        elif self.mode == "select_maze_edit":
            items = self._get_menu_items()
            if self.selected_index < len(items):
                maze_info = items[self.selected_index]
                self._launch_editor(maze_info["path"])
        
        elif self.mode == "select_maze_test":
            items = self._get_menu_items()
            if self.selected_index < len(items):
                maze_info = items[self.selected_index]
                self.selected_maze = maze_info["path"]
                self.mode = "select_algorithm"
                self.selected_index = 0
        
        elif self.mode == "select_algorithm":
            algorithms = ["ppo", "dqn"]
            if self.selected_index < len(algorithms):
                algo = algorithms[self.selected_index]
                self._launch_tester_algorithm(algo)
    
    def _delete_maze(self, maze_path: Path):
        """Delete a maze file."""
        try:
            if maze_path.exists():
                maze_path.unlink()
                # Refresh the list
                items = self._get_menu_items()
                if self.selected_index >= len(items):
                    self.selected_index = max(0, len(items) - 1)
        except Exception as e:
            print(f"Error deleting maze: {e}")
    
    def _finish_rename(self):
        """Finish renaming a maze."""
        if not self.renaming_maze or not self.rename_text.strip():
            self.mode = "select_maze_edit"
            self.rename_active = False
            return
        
        try:
            old_path = self.renaming_maze
            new_name = self.rename_text.strip()
            # Remove invalid filename characters
            invalid_chars = '<>:"/\\|?*'
            for char in invalid_chars:
                new_name = new_name.replace(char, '_')
            
            new_path = self.mazes_dir / f"{new_name}.json"
            
            # If name didn't change, just go back
            if old_path == new_path:
                self.mode = "select_maze_edit"
                self.rename_active = False
                return
            
            # If new name already exists, append number
            counter = 1
            original_new_path = new_path
            while new_path.exists():
                new_path = self.mazes_dir / f"{new_name}_{counter}.json"
                counter += 1
            
            # Rename the file
            if old_path.exists():
                old_path.rename(new_path)
            
            self.mode = "select_maze_edit"
            self.rename_active = False
            self.rename_text = ""
            self.renaming_maze = None
        except Exception as e:
            print(f"Error renaming maze: {e}")
            self.mode = "select_maze_edit"
            self.rename_active = False
    
    def _launch_editor(self, maze_path: Optional[Path] = None):
        """Launch the maze editor."""
        import subprocess
        import sys
        # Use the same Python interpreter that's running this script
        python_exe = sys.executable
        cmd = [python_exe, "scripts/maze_editor.py"]
        if maze_path:
            # Pass maze path to editor
            cmd.append("--load")
            cmd.append(str(maze_path))
        else:
            cmd.append("--size")
            cmd.append("10")
        
        pygame.quit()
        try:
            subprocess.run(cmd, cwd=project_root)
        except Exception as e:
            print(f"Error launching editor: {e}")
        sys.exit(0)
    
    def _launch_tester_algorithm(self, algorithm: str):
        """Launch visual trainer with selected algorithm."""
        import subprocess
        import sys
        # Use the same Python interpreter that's running this script (ensures venv is used)
        python_exe = sys.executable
        cmd = [
            python_exe, "scripts/visual_trainer.py",
            "--maze", str(self.selected_maze),
            "--algorithm", algorithm,
            "--total-steps", "5000",
        ]
        
        pygame.quit()
        try:
            subprocess.run(cmd, cwd=project_root)
        except Exception as e:
            print(f"Error launching trainer: {e}")
        sys.exit(0)
    
    def _render(self):
        """Render the current menu."""
        self._draw_background()
        
        if self.mode == "main":
            self._render_main_menu()
        elif self.mode in ("select_maze_edit", "select_maze_test"):
            self._render_maze_list()
        elif self.mode == "select_algorithm":
            self._render_algorithm_selection()
        elif self.mode == "rename_maze":
            self._render_rename_dialog()
        
        pygame.display.flip()
    
    def _draw_background(self):
        """Draw sophisticated gradient background with subtle patterns."""
        width, height = self.window_size
        
        # Main gradient
        for y in range(height):
            ratio = y / max(1, height - 1)
            r = int(self.colors["bg_top"][0] + (self.colors["bg_bottom"][0] - self.colors["bg_top"][0]) * ratio)
            g = int(self.colors["bg_top"][1] + (self.colors["bg_bottom"][1] - self.colors["bg_top"][1]) * ratio)
            b = int(self.colors["bg_top"][2] + (self.colors["bg_bottom"][2] - self.colors["bg_top"][2]) * ratio)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (width, y))
        
        # Subtle radial gradient overlay for depth
        center_x, center_y = width // 2, height // 2
        max_dist = math.sqrt(center_x**2 + center_y**2)
        
        # Draw subtle grid pattern
        grid_surf = pygame.Surface((width, height), pygame.SRCALPHA)
        grid_color = (255, 255, 255, 3)  # Very subtle white grid
        grid_spacing = 40
        for x in range(0, width, grid_spacing):
            pygame.draw.line(grid_surf, grid_color, (x, 0), (x, height), 1)
        for y in range(0, height, grid_spacing):
            pygame.draw.line(grid_surf, grid_color, (0, y), (width, y), 1)
        self.screen.blit(grid_surf, (0, 0))
    
    def _render_main_menu(self):
        """Render main menu."""
        width, height = self.window_size
        
        # Title with sophisticated styling
        title_text = "Maze Maker & RL Tester"
        
        # Multiple shadow layers for depth
        for offset in [(3, 3), (2, 2), (1, 1)]:
            shadow = self.title_font.render(title_text, True, (0, 0, 0))
            shadow_rect = shadow.get_rect(center=(width // 2 + offset[0], 100 + offset[1]))
            self.screen.blit(shadow, shadow_rect)
        
        # Main title with gradient effect (simulated with glow)
        title = self.title_font.render(title_text, True, self.colors["accent"])
        title_rect = title.get_rect(center=(width // 2, 100))
        self.screen.blit(title, title_rect)
        
        # Subtle glow effect
        glow = self.title_font.render(title_text, True, self.colors["selected_glow"])
        glow_surf = pygame.Surface(glow.get_size(), pygame.SRCALPHA)
        glow_surf.blit(glow, (0, 0))
        glow_surf.set_alpha(30)
        self.screen.blit(glow_surf, (title_rect.x - 2, title_rect.y - 2))
        
        # Subtitle
        subtitle = self.info_font.render("Enterprise-Grade Reinforcement Learning Platform", True, self.colors["text_dim"])
        subtitle_rect = subtitle.get_rect(center=(width // 2, 150))
        self.screen.blit(subtitle, subtitle_rect)
        
        # Tagline
        tagline = self.info_font.render("Design custom mazes • Test AI algorithms • Analyze performance", True, self.colors["text_muted"])
        tagline_rect = tagline.get_rect(center=(width // 2, 175))
        self.screen.blit(tagline, tagline_rect)
        
        # Menu options with enterprise styling
        options = [
            {"text": "Create New Maze", "icon": "➕", "desc": "Design custom maze layouts"},
            {"text": "Edit Existing Maze", "icon": "✏️", "desc": "Modify saved maze configurations"},
            {"text": "Test Algorithm on Maze", "icon": "🧪", "desc": "Run RL algorithms and analyze performance"}
        ]
        option_y = 240
        option_spacing = 95
        
        for i, opt in enumerate(options):
            y = option_y + i * option_spacing
            is_selected = (i == self.selected_index)
            
            # Card-style background with glass morphism
            card_width = 500
            card_height = 70
            card_x = (width - card_width) // 2
            card_y = y - 5
            
            if is_selected:
                # Animated glow effect
                pulse = 0.95 + 0.05 * (1 + math.sin(self.animation_time * 2.5))
                glow_alpha = int(100 * pulse)
                
                # Outer glow
                glow_rect = pygame.Rect(card_x - 5, card_y - 5, card_width + 10, card_height + 10)
                glow_surf = pygame.Surface((card_width + 10, card_height + 10), pygame.SRCALPHA)
                pygame.draw.rect(glow_surf, (*self.colors["selected_glow"], glow_alpha), 
                               glow_surf.get_rect(), border_radius=15)
                self.screen.blit(glow_surf, (card_x - 5, card_y - 5))
                
                # Shadow
                shadow_rect = pygame.Rect(card_x + 3, card_y + 3, card_width, card_height)
                shadow_surf = pygame.Surface((card_width, card_height), pygame.SRCALPHA)
                pygame.draw.rect(shadow_surf, (0, 0, 0, 120), shadow_surf.get_rect(), border_radius=12)
                self.screen.blit(shadow_surf, shadow_rect)
                
                # Main card with gradient border
                card_rect = pygame.Rect(card_x, card_y, card_width, card_height)
                pygame.draw.rect(self.screen, (20, 25, 35), card_rect, border_radius=12)
                pygame.draw.rect(self.screen, self.colors["selected"], card_rect, width=2, border_radius=12)
            else:
                # Subtle card for unselected items
                card_rect = pygame.Rect(card_x, card_y, card_width, card_height)
                card_surf = pygame.Surface((card_width, card_height), pygame.SRCALPHA)
                pygame.draw.rect(card_surf, (20, 25, 35, 150), card_surf.get_rect(), border_radius=12)
                pygame.draw.rect(card_surf, (40, 50, 70, 100), card_surf.get_rect(), width=1, border_radius=12)
                self.screen.blit(card_surf, card_rect)
            
            # Icon and text
            icon_text = self.menu_font.render(opt["icon"], True, self.colors["accent"] if is_selected else self.colors["text_dim"])
            icon_rect = icon_text.get_rect(midleft=(card_x + 20, y + 20))
            self.screen.blit(icon_text, icon_rect)
            
            # Main text
            text_color = self.colors["accent"] if is_selected else self.colors["text"]
            text = self.menu_font.render(opt["text"], True, text_color)
            text_rect = text.get_rect(midleft=(card_x + 60, y + 20))
            self.screen.blit(text, text_rect)
            
            # Description
            desc = self.info_font.render(opt["desc"], True, self.colors["text_dim"])
            desc_rect = desc.get_rect(midleft=(card_x + 60, y + 45))
            self.screen.blit(desc, desc_rect)
        
        # Instructions
        instructions = [
            "Use ↑↓ to navigate | Enter/Space to select | Esc to quit",
        ]
        for i, instr in enumerate(instructions):
            text = self.info_font.render(instr, True, self.colors["text_dim"])
            text_rect = text.get_rect(center=(width // 2, height - 40 - i * 25))
            self.screen.blit(text, text_rect)
    
    def _render_maze_list(self):
        """Render maze selection list."""
        width, height = self.window_size
        
        # Title with shadow
        title_text = "Select Maze to Edit" if self.mode == "select_maze_edit" else "Select Maze to Test"
        title_shadow = self.heading_font.render(title_text, True, (0, 0, 0))
        title = self.heading_font.render(title_text, True, self.colors["accent"])
        title_rect = title.get_rect(center=(width // 2 + 1, 61))
        shadow_rect = title_shadow.get_rect(center=(width // 2, 60))
        self.screen.blit(title_shadow, shadow_rect)
        self.screen.blit(title, title_rect)
        
        # Get mazes
        items = self._get_menu_items()
        
        if len(items) == 0:
            no_mazes = self.menu_font.render("No mazes found. Create one first!", True, self.colors["text_dim"])
            no_mazes_rect = no_mazes.get_rect(center=(width // 2, height // 2))
            self.screen.blit(no_mazes, no_mazes_rect)
        else:
            # List of mazes
            list_y = 120
            list_height = height - 200
            items_per_page = 8
            item_height = list_height // items_per_page
            
            visible_items = items[self.scroll_offset:self.scroll_offset + items_per_page]
            
            for i, maze_info in enumerate(visible_items):
                idx = self.scroll_offset + i
                y = list_y + i * item_height
                is_selected = (idx == self.selected_index)
                
                # Background with smooth animation and shadow
                if is_selected:
                    # Draw shadow
                    shadow_rect = pygame.Rect(52, y - 3, width - 96, item_height - 6)
                    shadow_surf = pygame.Surface((width - 96, item_height - 6), pygame.SRCALPHA)
                    pygame.draw.rect(shadow_surf, (0, 0, 0, 60), shadow_surf.get_rect(), border_radius=10)
                    self.screen.blit(shadow_surf, shadow_rect)
                    # Draw main background
                    rect = pygame.Rect(50, y - 5, width - 100, item_height - 10)
                    pygame.draw.rect(self.screen, (30, 41, 59), rect, border_radius=8)
                    pygame.draw.rect(self.screen, self.colors["selected"], rect, width=3, border_radius=8)
                
                # Maze info with better formatting
                name_color = self.colors["accent"] if is_selected else self.colors["text"]
                name_text = self.menu_font.render(maze_info["name"], True, name_color)
                self.screen.blit(name_text, (70, y + 8))
                
                info_text = f"Size: {maze_info['size']}x{maze_info['size']}  •  Walls: {maze_info['walls']}  •  Obstacles: {maze_info['obstacles']}"
                info = self.info_font.render(info_text, True, self.colors["text_dim"])
                self.screen.blit(info, (70, y + 35))
                
                # Show rename hint for selected item
                if is_selected and self.mode == "select_maze_edit":
                    rename_hint = self.info_font.render("Press R to rename", True, self.colors["accent"])
                    self.screen.blit(rename_hint, (width - 200, y + 8))
        
        # Instructions with rename option
        if self.mode == "select_maze_edit":
            instructions = [
                "Use ↑↓ to navigate | Enter/Space to edit | R to rename | Del to delete | Esc to go back",
            ]
        else:
            instructions = [
                "Use ↑↓ to navigate | Enter/Space to select | Esc to go back",
            ]
        for i, instr in enumerate(instructions):
            text = self.info_font.render(instr, True, self.colors["text_dim"])
            text_rect = text.get_rect(center=(width // 2, height - 40 - i * 25))
            self.screen.blit(text, text_rect)
    
    def _render_algorithm_selection(self):
        """Render algorithm selection menu."""
        width, height = self.window_size
        
        # Title with shadow
        maze_name = self.selected_maze.stem if hasattr(self, 'selected_maze') and self.selected_maze else "Unknown"
        title_text = f"Test Algorithm on: {maze_name}"
        title_shadow = self.heading_font.render(title_text, True, (0, 0, 0))
        title = self.heading_font.render(title_text, True, self.colors["accent"])
        title_rect = title.get_rect(center=(width // 2 + 1, 61))
        shadow_rect = title_shadow.get_rect(center=(width // 2, 60))
        self.screen.blit(title_shadow, shadow_rect)
        self.screen.blit(title, title_rect)
        
        # Algorithm options
        algorithms = [
            {"name": "PPO (Proximal Policy Optimization)", "key": "ppo", "desc": "On-policy, stable learning"},
            {"name": "DQN (Deep Q-Network)", "key": "dqn", "desc": "Off-policy, experience replay"},
        ]
        
        option_y = 200
        option_spacing = 100
        
        for i, algo in enumerate(algorithms):
            y = option_y + i * option_spacing
            is_selected = (i == self.selected_index)
            
            # Background with smooth animation and shadow
            if is_selected:
                # Draw shadow
                shadow_rect = pygame.Rect(width // 2 - 298, y - 13, 596, 82)
                shadow_surf = pygame.Surface((596, 82), pygame.SRCALPHA)
                pygame.draw.rect(shadow_surf, (0, 0, 0, 80), shadow_surf.get_rect(), border_radius=12)
                self.screen.blit(shadow_surf, shadow_rect)
                # Draw main background
                rect = pygame.Rect(width // 2 - 300, y - 15, 600, 80)
                pygame.draw.rect(self.screen, (30, 41, 59), rect, border_radius=10)
                pygame.draw.rect(self.screen, self.colors["selected"], rect, width=3, border_radius=10)
            
            # Name
            name_color = self.colors["accent"] if is_selected else self.colors["text"]
            name_text = self.menu_font.render(algo["name"], True, name_color)
            name_rect = name_text.get_rect(center=(width // 2, y + 10))
            self.screen.blit(name_text, name_rect)
            
            # Description
            desc_text = self.info_font.render(algo["desc"], True, self.colors["text_dim"])
            desc_rect = desc_text.get_rect(center=(width // 2, y + 40))
            self.screen.blit(desc_text, desc_rect)
        
        # Instructions
        instructions = [
            "Use ↑↓ to navigate | Enter/Space to start training | Esc to go back",
        ]
        for i, instr in enumerate(instructions):
            text = self.info_font.render(instr, True, self.colors["text_dim"])
            text_rect = text.get_rect(center=(width // 2, height - 40 - i * 25))
            self.screen.blit(text, text_rect)
    
    def _render_rename_dialog(self):
        """Render rename maze dialog."""
        width, height = self.window_size
        
        # Semi-transparent overlay
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        # Dialog box
        dialog_width = 600
        dialog_height = 200
        dialog_x = (width - dialog_width) // 2
        dialog_y = (height - dialog_height) // 2
        
        # Shadow
        shadow_rect = pygame.Rect(dialog_x + 4, dialog_y + 4, dialog_width, dialog_height)
        shadow_surf = pygame.Surface((dialog_width, dialog_height), pygame.SRCALPHA)
        pygame.draw.rect(shadow_surf, (0, 0, 0, 150), shadow_surf.get_rect(), border_radius=15)
        self.screen.blit(shadow_surf, shadow_rect)
        
        # Dialog background
        dialog_rect = pygame.Rect(dialog_x, dialog_y, dialog_width, dialog_height)
        pygame.draw.rect(self.screen, (30, 41, 59), dialog_rect, border_radius=15)
        pygame.draw.rect(self.screen, self.colors["accent"], dialog_rect, width=2, border_radius=15)
        
        # Title
        title = self.heading_font.render("Rename Maze", True, self.colors["accent"])
        title_rect = title.get_rect(center=(width // 2, dialog_y + 40))
        self.screen.blit(title, title_rect)
        
        # Input box
        input_y = dialog_y + 90
        input_rect = pygame.Rect(dialog_x + 40, input_y, dialog_width - 80, 40)
        pygame.draw.rect(self.screen, (15, 23, 42), input_rect, border_radius=8)
        pygame.draw.rect(self.screen, self.colors["accent"], input_rect, width=2, border_radius=8)
        
        # Text input with cursor
        display_text = self.rename_text
        if self.rename_active and int(self.animation_time * 2) % 2:
            display_text += "|"  # Blinking cursor
        
        text_surf = self.menu_font.render(display_text, True, self.colors["text"])
        text_rect = text_surf.get_rect(midleft=(input_rect.x + 10, input_rect.centery))
        # Clip text if too long
        if text_rect.width > input_rect.width - 20:
            # Show end of text
            text_surf = self.menu_font.render("..." + display_text[-(len(display_text) - 10):], True, self.colors["text"])
            text_rect = text_surf.get_rect(midleft=(input_rect.x + 10, input_rect.centery))
        self.screen.blit(text_surf, text_rect)
        
        # Instructions
        instructions = [
            "Type new name | Enter to confirm | Esc to cancel",
        ]
        for i, instr in enumerate(instructions):
            text = self.info_font.render(instr, True, self.colors["text_dim"])
            text_rect = text.get_rect(center=(width // 2, dialog_y + dialog_height - 30 - i * 20))
            self.screen.blit(text, text_rect)


def main():
    menu = MainMenu()
    menu.run()


if __name__ == "__main__":
    main()
