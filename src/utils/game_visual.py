import pygame
import numpy as np
import sys
import os
import time
import math
import random
from typing import Dict, List, Tuple, Optional, Any, Set
import torch
from agents.dqn import DQNAgent

# Set device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from pathlib import Path
from pygame.locals import *

class GridWorldVisualizer:
    """Enhanced PyGame-based visualizer for the GridWorld environment and agents.
    
    This visualizer provides high-quality rendering with textures, animations, 
    particle effects, and a modern UI. Features include:
    
    1. Texture-based terrain rendering
    2. Smooth agent animations with directional sprites
    3. Particle effects for wind and goals
    4. Gradient-based value function visualization
    5. Enhanced policy arrows with confidence indicators
    6. Modern UI panel with custom fonts
    7. Performance optimizations (dirty rectangles, frame rate control)
    
    Usage:
        ```python
        # Create visualizer with default window size
        vis = GridWorldVisualizer()
        
        # Run simulation
        vis.simulate(env, agent, num_episodes=5)
        
        # For single-frame rendering
        vis.render(env, agent, episode=1, step=10, reward=5.0)
        ```
    
    Performance Tips:
    - Textures are cached for better performance
    - Animation speed is frame-rate independent
    - Particle effects scale based on available performance
    - Use render() with dirty rectangles for most efficient updating
    """
    
    # Color constants with harmonized palette for cohesive visual design
    COLORS = {
        # Environment colors
        'background': (20, 20, 35),     # Dark blue-gray background
        'grid': (90, 95, 115),          # Medium gray grid lines
        
        # Actor colors
        'agent': (255, 140, 0),         # Bright orange for agent
        'goal': (50, 230, 100),         # Vivid green for goal
        
        # UI colors
        'text': (240, 240, 255),        # Almost white text
        'text_dark': (15, 15, 25),      # Dark text for light backgrounds
        'ui_button': (60, 60, 80),      # Button background
        'ui_button_hover': (80, 80, 100), # Button hover state
        'ui_button_text': (220, 220, 255), # Button text
        'ui_button_border': (100, 100, 130), # Button border
        'ui_info_text': (180, 180, 220),  # Info panel text
        'ui_panel_bg': (40, 40, 55),    # Info panel background
        'ui_panel_border': (80, 80, 100), # Info panel border
        'ui_header': (200, 200, 240),   # Header text color
        
        # Terrain colors - harmonized palette
        'normal': (144, 238, 144),      # #90EE90 - Light green for normal terrain
        'grass': (144, 238, 144),       # #90EE90 - Light green (same as normal)
        'ice': (180, 220, 255),         # #B4DCFF - Bright blue for ice
        'mud': (139, 69, 19),           # #8B4513 - Dark brown for mud
        'quicksand': (210, 180, 140),   # #D2B48C - Light brown for quicksand
        'water': (70, 130, 230),        # #4682E6 - Vibrant blue for water
        'sand': (210, 180, 140),        # Same as quicksand for consistency
        
        # Effect colors
        'wind': (150, 200, 255),        # Brighter blue for wind visibility
        'value_high': (255, 60, 60, 200), # Red for high values
        'value_low': (255, 240, 60, 200),  # Yellow for low values
        
        # Action colors
        'action_up': (100, 200, 255),    # Color for up action
        'action_right': (100, 255, 100), # Color for right action
        'action_down': (255, 200, 100),  # Color for down action
        'action_left': (255, 100, 100),   # Color for left action
        
        # Effect colors
        'value_high': (255, 60, 60, 200),  # More opaque red
        'value_low': (255, 240, 60, 200),   # More opaque yellow
        'ui_button': (60, 60, 80),      # Button background
        'ui_button_hover': (80, 80, 100),  # Button background when hovered
        'ui_button_text': (220, 220, 255),  # Button text color
        'ui_panel_bg': (40, 40, 50),    # Info panel background
        'ui_panel_border': (80, 80, 100),  # Info panel border
        'ui_header': (180, 180, 220)   # Header text color
    }
    
    # Button definitions - complete set of functional buttons
    BUTTONS = {
        'toggle_values': {
            'label': 'Toggle Values', 
            'action': 'toggle_values',
            'position': 'panel',
            'size': (180, 36),
            'tooltip': 'Show/hide state values',
            'active': True
        },
        'toggle_policy': {
            'label': 'Toggle Policy', 
            'action': 'toggle_policy',
            'position': 'panel',
            'size': (180, 36),
            'tooltip': 'Show/hide policy arrows',
            'active': True
        },
        'test_mode': {
            'label': 'Test Mode', 
            'action': 'test_mode',
            'position': 'panel',
            'size': (180, 36),
            'tooltip': 'Toggle between training/testing mode',
            'active': True
        },
        'fullscreen': {
            'label': 'Fullscreen',
            'action': 'fullscreen',
            'position': 'panel',
            'size': (180, 36),
            'tooltip': 'Toggle fullscreen mode',
            'active': True
        },
        'restart': {
            'label': 'Restart', 
            'action': 'restart',
            'position': 'panel',
            'size': (180, 36),
            'tooltip': 'Restart current episode',
            'active': True
        },
        'quit': {
            'label': 'Quit', 
            'action': 'quit',
            'position': 'panel',
            'size': (180, 36),
            'tooltip': 'Exit visualization',
            'active': True
        }
    }
    
    # Wind direction arrows
    WIND_ARROWS = {
        (0, 1): '→',   # Right
        (0, -1): '←',  # Left
        (-1, 0): '↑',  # Up
        (1, 0): '↓',   # Down
        (-1, 1): '↗',  # Up-Right
        (-1, -1): '↖', # Up-Left
        (1, 1): '↘',   # Down-Right
        (1, -1): '↙'   # Down-Left
    }
    
    # Wind particle directions (for animation)
    WIND_VECTORS = {
        (0, 1): (1, 0),     # Right
        (0, -1): (-1, 0),   # Left
        (-1, 0): (0, -1),   # Up
        (1, 0): (0, 1),     # Down
        (-1, 1): (1, -1),   # Up-Right
        (-1, -1): (-1, -1), # Up-Left
        (1, 1): (1, 1),     # Down-Right
        (1, -1): (-1, 1)    # Down-Left
    }

    def __init__(self, window_size: Tuple[int, int] = (1280, 800)):
        """Initialize the enhanced PyGame visualizer.
        
        Args:
            window_size: Tuple of (width, height) for the window
        """
        # Initialize PyGame and display
        pygame.init()
        pygame.font.init()
        
        # Set up display with flags for smoother rendering
        self.window_size = window_size
        
        # Base size for textures
        self.base_size = 128  # Default texture size
        
        # Check monitor resolution and adjust window size if needed
        display_info = pygame.display.Info()
        max_width = display_info.current_w - 100  # Leave some margin
        max_height = display_info.current_h - 100
        
        if window_size[0] > max_width or window_size[1] > max_height:
            # Scale down the window to fit the screen
            scale = min(max_width / window_size[0], max_height / window_size[1])
            window_size = (int(window_size[0] * scale), int(window_size[1] * scale))
            self.window_size = window_size
        
        self.screen = pygame.display.set_mode(window_size, pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE)
        pygame.display.set_caption("GridWorld RL Environment")
        
        # Find asset directories
        self.project_root = self._find_project_root()
        self.assets_dir = os.path.join(self.project_root, "assets")
        self.textures_dir = os.path.join(self.assets_dir, "textures")
        self.fonts_dir = os.path.join(self.assets_dir, "fonts")
        self.sounds_dir = os.path.join(self.assets_dir, "sounds")
        self.sprites_dir = os.path.join(self.assets_dir, "sprites")
        
        # Initialize texture cache
        self.textures = {}
        self.sprites = {}
        self.fonts = {}
        self.sounds = {}
        
        # Load assets
        self._load_assets()
        
        # Set up fonts - try to use custom fonts, fall back to system fonts
        self.title_font = self.fonts.get('title', pygame.font.SysFont('Arial', 24))
        self.font = self.fonts.get('main', pygame.font.SysFont('Arial', 20))
        self.small_font = self.fonts.get('small', pygame.font.SysFont('Arial', 16))
        
        # Initialize grid and visual parameters
        self.cell_size = None
        self.grid_offset = None
        self.info_panel_rect = None
        
        # Animation and simulation control
        self.running = False
        self.paused = False
        self.show_values = True
        self.show_policy = True
        self.animation_speed = 0.5  # seconds between steps
        self.test_mode = False      # Test mode toggle
        
        # UI state
        self.hover_buttons = set()  # Set of buttons being hovered
        self.ui_buttons = {}        # Dictionary of button rects
        self.last_ui_update = 0     # Time of last UI update for throttling
        self.is_fullscreen = False  # Fullscreen toggle
        
        # For value function visualization
        self.min_value = float('inf')
        self.max_value = float('-inf')
        
        # Animation state
        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 60
        self.dt = 1.0 / self.fps
        
        # Particle effects
        self.particles = []  # List of active particles
        self.dirty_rects = []  # For efficient rendering
        
        # Agent animation state
        self.agent_visual_pos = None  # Smoothly interpolated position
        self.agent_target_pos = None  # Target grid position
        self.agent_move_speed = 5.0   # Grid cells per second
        
        # Goal animation state
        self.goal_pulse = 0
        self.goal_pulse_speed = 2.0
        
        # UI state
        self.hover_buttons = set()  # Set of buttons being hovered
        self.ui_buttons = {}        # Dictionary of button rects
        
        # Agent information tracking
        self.active_agent_type = None    # Current agent type
        self.last_action = None          # Last action taken
        self.last_reward = None          # Last reward received
        self.last_state = None           # Last state
        self.last_transition = None      # Last state transition
    
    def _find_project_root(self) -> str:
        """Find the project root directory by looking for common markers."""
        # Start with the current file's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try to find project root by looking for common markers
        # (going up at most 3 levels from the current file)
        for _ in range(3):
            # Check if this looks like the project root
            if (os.path.isdir(os.path.join(current_dir, 'src')) and 
                os.path.isdir(os.path.join(current_dir, 'assets'))):
                return current_dir
                
            # Go up one level
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:  # We've reached the filesystem root
                break
            current_dir = parent_dir
        
        # If we can't find the project root, try to infer from the file structure
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    def _load_assets(self) -> None:
        """Load textures, fonts, sprites and sounds."""
        # Create asset directories if they don't exist
        os.makedirs(self.textures_dir, exist_ok=True)
        os.makedirs(self.fonts_dir, exist_ok=True)
        os.makedirs(self.sprites_dir, exist_ok=True)
        os.makedirs(self.sounds_dir, exist_ok=True)
        
        # Generate textures if they don't exist yet
        self._ensure_textures_exist()
        
        # Load terrain textures
        self._load_textures()
        
        # Load fonts
        self._load_fonts()
        
        # Load sprites (for agent and effects)
        self._load_sprites()
        
        # Load sounds
        self._load_sounds()
        
        # Animated terrain feature removed
    
    def _ensure_textures_exist(self) -> None:
        """Make sure texture files exist, generate them if needed."""
        try:
            from utils.texture_generator import TextureGenerator
            
            # Check if basic textures exist
            if not os.path.exists(os.path.join(self.textures_dir, "normal.png")):
                print("Generating textures...")
                generator = TextureGenerator(self.textures_dir)
                generator.generate_all()
                print("Textures generated successfully")
                
        except ImportError:
            print("Warning: texture_generator module not found, using fallback colors")
        except Exception as e:
            print(f"Warning: Failed to generate textures: {e}")
    
    def _load_textures(self) -> None:
        """Load terrain and UI textures for enhanced visual appearance."""
        # Define all textures to load
        texture_files = [
            # Terrain textures
            "normal.png",
            "ice.png",
            "mud.png", 
            "quicksand.png",
            "water.png",
            
            # Goal and UI textures
            "goal.png",
            "ui_panel.png",
            "ui_button.png",
            "ui_button_hover.png"
        ]
        
        # Create animated textures directory reference
        animated_textures_dir = os.path.join(self.textures_dir, "animated")
        
        # Mapping of static textures to animated alternatives
        animated_alternatives = {
            "normal.png": "Grass Texture 1.jpg",
            "mud.png": "mud.PNG"
        }
        
        # Load each texture
        for texture_file in texture_files:
            texture_name = os.path.splitext(texture_file)[0]
            texture_loaded = False
            
            # Try standard texture path first
            try:
                path = os.path.join(self.textures_dir, texture_file)
                if os.path.exists(path):
                    self.textures[texture_name] = pygame.image.load(path).convert_alpha()
                    print(f"Loaded texture: {texture_name}")
                    texture_loaded = True
                # If standard texture doesn't exist, try animated alternative
                elif texture_file in animated_alternatives:
                    animated_path = os.path.join(animated_textures_dir, animated_alternatives[texture_file])
                    if os.path.exists(animated_path):
                        self.textures[texture_name] = pygame.image.load(animated_path).convert_alpha()
                        # Resize to base texture size if needed
                        if self.textures[texture_name].get_width() != self.base_size or \
                           self.textures[texture_name].get_height() != self.base_size:
                            self.textures[texture_name] = pygame.transform.scale(
                                self.textures[texture_name], (self.base_size, self.base_size)
                            )
                        print(f"Loaded texture: {texture_name} (from animated folder)")
                        texture_loaded = True
                
                if not texture_loaded:
                    # Only print warning for missing terrain textures
                    if texture_file not in ["ui_panel.png", "ui_button.png", "ui_button_hover.png"]:
                        print(f"Warning: Texture file '{texture_file}' not found. Using color fallback.")
            except Exception as e:
                print(f"Warning: Failed to load texture '{texture_file}': {e}")
    
    def _load_fonts(self) -> None:
        """Load custom fonts."""
        # Try to find font files
        font_extensions = ['.ttf', '.otf']
        
        if os.path.exists(self.fonts_dir):
            for file in os.listdir(self.fonts_dir):
                for ext in font_extensions:
                    if file.lower().endswith(ext):
                        try:
                            font_path = os.path.join(self.fonts_dir, file)
                            font_name = os.path.splitext(file)[0].lower()
                            
                            # Load each font in different sizes
                            if "robot" in font_name.lower():
                                self.fonts['main'] = pygame.font.Font(font_path, 20)
                                self.fonts['small'] = pygame.font.Font(font_path, 16)
                            elif "orbit" in font_name.lower():
                                self.fonts['title'] = pygame.font.Font(font_path, 24)
                            elif "press" in font_name.lower() or "start" in font_name.lower():
                                self.fonts['pixel'] = pygame.font.Font(font_path, 16)
                                
                            print(f"Loaded font: {font_name}")
                        except Exception as e:
                            print(f"Warning: Failed to load font '{file}': {e}")
    
    def _load_sprites(self) -> None:
        """Load agent and effect sprites."""
        # Define sprite files to load
        sprite_files = [
            "agent.png", 
            "agent_up.png", 
            "agent_down.png", 
            "agent_left.png", 
            "agent_right.png",
            "particle_wind.png", 
            "particle_sparkle.png"
        ]
        
        # Check sprite folder and load sprites
        for sprite_file in sprite_files:
            try:
                path = os.path.join(self.textures_dir, sprite_file)
                if os.path.exists(path):
                    sprite_name = os.path.splitext(sprite_file)[0]
                    self.sprites[sprite_name] = pygame.image.load(path).convert_alpha()
                    print(f"Loaded sprite: {sprite_name}")
            except Exception as e:
                print(f"Warning: Failed to load sprite '{sprite_file}': {e}")
    
    def _load_sounds(self) -> None:
        """Load sound effects."""
        # Try to initialize mixer if it's not already
        if not pygame.mixer.get_init():
            try:
                pygame.mixer.init()
            except:
                print("Warning: Pygame mixer could not be initialized")
                return
        
        # Define sound files to load
        sound_files = {
            "move": "move.wav",
            "win": "win.wav",
            "collision": "collision.wav",
            "wind": "wind.wav",
            "click": "click.wav"  # For UI interactions
        }
        
        # Check sound folder and load sounds
        for sound_name, sound_file in sound_files.items():
            try:
                path = os.path.join(self.sounds_dir, sound_file)
                if os.path.exists(path):
                    self.sounds[sound_name] = pygame.mixer.Sound(path)
                    # Set appropriate volume
                    if sound_name == "click":
                        self.sounds[sound_name].set_volume(0.3)  # Quieter UI sounds
            except Exception as e:
                print(f"Warning: Failed to load sound '{sound_file}': {e}")
                
        # Create simple click sound if not found
        if "click" not in self.sounds and pygame.mixer.get_init():
            try:
                # Create a simple synthetic click sound
                buffer = bytearray(3000)
                for i in range(100):  # First part of the sound
                    buffer[i] = 127 + int(127 * math.sin(i * 0.2))
                for i in range(100, 400):  # Decay
                    buffer[i] = 127 + int((400-i)/300 * 127 * math.sin(i * 0.1))
                
                click_sound = pygame.mixer.Sound(buffer=buffer)
                click_sound.set_volume(0.2)
                self.sounds["click"] = click_sound
            except Exception as e:
                print(f"Warning: Failed to create synthetic click sound: {e}")
    
    # _load_animated_terrain method removed

    def _calculate_grid_dimensions(self, env) -> None:
        """Calculate grid cell size and offset based on environment size with improved scaling."""
        grid_height, grid_width = env.config["grid_size"]
        
        # Calculate info panel width with better scaling for different screen sizes
        if hasattr(self, 'is_fullscreen') and self.is_fullscreen:
            # In fullscreen mode, use a smaller percentage for the panel
            panel_pct = 0.2
            min_panel_width = 300
        else:
            # In windowed mode, use standard sizing
            panel_pct = 0.25
            min_panel_width = 280
            
        info_panel_width = max(min_panel_width, int(self.window_size[0] * panel_pct))
        
        # Calculate available space for grid
        available_width = self.window_size[0] - info_panel_width
        available_height = self.window_size[1]
        
        # Calculate cell size to fit grid while maintaining square cells
        # Use adaptive padding based on screen size
        padding_factor = 0.9
        if self.window_size[0] > 1600 or self.window_size[1] > 1000:
            padding_factor = 0.95  # Less padding for larger screens
            
        cell_width = available_width / grid_width * padding_factor
        cell_height = available_height / grid_height * padding_factor
        self.cell_size = min(cell_width, cell_height)
        
        # Center the grid in available space
        total_grid_width = self.cell_size * grid_width
        total_grid_height = self.cell_size * grid_height
        
        self.grid_offset = (
            (available_width - total_grid_width) / 2,
            (available_height - total_grid_height) / 2
        )
        
        # Define info panel area with border
        self.info_panel_rect = pygame.Rect(
            available_width, 0, info_panel_width, self.window_size[1]
        )
        
        # Reset UI button locations when grid is recalculated
        self.ui_buttons = {}
        
        # Initialize agent visual position if it's not set yet
        if env.agent_pos is not None and self.agent_visual_pos is None:
            grid_pos = env.agent_pos
            self.agent_visual_pos = np.array([
                self.grid_offset[0] + (grid_pos[1] + 0.5) * self.cell_size,
                self.grid_offset[1] + (grid_pos[0] + 0.5) * self.cell_size
            ])
            self.agent_target_pos = self.agent_visual_pos.copy()
            
        print(f"Grid recalculated - cell size: {self.cell_size:.1f}px, panel width: {info_panel_width}px")

    def _draw_background(self) -> None:
        """Draw a subtle gradient background for visual depth."""
        # Get window dimensions
        width, height = self.window_size
        
        # Create a subtle vertical gradient from dark to slightly lighter
        top_color = (20, 20, 30)  # Dark blue-black
        bottom_color = (30, 30, 45)  # Slightly lighter blue-black
        
        # Create a reusable gradient surface for better performance
        if not hasattr(self, 'gradient_surface') or self.frame_count <= 1:
            # Create gradient surface on first frame or when needed
            self.gradient_surface = pygame.Surface((width, height), pygame.SRCALPHA)
            
            # Draw gradient lines
            for y in range(height):
                t = y / height
                r = int(top_color[0] + (bottom_color[0] - top_color[0]) * t)
                g = int(top_color[1] + (bottom_color[1] - top_color[1]) * t)
                b = int(top_color[2] + (bottom_color[2] - top_color[2]) * t)
                pygame.draw.line(self.gradient_surface, (r, g, b), (0, y), (width, y))
        
        # Apply the gradient
        self.screen.blit(self.gradient_surface, (0, 0))
        
        # Mark the entire screen as needing update
        self.dirty_rects.append(pygame.Rect(0, 0, width, height))

    def _draw_grid(self, env) -> None:
        """Draw the grid structure with normal terrain cells using consistent colors."""
        grid_height, grid_width = env.config["grid_size"]
        
        # Draw normal terrain for all cells first - use the fixed color
        normal_color = self.COLORS['normal']  # Light green #90EE90
        
        for i in range(grid_height):
            for j in range(grid_width):
                cell_rect = self._get_cell_rect(i, j)
                
                # Always use the fixed color for consistent appearance
                pygame.draw.rect(
                    self.screen,
                    normal_color,
                    cell_rect
                )
        
        # Draw grid lines
        for i in range(grid_height + 1):
            start_pos = (
                self.grid_offset[0],
                self.grid_offset[1] + i * self.cell_size
            )
            end_pos = (
                self.grid_offset[0] + grid_width * self.cell_size,
                self.grid_offset[1] + i * self.cell_size
            )
            # Draw solid grid lines for better visibility
            pygame.draw.line(self.screen, self.COLORS['grid'], start_pos, end_pos, 1)
            
        for j in range(grid_width + 1):
            start_pos = (
                self.grid_offset[0] + j * self.cell_size,
                self.grid_offset[1]
            )
            end_pos = (
                self.grid_offset[0] + j * self.cell_size,
                self.grid_offset[1] + grid_height * self.cell_size
            )
            pygame.draw.line(self.screen, self.COLORS['grid'], start_pos, end_pos, 1)

    def _draw_terrain(self, env) -> None:
        """Draw different terrain types with enhanced visuals while maintaining consistency."""
        for terrain_type, data in env.config.get("terrain", {}).items():
            # Get the fixed color for this terrain type
            color = self.COLORS.get(terrain_type, self.COLORS['grid'])
            
            # Draw each cell of this terrain type
            for pos in data["positions"]:
                cell_rect = self._get_cell_rect(pos[0], pos[1])
                
                # Use texture if available, otherwise use color
                if terrain_type in self.textures:
                    # Scale texture to fit the cell
                    scaled_texture = pygame.transform.scale(
                        self.textures[terrain_type],
                        (int(self.cell_size), int(self.cell_size))
                    )
                    self.screen.blit(scaled_texture, cell_rect)
                else:
                    # Use solid color with visual improvements
                    pygame.draw.rect(self.screen, color, cell_rect)
                    
                    # Add some texture simulation based on terrain type
                    if terrain_type == 'ice':
                        # Add shine pattern
                        for i in range(3):
                            shine_size = int(self.cell_size * (0.2 + i * 0.15))
                            shine_pos = (
                                cell_rect.x + int(self.cell_size * 0.1) + i * int(self.cell_size * 0.15),
                                cell_rect.y + int(self.cell_size * 0.1) + i * int(self.cell_size * 0.1)
                            )
                            pygame.draw.circle(
                                self.screen, 
                                (220, 240, 255, 150), 
                                (shine_pos[0], shine_pos[1]), 
                                int(shine_size * 0.25),
                                1
                            )
                    elif terrain_type == 'mud':
                        # Add mud texture pattern
                        for _ in range(8):
                            mud_x = cell_rect.x + random.randint(5, int(self.cell_size) - 5)
                            mud_y = cell_rect.y + random.randint(5, int(self.cell_size) - 5)
                            mud_size = random.randint(3, 7)
                            pygame.draw.circle(
                                self.screen,
                                (110, 60, 15),
                                (mud_x, mud_y),
                                mud_size
                            )
                    elif terrain_type == 'quicksand':
                        # Add swirl pattern
                        center_x = cell_rect.x + self.cell_size // 2
                        center_y = cell_rect.y + self.cell_size // 2
                        for radius in range(3, int(self.cell_size * 0.4), 5):
                            pygame.draw.circle(
                                self.screen,
                                (190, 160, 120),
                                (center_x, center_y),
                                radius,
                                1
                            )
                
                # Draw a subtle border for visual separation
                pygame.draw.rect(
                    self.screen, 
                    (color[0]*0.8, color[1]*0.8, color[2]*0.8), 
                    cell_rect, 
                    1
                )
                
                # Add a label for clarity
                terrain_label = terrain_type[0].upper()  # First letter
                label = self.small_font.render(terrain_label, True, self.COLORS['text_dark'])
                label_rect = label.get_rect(center=(cell_rect.centerx, cell_rect.centery))
                self.screen.blit(label, label_rect)

    def _draw_wind_zones(self, env) -> None:
        """Draw wind zones with enhanced particle effects and visual clarity."""
        for wind_zone in env.config.get("wind_zones", []):
            direction = wind_zone["direction"]
            strength = wind_zone["strength"]
            arrow = self.WIND_ARROWS.get(direction, '•')
            
            for pos in wind_zone["area"]:
                cell_rect = self._get_cell_rect(pos[0], pos[1])
                
                # Calculate wind direction angle for visual effects
                angle_rad = math.atan2(direction[0], direction[1])  # Note: Reversed because our grid has y-down
                angle_deg = math.degrees(angle_rad)
                
                # Draw wind zone background with directional gradient
                wind_color = self.COLORS['wind']
                
                # Create a semi-transparent gradient overlay that shows wind direction
                s = pygame.Surface((int(self.cell_size), int(self.cell_size)), pygame.SRCALPHA)
                
                # Create striped pattern in the direction of wind
                stripe_width = int(max(2, self.cell_size / 20))  # Thinner stripes for larger cells
                stripe_spacing = stripe_width * 2
                
                # Calculate direction perpendicular to wind for stripe orientation
                perp_x = -direction[1]
                perp_y = direction[0]
                
                # Normalize perpendicular vector
                length = math.sqrt(perp_x*perp_x + perp_y*perp_y)
                if length > 0:
                    perp_x /= length
                    perp_y /= length
                
                # Draw stripes perpendicular to wind direction
                for offset in range(-int(self.cell_size*1.5), int(self.cell_size*1.5), stripe_spacing):
                    # Calculate start and end points of stripe
                    center_x = self.cell_size / 2
                    center_y = self.cell_size / 2
                    
                    # Calculate the offset from center along the perpendicular direction
                    offset_x = perp_x * offset
                    offset_y = perp_y * offset
                    
                    # Calculate points for the stripe
                    start_x = center_x + offset_x - direction[1] * self.cell_size
                    start_y = center_y + offset_y - direction[0] * self.cell_size
                    end_x = center_x + offset_x + direction[1] * self.cell_size
                    end_y = center_y + offset_y + direction[0] * self.cell_size
                    
                    # Draw the stripe
                    pygame.draw.line(
                        s, 
                        (wind_color[0], wind_color[1], wind_color[2], 130), 
                        (start_x, start_y), 
                        (end_x, end_y), 
                        stripe_width
                    )
                
                # Apply to cell
                self.screen.blit(s, cell_rect)
                
                # Create more particles for stronger wind
                # Scale particle generation with wind strength and frame rate
                base_particle_chance = 0.4  # Base chance per frame
                particle_chance = base_particle_chance * strength * self.dt * 60  # Adjust for framerate
                
                # Generate multiple particles per frame for stronger winds
                num_particles = 0
                if strength > 0.7:
                    num_particles = 2  # Strong wind: 2 particles
                elif strength > 0.4:
                    num_particles = 1  # Medium wind: 1 particle
                else:
                    # Weak wind: probabilistic particle generation
                    if random.random() < particle_chance:
                        num_particles = 1
                
                # Create the calculated number of particles
                for _ in range(num_particles):
                    # Create wind particle at strategic positions within cell
                    # Position particles toward the upwind side for more realistic flow
                    upwind_bias = 0.7  # Bias toward the upwind side
                    
                    # Calculate position with bias against the wind direction
                    particle_x = cell_rect.x + self.cell_size * (0.5 - direction[1] * upwind_bias * random.random())
                    particle_y = cell_rect.y + self.cell_size * (0.5 - direction[0] * upwind_bias * random.random())
                    
                    particle_pos = (particle_x, particle_y)
                    self._create_wind_particle(particle_pos, direction, strength)
                
                # Draw direction arrow with enhanced visual effect
                arrow_color = (210, 240, 255)  # Brighter blue for better contrast
                
                # Calculate arrow size based on wind strength
                arrow_size = int(self.cell_size * (0.2 + strength * 0.3))
                
                # Draw dynamic pulsing glow behind arrow
                pulse = (math.sin(self.frame_count * 0.1) + 1) * 0.5
                glow_size = arrow_size * (1.2 + pulse * 0.4)
                glow = pygame.Surface((int(glow_size), int(glow_size)), pygame.SRCALPHA)
                
                # Multi-layer glow for more dramatic effect
                for radius in range(int(glow_size/2), 0, -3):
                    alpha = int(60 * (radius / (glow_size/2)))
                    pygame.draw.circle(
                        glow,
                        (180, 220, 255, alpha),
                        (int(glow_size/2), int(glow_size/2)),
                        radius
                    )
                glow_rect = glow.get_rect(center=cell_rect.center)
                self.screen.blit(glow, glow_rect)
                
                # Draw animated arrow that subtly moves in the wind direction
                text_size = int(24 + 8 * min(1.0, strength))  # Size based on wind strength
                arrow_font = pygame.font.SysFont('Arial', text_size, bold=True)
                text = arrow_font.render(arrow, True, arrow_color)
                
                # Make arrow position subtly animated
                offset_x = math.sin(self.frame_count * 0.1) * 2 * direction[1]
                offset_y = math.sin(self.frame_count * 0.1) * 2 * direction[0]
                
                text_rect = text.get_rect(
                    center=(cell_rect.center[0] + offset_x, cell_rect.center[1] + offset_y)
                )
                self.screen.blit(text, text_rect)
                
                # Draw wind strength indicator with better contrast and styling
                strength_text = self.small_font.render(f"{strength:.1f}", True, (255, 255, 255))
                strength_bg = pygame.Surface((strength_text.get_width() + 10, strength_text.get_height() + 6), pygame.SRCALPHA)
                pygame.draw.rect(
                    strength_bg, 
                    (0, 0, 0, 120), 
                    strength_bg.get_rect(), 
                    border_radius=5
                )
                strength_bg_rect = strength_bg.get_rect(
                    center=(cell_rect.centerx, cell_rect.centery + 25)
                )
                self.screen.blit(strength_bg, strength_bg_rect)
                
                strength_rect = strength_text.get_rect(
                    center=strength_bg_rect.center
                )
                self.screen.blit(strength_text, strength_rect)
                
    def _create_wind_particle(self, position, direction, strength):
        """Create a wind particle effect at the given position."""
        # Only create particles if we have the sprite
        if "particle_wind" not in self.sprites:
            return
            
        # Convert direction tuple to vector
        vec = self.WIND_VECTORS.get(direction, (0, 0))
        
        # Enhanced wind visualization with improved parameters
        particle = {
            'pos': list(position),
            'vel': [vec[0] * (strength * 50 + random.uniform(-3, 3)), 
                   vec[1] * (strength * 50 + random.uniform(-3, 3))],
            'lifetime': random.uniform(1.0, 2.5),  # Even longer lifetime for better visibility
            'age': 0,
            'sprite': self.sprites["particle_wind"],
            'size': random.uniform(1.2, 1.8),      # Larger particles for more impact
            'color': (200, 230, 255, 240),         # More vibrant blue with higher alpha
            'pulsate': random.random() > 0.5,      # 50% chance to pulsate for visual variety
            'pulse_speed': random.uniform(3.0, 6.0) # Speed of pulsation
        }
        
        self.particles.append(particle)
        
    def _update_particles(self, dt):
        """Update all particle effects."""
        # Update existing particles
        new_particles = []
        for p in self.particles:
            # Update position
            p['pos'][0] += p['vel'][0] * dt
            p['pos'][1] += p['vel'][1] * dt
            
            # Update age
            p['age'] += dt
            
            # Keep if still alive
            if p['age'] < p['lifetime']:
                new_particles.append(p)
            
        # Replace the particle list with updated particles
        self.particles = new_particles
        
    def _draw_particles(self):
        """Draw all active particle effects with enhanced visuals."""
        for p in self.particles:
            # Calculate alpha based on remaining lifetime with smoother fade
            life_ratio = 1.0 - (p['age'] / p['lifetime'])
            
            # Improved fade curve with a stronger middle period
            if life_ratio < 0.2:
                # Fade in
                alpha_ratio = life_ratio * 5  # 0 to 1 over the first 20% of lifetime
            elif life_ratio > 0.8:
                # Fade out
                alpha_ratio = (life_ratio - 0.8) * 5  # 1 to 0 over the last 20% of lifetime
            else:
                # Full visibility in the middle 60% of lifetime
                alpha_ratio = 1.0
                
            # Base alpha on the fade curve
            base_alpha = int(255 * alpha_ratio)
            
            # Add pulsation effect for wind particles if specified
            if p.get('pulsate', False):
                pulse_factor = (math.sin(p['age'] * p['pulse_speed']) + 1) * 0.5  # 0 to 1 pulsation
                alpha = int(base_alpha * (0.7 + 0.3 * pulse_factor))  # Pulsate between 70% and 100% of base alpha
                size_factor = 1.0 + 0.2 * pulse_factor  # Size pulsation between 100% and 120%
            else:
                alpha = base_alpha
                size_factor = 1.0
                
            # Scale sprite with potential pulsation
            size = int(p['sprite'].get_width() * p['size'] * size_factor)
            scaled_sprite = pygame.transform.scale(p['sprite'], (size, size))
            
            # Get particle color with fallback to default if not present
            particle_color = p.get('color', (255, 255, 255, 200))  # White is the default fallback color
            
            # Apply color tint for all particles
            # Create a colored version of the sprite with improved blending
            colored_sprite = pygame.Surface((size, size), pygame.SRCALPHA)
            
            # More efficient coloring method
            for y in range(size):
                for x in range(size):
                    pixel = scaled_sprite.get_at((x, y))
                    if pixel[3] > 0:  # If not completely transparent
                        # Blend the particle color with the sprite
                        colored_sprite.set_at((x, y), (
                            int(particle_color[0] * 0.8 + pixel[0] * 0.2),  
                            int(particle_color[1] * 0.8 + pixel[1] * 0.2),
                            int(particle_color[2] * 0.8 + pixel[2] * 0.2),
                            min(255, int(pixel[3] * (particle_color[3] / 255)))
                        ))
            scaled_sprite = colored_sprite
            
            # Apply final alpha based on lifetime
            scaled_sprite.set_alpha(alpha)
            
            # Draw at position (centered)
            pos = (int(p['pos'][0] - size/2), int(p['pos'][1] - size/2))
            self.screen.blit(scaled_sprite, pos)
            
            # Enhanced glow effect with variable size and intensity
            glow_size = size * 1.4  # Larger glow for more impact
            glow_surf = pygame.Surface((int(glow_size), int(glow_size)), pygame.SRCALPHA)
            
            # Get particle color with fallback to default if not present
            particle_color = p.get('color', (255, 255, 255, 200))  # White is the default fallback color
                
            # For wind particles, create a more directional glow based on velocity
            if 'vel' in p and (abs(p['vel'][0]) > 20 or abs(p['vel'][1]) > 20):
                # Normalize velocity vector
                vel_mag = math.sqrt(p['vel'][0]**2 + p['vel'][1]**2)
                norm_vel = [p['vel'][0]/vel_mag, p['vel'][1]/vel_mag]
                
                # Create directional glow (elongated ellipse)
                stretch_factor = 1.5
                ellipse_width = int(glow_size / stretch_factor)
                ellipse_height = int(glow_size * stretch_factor)
                
                # Rotate surface based on velocity direction
                angle = math.degrees(math.atan2(norm_vel[1], norm_vel[0])) - 90
                
                # Create elongated glow
                pygame.draw.ellipse(
                    glow_surf,
                    (particle_color[0], particle_color[1], particle_color[2], int(alpha * 0.4)),
                    pygame.Rect(
                        glow_size/2 - ellipse_width/2, 
                        glow_size/2 - ellipse_height/2,
                        ellipse_width, 
                        ellipse_height
                    )
                )
                
                # Rotate the glow to match direction
                glow_surf = pygame.transform.rotate(glow_surf, angle)
                
                # Get the new rect after rotation
                glow_rect = glow_surf.get_rect(center=(glow_size/2, glow_size/2))
                
                # Calculate new position considering rotated surface
                glow_pos = (int(p['pos'][0] - glow_rect.width/2), int(p['pos'][1] - glow_rect.height/2))
            else:
                # Regular circular glow for non-wind particles
                pygame.draw.circle(
                    glow_surf, 
                    (particle_color[0], particle_color[1], particle_color[2], int(alpha * 0.4)),
                    (int(glow_size/2), int(glow_size/2)), 
                    int(glow_size/2)
                )
                glow_pos = (int(p['pos'][0] - glow_size/2), int(p['pos'][1] - glow_size/2))
            
            # Draw the glow
            self.screen.blit(glow_surf, glow_pos)
            
            # Add to dirty rects for efficient rendering
            self.dirty_rects.append(pygame.Rect(
                min(pos[0], glow_pos[0]),
                min(pos[1], glow_pos[1]),
                max(size, glow_surf.get_width()),
                max(size, glow_surf.get_height())
            ))

    def _draw_value_heatmap(self, env, agent) -> None:
        """Disabled: Value function heatmap for clarity and realism focus.
        
        Note: Heatmap visualization is disabled to focus on terrain clarity
        and prevent visual confusion in the environment.
        """
        # Function intentionally disabled - heatmaps would create visual confusion
        return

    def _draw_policy_arrows(self, env, agent) -> None:
        """Draw policy arrows showing the best action in each state with visual enhancements."""
        if not self.show_policy:
            return
            
        action_arrows = ['↑', '→', '↓', '←']
        # Direction vectors for drawing arrow lines (dy, dx)
        action_vectors = [
            (0, -0.3),  # Up
            (0.3, 0),   # Right
            (0, 0.3),   # Down
            (-0.3, 0)   # Left
        ]
        
        for i in range(env.grid_height):
            for j in range(env.grid_width):
                if (i, j) == tuple(env.config["goal_pos"]):
                    continue
                    
                action = None
                q_values = None
                
                try:
                    if hasattr(agent, 'Q'):  # Q-Learning or SARSA
                        q_values = agent.Q[i, j]
                        action = np.argmax(q_values)
                    elif hasattr(agent, 'policy'):  # Value/Policy Iteration
                        action = agent.policy[i, j]
                    elif hasattr(agent, 'online_net') or isinstance(agent, torch.nn.Module):  # DQN
                        device = getattr(agent, 'device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                        model = agent.online_net if hasattr(agent, 'online_net') else agent
                        # Create a properly formatted state tensor for DQN
                        state = torch.FloatTensor([i, j]).to(device)
                        
                        try:
                            # Try different input formats to handle various model architectures
                            with torch.no_grad():
                                # Format 1: Direct input
                                q_values_tensor = model(state)
                                if q_values_tensor.dim() > 1:
                                    action = q_values_tensor.argmax(dim=1).item()
                                    q_values = q_values_tensor[0].cpu().numpy()
                                else:
                                    action = q_values_tensor.argmax().item()
                                    q_values = q_values_tensor.cpu().numpy()
                        except Exception:
                            try:
                                # Format 2: Unsqueezed input (batch dimension)
                                with torch.no_grad():
                                    q_values_tensor = model(state.unsqueeze(0))
                                    action = q_values_tensor.argmax(dim=1).item()
                                    q_values = q_values_tensor[0].cpu().numpy()
                            except Exception:
                                # If all attempts fail, default to no action (will be skipped)
                                action = None
                except Exception as e:
                    # If visualization fails, print error but continue
                    print(f"Warning: Error drawing policy arrow: {e}")
                    continue
                
                if action is not None:
                    cell_rect = self._get_cell_rect(i, j)
                    cell_center = cell_rect.center
                    
                    # Draw fancy arrow based on action
                    # Change arrow color based on animation time for pulsing effect
                    pulse = (math.sin(self.frame_count * 0.05) + 1) * 0.5
                    base_color = (150, 200, 255)  # Light blue
                    arrow_color = (
                        int(base_color[0] + pulse * 50),
                        int(base_color[1] + pulse * 50),
                        int(base_color[2])
                    )
                    
                    # Draw arrow with larger size and glow effect
                    arrow_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                    
                    # Draw arrow glow (larger, semi-transparent)
                    glow_size = int(self.cell_size * 0.35)
                    pygame.draw.circle(
                        arrow_surface,
                        (*arrow_color, 60),  # Semi-transparent
                        (self.cell_size // 2, self.cell_size // 2),
                        glow_size
                    )
                    
                    # Draw arrow text
                    arrow_font = self.fonts.get('title', self.font)
                    text = arrow_font.render(action_arrows[action], True, arrow_color)
                    text_rect = text.get_rect(center=(self.cell_size // 2, self.cell_size // 2))
                    arrow_surface.blit(text, text_rect)
                    
                    # Draw a line in the direction of the action
                    dx, dy = action_vectors[action]
                    start_pos = (self.cell_size // 2, self.cell_size // 2)
                    end_pos = (int(self.cell_size * (0.5 + dx)), int(self.cell_size * (0.5 + dy)))
                    pygame.draw.line(arrow_surface, arrow_color, start_pos, end_pos, 3)
                    
                    # Draw a small circle at the end of the line
                    pygame.draw.circle(arrow_surface, arrow_color, end_pos, 5)
                    
                    # Apply to screen
                    self.screen.blit(arrow_surface, cell_rect)
                    
                    # If we have Q-values, visualize the action confidence
                    if q_values is not None and len(q_values) >= 4:
                        # Calculate action confidence (how much better is the best action)
                        # compared to the average of all other actions
                        best_q = q_values[action]
                        other_qs = [q_values[a] for a in range(len(q_values)) if a != action]
                        if other_qs:
                            avg_other_q = sum(other_qs) / len(other_qs)
                            confidence = max(0, min(1, (best_q - avg_other_q) / max(1, abs(best_q))))
                            
                            # Draw confidence indicator as a small bar below the arrow
                            bar_width = int(self.cell_size * 0.6 * confidence)
                            bar_height = 4
                            bar_rect = pygame.Rect(
                                cell_rect.centerx - bar_width // 2,
                                cell_rect.centery + self.cell_size // 4,
                                bar_width,
                                bar_height
                            )
                            pygame.draw.rect(self.screen, arrow_color, bar_rect)

    def _draw_goal(self, env) -> None:
        """Draw the goal with consistent appearance using texture if available."""
        goal_pos = env.config["goal_pos"]
        goal_rect = self._get_cell_rect(*goal_pos)
        
        # Add a subtle glow effect for all goal renderings
        self.goal_pulse = (self.goal_pulse + self.dt * self.goal_pulse_speed) % (2 * math.pi)
        glow_size = int(self.cell_size * (0.9 + 0.1 * math.sin(self.goal_pulse)))
        glow_surface = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        
        # Create a circular glow
        glow_color = (120, 255, 120, 100)
        pygame.draw.circle(glow_surface, glow_color, 
                          (glow_size // 2, glow_size // 2), 
                          glow_size // 2)
        
        # Position the glow centered on the goal
        glow_pos = (
            goal_rect.centerx - glow_size // 2,
            goal_rect.centery - glow_size // 2
        )
        
        # Draw glow first (under the goal texture or color)
        self.screen.blit(glow_surface, glow_pos)
        
        # First check textures collection (preferred)
        if "goal" in self.textures:
            # Scale texture to fit the cell
            scaled_texture = pygame.transform.scale(
                self.textures["goal"],
                (int(self.cell_size), int(self.cell_size))
            )
            self.screen.blit(scaled_texture, goal_rect)
        # Then check sprites collection (fallback)
        elif "goal" in self.sprites:
            # Scale sprite to fit the cell
            scaled_sprite = pygame.transform.scale(
                self.sprites["goal"],
                (int(self.cell_size), int(self.cell_size))
            )
            self.screen.blit(scaled_sprite, goal_rect)
        else:
            # Final fallback to color if no texture is available
            pygame.draw.rect(self.screen, self.COLORS['goal'], goal_rect)
            
            # Add a border for better visibility
            pygame.draw.rect(self.screen, (40, 180, 80), goal_rect, 2)
            
            # Add a clear label
            text = self.font.render("G", True, self.COLORS['text_dark'])
            text_rect = text.get_rect(center=goal_rect.center)
            self.screen.blit(text, text_rect)
            
        # Create sparkle particles occasionally for additional effect
        if random.random() < 0.1:  # 10% chance each frame
            particle_pos = (
                goal_rect.centerx + random.uniform(-self.cell_size/4, self.cell_size/4),
                goal_rect.centery + random.uniform(-self.cell_size/4, self.cell_size/4)
            )
            self._create_sparkle_particle(particle_pos)
    
    def _create_sparkle_particle(self, position):
        """Create a sparkle particle at the given position."""
        if "particle_sparkle" not in self.sprites:
            return
            
        # Create sparkle with random movement and lifetime
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(5, 20)
        particle = {
            'pos': list(position),
            'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
            'lifetime': random.uniform(0.3, 1.0),
            'age': 0,
            'sprite': self.sprites["particle_sparkle"],
            'size': random.uniform(0.4, 0.8),
            'rot_speed': random.uniform(-5, 5),  # Rotation speed in radians/sec
            'color': (255, 255, 160, 220)  # Bright yellow color with high alpha
        }
        
        self.particles.append(particle)
    
    def _draw_agent(self, env) -> None:
        """Draw the agent with smooth animation."""
        # Get target position in grid coordinates
        grid_pos = env.agent_pos
        
        # Calculate target pixel position
        target_pos = np.array([
            self.grid_offset[0] + (grid_pos[1] + 0.5) * self.cell_size,
            self.grid_offset[1] + (grid_pos[0] + 0.5) * self.cell_size
        ])
        
        # Initialize agent position if needed
        if self.agent_visual_pos is None:
            self.agent_visual_pos = target_pos.copy()
            self.agent_target_pos = target_pos.copy()
            
        # Update target position
        self.agent_target_pos = target_pos.copy()
        
        # Smoothly animate the agent position
        if not np.array_equal(self.agent_visual_pos, self.agent_target_pos):
            # Calculate movement vector
            move_vec = self.agent_target_pos - self.agent_visual_pos
            distance = np.linalg.norm(move_vec)
            
            # Calculate maximum movement for this frame
            max_move = self.cell_size * self.agent_move_speed * self.dt
            
            if distance <= max_move:
                # We can reach the target this frame
                self.agent_visual_pos = self.agent_target_pos.copy()
            else:
                # Move towards target
                self.agent_visual_pos += move_vec * max_move / distance
        
        # Determine agent direction for appropriate sprite
        direction = "agent"  # Default
        if hasattr(self, 'previous_agent_pos') and self.previous_agent_pos is not None:
            # Calculate direction from previous to current position
            dy = grid_pos[0] - self.previous_agent_pos[0]
            dx = grid_pos[1] - self.previous_agent_pos[1]
            
            if abs(dy) > abs(dx):  # Vertical movement dominates
                direction = "agent_down" if dy > 0 else "agent_up"
            else:  # Horizontal movement dominates
                direction = "agent_right" if dx > 0 else "agent_left"
        
        # Update previous position
        self.previous_agent_pos = grid_pos.copy()
        
        # Get the appropriate sprite
        agent_sprite = self.sprites.get(direction, self.sprites.get("agent"))
        
        if agent_sprite is not None:
            # Scale sprite to appropriate size (80% of cell size)
            sprite_size = int(self.cell_size * 0.8)
            scaled_sprite = pygame.transform.scale(agent_sprite, (sprite_size, sprite_size))
            
            # Draw sprite centered on agent position
            sprite_pos = (
                int(self.agent_visual_pos[0] - sprite_size // 2),
                int(self.agent_visual_pos[1] - sprite_size // 2)
            )
            self.screen.blit(scaled_sprite, sprite_pos)
            
            # Add to dirty rects for efficient rendering
            self.dirty_rects.append(pygame.Rect(
                sprite_pos[0], sprite_pos[1], sprite_size, sprite_size
            ))
        else:
            # Fallback to simple circle if sprites aren't available
            pygame.draw.circle(
                self.screen, 
                self.COLORS['agent'],
                (int(self.agent_visual_pos[0]), int(self.agent_visual_pos[1])), 
                int(self.cell_size * 0.4)
            )
    
    def _draw_info_panel(self, env, agent=None, episode: Optional[int] = None,
                         step: Optional[int] = None, reward: Optional[float] = None) -> None:
        """Draw a modern information panel on the right side."""
        if self.info_panel_rect is None:
            return
            
        # Draw panel background
        if "ui_panel" in self.textures:
            # Scale panel texture
            panel_texture = pygame.transform.scale(
                self.textures["ui_panel"],
                (self.info_panel_rect.width, self.info_panel_rect.height)
            )
            self.screen.blit(panel_texture, self.info_panel_rect)
        else:
            # Fallback to simple rectangle
            pygame.draw.rect(self.screen, self.COLORS['ui_panel_bg'], self.info_panel_rect)
            pygame.draw.rect(self.screen, self.COLORS['ui_panel_border'], self.info_panel_rect, 2)  # Border
        
        # Initialize text position
        panel_x = self.info_panel_rect.x + 20
        y_offset = 30
        line_height = 30
        small_line_height = 25
        
        # Draw title
        title_font = self.fonts.get('title', self.font)
        title = title_font.render("GridWorld RL", True, (220, 220, 255))
        self.screen.blit(title, (panel_x, y_offset))
        y_offset += line_height + 10
        
        # Draw agent type with box highlighting
        agent_type = self.active_agent_type if self.active_agent_type else "Unknown Agent"
            
        # Draw agent info box with border
        agent_box = pygame.Rect(panel_x - 10, y_offset - 5, self.info_panel_rect.width - 30, 40)
        pygame.draw.rect(self.screen, (50, 50, 60), agent_box)
        pygame.draw.rect(self.screen, (100, 100, 120), agent_box, 1)
        
        # Add agent icon if available
        if "agent" in self.sprites:
            icon_size = 30
            icon = pygame.transform.scale(self.sprites["agent"], (icon_size, icon_size))
            self.screen.blit(icon, (panel_x, y_offset))
            agent_text = self.font.render(f"Agent: {agent_type}", True, self.COLORS['ui_header'])
            self.screen.blit(agent_text, (panel_x + icon_size + 10, y_offset))
        else:
            agent_text = self.font.render(f"Agent: {agent_type}", True, self.COLORS['ui_header'])
            self.screen.blit(agent_text, (panel_x, y_offset))
            
        y_offset += line_height + 10
        
        # Draw stats in a box
        stats_box = pygame.Rect(panel_x - 10, y_offset, self.info_panel_rect.width - 30, 150)
        pygame.draw.rect(self.screen, (45, 45, 55), stats_box)
        pygame.draw.rect(self.screen, (90, 90, 110), stats_box, 1)
        
        stats_x = panel_x
        stats_y = y_offset + 10
        
        # Episode info
        if episode is not None:
            text = self.font.render(f"Episode: {episode}", True, (220, 220, 255))
            self.screen.blit(text, (stats_x, stats_y))
            stats_y += line_height
        
        # Step info
        if step is not None:
            text = self.font.render(f"Step: {step}", True, (220, 220, 255))
            self.screen.blit(text, (stats_x, stats_y))
            stats_y += line_height
        
        # Last action taken with color indicator
        if hasattr(self, 'last_action') and self.last_action is not None:
            # Map action to direction name
            action_names = ['Up', 'Right', 'Down', 'Left']
            action_colors = [
                self.COLORS['action_up'],
                self.COLORS['action_right'],
                self.COLORS['action_down'],
                self.COLORS['action_left']
            ]
            
            if 0 <= self.last_action < len(action_names):
                action_name = action_names[self.last_action]
                action_color = action_colors[self.last_action]
                
                # Draw color indicator
                indicator_rect = pygame.Rect(stats_x, stats_y + 8, 15, 15)
                pygame.draw.rect(self.screen, action_color, indicator_rect)
                pygame.draw.rect(self.screen, (180, 180, 200), indicator_rect, 1)
                
                text = self.font.render(f"Action: {action_name}", True, (220, 220, 255))
                self.screen.blit(text, (stats_x + 25, stats_y))
                stats_y += line_height
        
        # Reward info with color based on value
        reward_to_display = reward if reward is not None else self.last_reward
        if reward_to_display is not None:
            reward_color = (220, 220, 255)  # Default color
            if reward_to_display > 0:
                # Green for positive rewards
                intensity = min(1.0, reward_to_display / 100.0)
                reward_color = (
                    int(220 * (1 - intensity)),
                    int(220 + 35 * intensity),
                    int(220 * (1 - intensity) + 35 * intensity)
                )
            elif reward_to_display < 0:
                # Red for negative rewards
                intensity = min(1.0, abs(reward_to_display) / 100.0)
                reward_color = (
                    int(220 + 35 * intensity),
                    int(220 * (1 - intensity)),
                    int(220 * (1 - intensity))
                )
                
            reward_text = f"Reward: {reward_to_display:.1f}" if reward_to_display is not None else "Reward: 0.0"
            text = self.font.render(reward_text, True, reward_color)
            self.screen.blit(text, (stats_x, stats_y))
            stats_y += line_height
            
        # State transition visualization if available
        if self.last_transition is not None:
            from_state, to_state = self.last_transition
            if from_state != to_state:  # Only show if there was actual movement
                transition_text = f"Move: ({from_state[0]},{from_state[1]}) → ({to_state[0]},{to_state[1]})"
                text = self.small_font.render(transition_text, True, (200, 200, 230))
                self.screen.blit(text, (stats_x, stats_y))
        
        y_offset += 170  # Move below stats box
        
        # Draw the header for the buttons section
        y_offset += 20
        header_text = self.font.render("Controls", True, (200, 200, 240))
        header_rect = pygame.Rect(panel_x - 10, y_offset, self.info_panel_rect.width - 30, 30)
        pygame.draw.line(self.screen, (100, 100, 130), 
                        (panel_x - 10, y_offset + 25),
                        (panel_x + self.info_panel_rect.width - 40, y_offset + 25), 2)
        self.screen.blit(header_text, (panel_x, y_offset))
        
        # Setup for buttons that will be drawn by _draw_buttons
        y_offset += 35  # Leave space for the buttons to be drawn
        
        # We'll let the _draw_buttons method handle drawing the actual buttons
        # Just update button labels based on state
        self.BUTTONS['toggle_values']['label'] = "Hide Values" if self.show_values else "Show Values"
        self.BUTTONS['toggle_policy']['label'] = "Hide Policy" if self.show_policy else "Show Policy"
        self.BUTTONS['test_mode']['label'] = "Exit Test Mode" if self.test_mode else "Test Mode"
        self.BUTTONS['fullscreen']['label'] = "Exit Fullscreen" if self.is_fullscreen else "Fullscreen"
        
        # Add keyboard controls section
        controls_y = y_offset + 280  # Position below where buttons will be drawn
        header = self.font.render("Keyboard Controls", True, (180, 180, 220))
        self.screen.blit(header, (panel_x, controls_y))
        controls_y += line_height
        
        controls = [
            ("Space", "Pause"),
            ("V", "Toggle Values"),
            ("P", "Toggle Policy"),
            ("+", "Speed Up"),
            ("-", "Slow Down"),
            ("Q", "Quit")
        ]
        
        for key, action in controls:
            # Draw key in a small box
            key_text = self.small_font.render(key, True, (220, 220, 255))
            key_box = pygame.Rect(panel_x, controls_y, 30, 20)
            pygame.draw.rect(self.screen, (60, 60, 80), key_box)
            pygame.draw.rect(self.screen, (100, 100, 120), key_box, 1)
            
            # Center text in box
            key_rect = key_text.get_rect(center=key_box.center)
            self.screen.blit(key_text, key_rect)
            
            # Draw action description
            action_text = self.small_font.render(action, True, (180, 180, 220))
            self.screen.blit(action_text, (panel_x + 40, controls_y))
            
            controls_y += small_line_height
        
        # Add mode indicator at the bottom
        mode_text = "Test Mode" if self.test_mode else "Training Mode"
        mode_color = (100, 255, 100) if self.test_mode else (255, 200, 100)
        text = self.font.render(mode_text, True, mode_color)
        self.screen.blit(text, (panel_x, self.info_panel_rect.bottom - 60))
        
        # Add FPS counter at the bottom
        fps_text = self.small_font.render(f"FPS: {int(1 / max(0.001, self.dt))}", True, (150, 150, 180))
        self.screen.blit(fps_text, (panel_x, self.info_panel_rect.bottom - 30))

    def _draw_buttons(self) -> None:
        """Draw UI buttons on the panel with improved layout."""
        # Calculate button spacing based on screen size and number of buttons
        active_buttons = [k for k, v in self.BUTTONS.items() if v.get('active', True) and v['position'] == 'panel']
        num_buttons = len(active_buttons)
        
        if num_buttons == 0:
            return
            
        # Calculate adaptive spacing
        panel_height = self.info_panel_rect.height
        available_height = panel_height - 300  # Reserve space for other UI elements
        button_height = self.BUTTONS[active_buttons[0]]['size'][1]
        
        # Calculate spacing to distribute buttons evenly
        button_spacing = min(45, max(40, available_height / (num_buttons + 1)))
        
        # Start buttons at a calculated position
        button_start_y = 200  # Starting position
        
        # Clear previous button positions
        self.ui_buttons = {}
        
        # Draw each active button in the panel
        i = 0
        for button_key, button in self.BUTTONS.items():
            if button.get('active', True) and button['position'] == 'panel':
                # Center button horizontally in the panel
                x = self.info_panel_rect.x + (self.info_panel_rect.width - button['size'][0]) // 2
                y = button_start_y + i * button_spacing
                
                # Update button label based on state
                label = button['label']
                is_hover = button_key in self.hover_buttons
                
                # Draw the button with hover effect and tooltip
                self.ui_buttons[button_key] = self._draw_button(
                    button_key, label, (x, y), button['size'], is_hover
                )
                
                # Draw tooltip if button is hovered
                if is_hover and 'tooltip' in button:
                    tooltip_y = y + button['size'][1] + 5
                    tooltip_text = self.small_font.render(button['tooltip'], True, (200, 200, 230))
                    tooltip_rect = tooltip_text.get_rect(center=(x + button['size'][0]//2, tooltip_y))
                    self.screen.blit(tooltip_text, tooltip_rect)
                
                i += 1
    
    def _draw_button(self, key, label, pos, size, is_hover=False):
        """Draw a button with hover effect and return its rect."""
        button_rect = pygame.Rect(pos, size)
        
        # Use texture if available
        if "ui_button" in self.textures and "ui_button_hover" in self.textures:
            texture = self.textures["ui_button_hover"] if is_hover else self.textures["ui_button"]
            # Scale texture to button size
            scaled_texture = pygame.transform.scale(texture, size)
            self.screen.blit(scaled_texture, button_rect)
        else:
            # Fallback to colors with improved visual style
            if is_hover:
                # Create a gradient for hover state
                button_surface = pygame.Surface(size, pygame.SRCALPHA)
                color1 = self.COLORS['ui_button_hover']
                color2 = (min(color1[0] + 20, 255), min(color1[1] + 20, 255), min(color1[2] + 20, 255))
                
                for i in range(size[1]):
                    ratio = i / size[1]
                    r = int(color1[0] + (color2[0] - color1[0]) * ratio)
                    g = int(color1[1] + (color2[1] - color1[1]) * ratio)
                    b = int(color1[2] + (color2[2] - color1[2]) * ratio)
                    pygame.draw.line(button_surface, (r, g, b), (0, i), (size[0], i))
                    
                self.screen.blit(button_surface, button_rect)
                # Add a highlight border
                pygame.draw.rect(self.screen, (180, 180, 220), button_rect, 2)
            else:
                # Normal state
                color = self.COLORS['ui_button']
                pygame.draw.rect(self.screen, color, button_rect)
                pygame.draw.rect(self.screen, self.COLORS['ui_button_border'], button_rect, 1)
        
        # Draw text with shadow for better readability
        shadow_color = (20, 20, 30)
        text_color = (230, 230, 255) if is_hover else self.COLORS['ui_button_text']
        
        # Text shadow
        shadow_text = self.small_font.render(label, True, shadow_color)
        shadow_rect = shadow_text.get_rect(center=(button_rect.center[0] + 1, button_rect.center[1] + 1))
        self.screen.blit(shadow_text, shadow_rect)
        
        # Main text
        text = self.small_font.render(label, True, text_color)
        text_rect = text.get_rect(center=button_rect.center)
        self.screen.blit(text, text_rect)
        
        # Add tooltip if hovering
        if is_hover and 'tooltip' in self.BUTTONS[key]:
            tooltip = self.BUTTONS[key]['tooltip']
            tooltip_surface = self.small_font.render(tooltip, True, (200, 200, 200))
            tooltip_rect = tooltip_surface.get_rect(midtop=(button_rect.centerx, button_rect.bottom + 5))
            
            # Draw tooltip background
            padding = 5
            tooltip_bg_rect = pygame.Rect(
                tooltip_rect.left - padding, 
                tooltip_rect.top - padding,
                tooltip_rect.width + padding * 2, 
                tooltip_rect.height + padding * 2
            )
            pygame.draw.rect(self.screen, (40, 40, 50), tooltip_bg_rect)
            pygame.draw.rect(self.screen, (80, 80, 100), tooltip_bg_rect, 1)
            
            # Draw tooltip text
            self.screen.blit(tooltip_surface, tooltip_rect)
        
        return button_rect
        
    def _handle_button_click(self, button_action):
        """Handle button click actions."""
        # Play click sound if available
        if 'click' in self.sounds:
            self.sounds['click'].play()
        
        if button_action == 'toggle_values':
            self.show_values = not self.show_values
            return True
        elif button_action == 'toggle_policy':
            self.show_policy = not self.show_policy
            return True
        elif button_action == 'test_mode':
            self.test_mode = not self.test_mode
            return True
        elif button_action == 'fullscreen':
            # Toggle fullscreen mode with improved scaling
            self.is_fullscreen = not self.is_fullscreen
            if self.is_fullscreen:
                # Save window size for restoring later
                self._windowed_size = self.window_size
                # Get current display info
                display_info = pygame.display.Info()
                
                # Set fullscreen mode with desktop resolution
                self.screen = pygame.display.set_mode(
                    (display_info.current_w, display_info.current_h),
                    pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
                )
                self.window_size = (display_info.current_w, display_info.current_h)
                
                # Store scale factor between windowed and fullscreen for UI scaling
                self.width_scale = display_info.current_w / self._windowed_size[0]
                self.height_scale = display_info.current_h / self._windowed_size[1]
                
                # Play sound effect if available
                if "click" in self.sounds:
                    self.sounds["click"].play()
            else:
                # Restore windowed mode with previous size
                self.screen = pygame.display.set_mode(
                    self._windowed_size,
                    pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE
                )
                self.window_size = self._windowed_size
                
                # Reset scale factors
                self.width_scale = 1.0
                self.height_scale = 1.0
                
                # Play sound effect if available
                if "click" in self.sounds:
                    self.sounds["click"].play()
                    
            # Clear any cached UI surfaces that need rescaling
            if hasattr(self, 'gradient_surface'):
                del self.gradient_surface
                
            # Force regeneration of all UI elements
            self.cell_size = None
            self.grid_offset = None
            self.info_panel_rect = None
            return 'resize'
        elif button_action == 'restart':
            # Signal to restart the simulation
            return 'restart'
        elif button_action == 'quit':
            # Signal to quit the simulation
            self.running = False
            return 'quit'
        return False
        
    def _process_mouse_events(self, event):
        """Process mouse events for UI interaction."""
        # Throttle UI updates for better performance
        current_time = time.time()
        throttle_ui = False
        
        # Only throttle motion events (clicks should always be responsive)
        if event.type == pygame.MOUSEMOTION:
            if current_time - self.last_ui_update < 0.033:  # Limit to ~30fps for UI updates
                throttle_ui = True
            else:
                self.last_ui_update = current_time
        
        if event.type == pygame.MOUSEMOTION and not throttle_ui:
            # Check for button hover
            old_hover_buttons = self.hover_buttons.copy()
            self.hover_buttons.clear()
            
            for button_key, button_rect in self.ui_buttons.items():
                if button_rect.collidepoint(event.pos):
                    self.hover_buttons.add(button_key)
                    # Change cursor to hand when hovering over buttons
                    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                    break
            else:
                # Reset cursor if not hovering over any button
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
            
            # Return True if hover state changed
            return old_hover_buttons != self.hover_buttons
            
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Check for button clicks
            if event.button == 1:  # Left mouse button
                for button_key, button_rect in self.ui_buttons.items():
                    if button_rect.collidepoint(event.pos):
                        # Visual feedback for click
                        self.hover_buttons.add(button_key)
                        # Handle the click action
                        button_action = self.BUTTONS[button_key]['action']
                        result = self._handle_button_click(button_action)
                        return result
        
        # Handle window resize event with improved scaling behavior
        elif event.type == pygame.VIDEORESIZE:
            # Store previous size for calculating scale
            prev_size = self.window_size
            
            # Update window size
            self.window_size = (event.w, event.h)
            
            # Calculate scale factors compared to previous size
            if not hasattr(self, 'width_scale'):
                self.width_scale = 1.0
                self.height_scale = 1.0
            else:
                # Update scale factors incrementally
                self.width_scale *= (event.w / prev_size[0])
                self.height_scale *= (event.h / prev_size[1])
                
            # Recreate screen with new size
            self.screen = pygame.display.set_mode(
                self.window_size, 
                pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE
            )
            
            # Clear any cached UI surfaces
            if hasattr(self, 'gradient_surface'):
                del self.gradient_surface
                
            # Reset grid calculations for new size
            self.cell_size = None
            self.grid_offset = None
            self.info_panel_rect = None
            
            # Print info about the resize
            print(f"Window resized to {self.window_size}, scale: {self.width_scale:.2f}x{self.height_scale:.2f}")
            return 'resize'
                        
        return False

    def render(self, env, agent, episode: Optional[int] = None,
              step: Optional[int] = None, reward: Optional[float] = None) -> None:
        """Render the current state of the environment with visual enhancements.
        
        Args:
            env: GridWorld environment instance
            agent: RL agent instance
            episode: Current episode number
            step: Current step number
            reward: Current reward
        """
        # Make sure grid dimensions are calculated before rendering
        if self.cell_size is None or self.grid_offset is None:
            self._calculate_grid_dimensions(env)
            
        # Record last state and action for visualization
        if hasattr(env, 'agent_pos'):
            # Store the last state
            if self.last_state is None:
                self.last_state = tuple(env.agent_pos)
            else:
                # Store the last transition
                self.last_transition = (self.last_state, tuple(env.agent_pos))
                self.last_state = tuple(env.agent_pos)
        
        # Calculate frame time for smooth animations
        current_time = time.time()
        if hasattr(self, 'last_time'):
            self.dt = min(0.1, current_time - self.last_time)  # Cap at 0.1s to prevent large jumps
        self.last_time = current_time
        self.frame_count += 1
        
        # Store current reward
        self.last_reward = reward
        
        # Reset dirty rects tracking for this frame
        self.dirty_rects = []
        
        # Update particle effects
        self._update_particles(self.dt)
        
        # Fill the background with a consistent dark color
        self.screen.fill((20, 20, 30))  # Dark blue-black background
        
        # Draw a subtle gradient background for depth
        self._draw_background()
        
        # Draw grid and terrain elements
        self._draw_grid(env)
        self._draw_terrain(env)
        self._draw_wind_zones(env)
        
        # Try to detect agent type if not set
        if self.active_agent_type is None and agent is not None:
            if hasattr(agent, 'algorithm'):
                self.active_agent_type = agent.algorithm
            elif hasattr(agent, '__class__'):
                self.active_agent_type = agent.__class__.__name__
                # Clean up the name if it's a class name
                if self.active_agent_type.endswith('Agent'):
                    self.active_agent_type = self.active_agent_type[:-5]  # Remove 'Agent'
                if self.active_agent_type == "DQN":
                    self.active_agent_type = "DQN"  # Keep as is, it's an acronym
        
        if agent is not None:
            self._draw_value_heatmap(env, agent)
            self._draw_policy_arrows(env, agent)
        
        # Draw goal and particles
        self._draw_goal(env)
        self._draw_particles()
        
        # Draw agent with animation
        self._draw_agent(env)
        
        # Clear UI buttons before redrawing
        self.ui_buttons = {}
        
        # Draw info panel
        self._draw_info_panel(env, agent, episode, step, reward)
        
        # Draw buttons
        self._draw_buttons()
        
        # Draw help text at the bottom of the screen
        help_text = "F: Fullscreen | Space: Pause | V: Values | P: Policy | T: Test Mode | R: Restart | Esc: Quit"
        help_surface = self.small_font.render(help_text, True, (150, 150, 180))
        help_rect = help_surface.get_rect(bottomleft=(10, self.window_size[1] - 10))
        self.screen.blit(help_surface, help_rect)
        
        # IMPORTANT: Always use display.flip() to update the entire screen
        # This fixes the black screen issue by ensuring the full environment is always drawn
        pygame.display.flip()

    def simulate(self, env, agent, num_episodes: int = 1, max_steps: int = 200) -> None:
        """Run an interactive simulation of the environment.
        
        Args:
            env: GridWorld environment instance
            agent: RL agent instance
            num_episodes: Number of episodes to simulate
            max_steps: Maximum steps per episode
        """
        self.running = True
        episode = 0
        
        try:
            # Force an initial complete render of the environment before starting simulation
            # This ensures the grid is visible immediately upon startup
            self.render(env, agent, episode=1, step=0, reward=0.0)
            # Ensure display is properly initialized with two calls
            # (sometimes first flip doesn't display correctly due to driver issues)
            pygame.display.flip()
            pygame.time.wait(50)  # Short pause
            pygame.display.flip()  # Second flip to ensure display is refreshed
            pygame.time.wait(300)  # Pause to ensure first frame is visible
            
            while self.running and episode < num_episodes:
                obs = env.reset()
                done = False
                total_reward = 0
                step = 0
                
                # Render the initial state of each episode
                self.render(env, agent, episode=episode+1, step=0, reward=total_reward)
                pygame.display.flip()  # Full screen update for each new episode
                
                while not done and step < max_steps:
                    # Handle events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.running = False
                            break
                        elif event.type == pygame.MOUSEMOTION or event.type == pygame.MOUSEBUTTONDOWN:
                            # Handle mouse interactions
                            ui_changed = self._process_mouse_events(event)
                            
                            if ui_changed:
                                # Handle different UI change results
                                if ui_changed == 'restart':
                                    # Reset the environment and start over
                                    obs = env.reset()
                                    done = False
                                    total_reward = 0
                                    step = 0
                                    self.render(env, agent, episode+1, step, total_reward)
                                    pygame.display.flip()
                                    continue
                                elif ui_changed == 'resize':
                                    # Recalculate grid after resize
                                    self._calculate_grid_dimensions(env)
                                
                                # Update the display for any UI change
                                self.render(env, agent, episode+1, step, total_reward)
                                pygame.display.flip()
                                
                        elif event.type == pygame.KEYDOWN:
                            need_redraw = True
                            
                            if event.key in (pygame.K_q, pygame.K_ESCAPE):
                                if self.is_fullscreen:  # Exit fullscreen mode on ESC
                                    self._handle_button_click('fullscreen')
                                else:  # Quit on ESC when not in fullscreen
                                    self.running = False
                                    break
                            elif event.key == pygame.K_SPACE:
                                self.paused = not self.paused
                            elif event.key == pygame.K_v:
                                self.show_values = not self.show_values
                            elif event.key == pygame.K_p:
                                self.show_policy = not self.show_policy
                            elif event.key == pygame.K_t:
                                # Toggle test mode with T key
                                self.test_mode = not self.test_mode
                            elif event.key == pygame.K_f:
                                # Toggle fullscreen with F key
                                result = self._handle_button_click('fullscreen')
                                if result == 'resize':
                                    self._calculate_grid_dimensions(env)
                            elif event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS:
                                self.animation_speed = max(0.1, self.animation_speed - 0.1)
                            elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                                self.animation_speed = min(2.0, self.animation_speed + 0.1)
                            elif event.key == pygame.K_r:
                                # Restart with R key
                                obs = env.reset()
                                done = False
                                total_reward = 0
                                step = 0
                            else:
                                need_redraw = False
                                
                            # Force redraw when key actions change visualization
                            if need_redraw:
                                self.render(env, agent, episode+1, step, total_reward)
                                pygame.display.flip()
                    
                    if not self.running:
                        break
                    
                    # Always render when paused to ensure display stays responsive
                    if self.paused:
                        self.render(env, agent, episode+1, step, total_reward)
                        pygame.display.flip()  # Ensure full screen update while paused
                        pygame.time.wait(100)  # Short delay when paused to reduce CPU usage
                        continue
                        
                    try:
                        # Get action based on agent type
                        if hasattr(agent, 'act'):
                            action = agent.act(obs)
                        elif hasattr(agent, 'online_net'):  # Agent is DQN agent
                            flat_obs = obs if not isinstance(obs, tuple) else obs[0]
                            # Use direct inference on the model with epsilon-greedy policy
                            if random.random() < 0.01:  # Small exploration chance
                                action = random.randint(0, env.action_space.n - 1)
                            else:
                                with torch.no_grad():
                                    state_tensor = torch.FloatTensor(flat_obs).to(device)
                                    q_values = agent.online_net(state_tensor)
                                    action = torch.argmax(q_values).item()
                        elif isinstance(agent, torch.nn.Module):  # Agent is direct DQN model
                            flat_obs = obs if not isinstance(obs, tuple) else obs[0]
                            # Use direct inference on the model with epsilon-greedy policy
                            if random.random() < 0.01:  # Small exploration chance
                                action = random.randint(0, env.action_space.n - 1)
                            else:
                                with torch.no_grad():
                                    state_tensor = torch.FloatTensor(flat_obs).to(device)
                                    q_values = agent(state_tensor)
                                    action = torch.argmax(q_values).item()
                        else:
                            # Default random action if agent type can't be determined
                            action = env.action_space.sample()
                            print("Warning: Using random action because agent type couldn't be determined")
                        
                        # Take step in environment
                        obs, reward, done, _ = env.step(action)
                        total_reward += reward
                        
                        # Render with full screen update for each step
                        self.render(env, agent, episode + 1, step + 1, total_reward)
                        # pygame.display.flip() is called inside render()
                        
                        # Ensure the frame rate is consistent
                        pygame.time.wait(int(self.animation_speed * 1000))
                        
                        step += 1
                    except Exception as e:
                        print(f"Error during simulation step: {e}")
                        import traceback
                        traceback.print_exc()
                        # Continue with next step instead of crashing
                
                if self.running:
                    episode += 1
        finally:
            # Always make sure to clean up PyGame resources
            self.cleanup()
        
    def cleanup(self):
        """Clean up PyGame resources."""
        try:
            pygame.quit()
        except Exception as e:
            print(f"Warning: Error during PyGame cleanup: {e}")

    def _get_cell_rect(self, row: int, col: int) -> pygame.Rect:
        """Get the pygame Rect for a grid cell."""
        if self.cell_size is None or self.grid_offset is None:
            # Make sure grid dimensions are calculated first
            raise ValueError("Grid dimensions not calculated. Call _calculate_grid_dimensions first.")
            
        return pygame.Rect(
            int(self.grid_offset[0] + col * self.cell_size),
            int(self.grid_offset[1] + row * self.cell_size),
            int(self.cell_size),
            int(self.cell_size)
        )

def main():
    """Example usage of the GridWorld visualizer."""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from envs.gridworld import GridWorldEnv
    from agents.q_learning import QLearningAgent
    from config import DEFAULT_ENV_CONFIG, QL_AGENT_CONFIG
    
    # Create environment and agent
    env = GridWorldEnv(DEFAULT_ENV_CONFIG)
    agent = QLearningAgent(env.observation_space, env.action_space,
                          env.config["grid_size"], QL_AGENT_CONFIG)
    
    # Create and run visualizer
    vis = GridWorldVisualizer()
    vis.simulate(env, agent)

if __name__ == "__main__":
    main()