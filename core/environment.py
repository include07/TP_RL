"""
GridWorld Environment - Custom implementation
"""
import numpy as np
from settings import ENV_CONFIG, REWARDS, DIRECTIONS


class GridEnvironment:
    """Custom grid-based environment for RL experiments"""
    
    def __init__(self, grid_size=None, targets=None, barriers=None, 
                 start_pos=None, rewards=None, max_steps=None):
        """
        Initialize grid environment
        
        Parameters:
            grid_size: Dimension of square grid
            targets: List of goal coordinates
            barriers: List of obstacle coordinates
            start_pos: Initial agent position
            rewards: Dictionary of reward values
            max_steps: Maximum steps per episode
        """
        self.size = grid_size or ENV_CONFIG['grid_dimensions']
        self.targets = targets or ENV_CONFIG['target_locations']
        self.barriers = barriers or ENV_CONFIG['barrier_locations']
        self.initial_position = start_pos or ENV_CONFIG['start_location']
        self.reward_structure = rewards or REWARDS
        self.max_steps = max_steps or ENV_CONFIG['max_episode_steps']
        
        self.num_actions = 4
        self.num_states = self.size * self.size
        
        self.current_position = None
        self.step_count = 0
        self.episode_history = []
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_position = list(self.initial_position)
        self.step_count = 0
        self.episode_history = [tuple(self.current_position)]
        return self._position_to_state(self.current_position)
    
    def execute_action(self, action_idx):
        """
        Execute action and return transition information
        
        Returns:
            next_state: New state after action
            reward: Reward received
            done: Episode termination flag
            info: Additional data
        """
        self.step_count += 1
        
        # Map action to direction
        direction_keys = list(DIRECTIONS.keys())
        direction = DIRECTIONS[direction_keys[action_idx]]
        
        # Calculate new position
        new_row = self.current_position[0] + direction[0]
        new_col = self.current_position[1] + direction[1]
        
        # Check boundaries
        if self._is_valid_position(new_row, new_col):
            self.current_position = [new_row, new_col]
            reward = self.reward_structure['movement_cost']
        else:
            # Hit boundary - stay in place with penalty
            reward = self.reward_structure['boundary_penalty']
        
        # Check terminal conditions
        done = False
        pos_tuple = tuple(self.current_position)
        
        if pos_tuple in self.targets:
            reward = self.reward_structure['target_reached']
            done = True
        elif pos_tuple in self.barriers:
            reward = self.reward_structure['barrier_hit']
            done = True
        
        # Max steps reached
        if self.step_count >= self.max_steps:
            done = True
        
        self.episode_history.append(pos_tuple)
        next_state = self._position_to_state(self.current_position)
        
        info = {
            'steps': self.step_count,
            'position': pos_tuple,
            'success': pos_tuple in self.targets
        }
        
        return next_state, reward, done, info
    
    def _is_valid_position(self, row, col):
        """Check if position is within grid boundaries"""
        return 0 <= row < self.size and 0 <= col < self.size
    
    def _position_to_state(self, position):
        """Convert 2D position to 1D state index"""
        return position[0] * self.size + position[1]
    
    def _state_to_position(self, state):
        """Convert 1D state index to 2D position"""
        row = state // self.size
        col = state % self.size
        return (row, col)
    
    def get_valid_states(self):
        """Return all valid state indices"""
        return list(range(self.num_states))
    
    def compute_transition(self, state, action_idx):
        """
        Compute transition for planning algorithms
        
        Returns:
            next_state: Resulting state
            reward: Expected reward
            terminal: Whether state is terminal
        """
        position = list(self._state_to_position(state))
        
        # Get direction
        direction_keys = list(DIRECTIONS.keys())
        direction = DIRECTIONS[direction_keys[action_idx]]
        
        # Calculate next position
        next_row = position[0] + direction[0]
        next_col = position[1] + direction[1]
        
        # Handle boundaries
        if self._is_valid_position(next_row, next_col):
            next_position = (next_row, next_col)
            reward = self.reward_structure['movement_cost']
        else:
            next_position = tuple(position)
            reward = self.reward_structure['boundary_penalty']
        
        # Check terminal conditions
        terminal = False
        if next_position in self.targets:
            reward = self.reward_structure['target_reached']
            terminal = True
        elif next_position in self.barriers:
            reward = self.reward_structure['barrier_hit']
            terminal = True
        
        next_state = self._position_to_state(next_position)
        return next_state, reward, terminal
    
    def get_trajectory(self):
        """Return episode trajectory"""
        return self.episode_history.copy()
    
    def render_grid(self):
        """Create text representation of grid"""
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        
        # Mark barriers
        for barrier in self.barriers:
            grid[barrier[0]][barrier[1]] = 'X'
        
        # Mark targets
        for target in self.targets:
            grid[target[0]][target[1]] = 'G'
        
        # Mark agent
        if self.current_position:
            grid[self.current_position[0]][self.current_position[1]] = 'A'
        
        # Print grid
        for row in grid:
            print(' '.join(row))
        print()
