"""
Environment and training settings for GridWorld RL
"""

# Environment Configuration
ENV_CONFIG = {
    'grid_dimensions': 6,
    'start_location': (0, 0),
    'target_locations': [(5, 5)],
    'barrier_locations': [(2, 2), (3, 3), (1, 4)],
    'max_episode_steps': 150
}

# Reward Structure
REWARDS = {
    'target_reached': 15.0,
    'barrier_hit': -8.0,
    'movement_cost': -0.2,
    'boundary_penalty': -1.0
}

# Value Iteration Settings
VI_PARAMS = {
    'gamma': 0.95,
    'theta': 1e-5,
    'max_iter': 2000
}

# Q-Learning Configuration
QL_PARAMS = {
    'alpha': 0.15,
    'gamma': 0.95,
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,
    'epsilon_decay': 0.997,
    'episodes': 800
}

# Direction mappings
DIRECTIONS = {
    'north': (-1, 0),
    'south': (1, 0),
    'west': (0, -1),
    'east': (0, 1)
}

DIRECTION_SYMBOLS = {
    'north': '↑',
    'south': '↓',
    'west': '←',
    'east': '→'
}
