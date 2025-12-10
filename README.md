# Reinforcement Learning Training Lab

An interactive web application for experimenting with reinforcement learning algorithms in a grid-based environment. Built with Streamlit for real-time configuration, training, and visualization.

## 🎯 Features

- **Interactive Configuration**: Customize grid size, obstacles, rewards, and starting positions
- **Multiple Algorithms**: 
  - Value Iteration (Dynamic Programming)
  - Q-Learning (Temporal Difference)
  - Random Baseline (for comparison)
- **Real-time Training**: Watch agents learn and visualize convergence
- **Rich Visualizations**:
  - Policy arrows showing learned behavior
  - State value heatmaps
  - Training curves and metrics
  - Episode trajectories
  - Algorithm comparisons
- **Streamlit Interface**: No coding required - configure everything through the UI

## 📁 Project Structure

```
TP_RL/
├── app.py                      # Main Streamlit application
├── settings.py                 # Configuration parameters
├── requirements.txt            # Python dependencies
├── core/
│   └── environment.py          # Grid environment implementation
├── algorithms/
│   ├── value_iteration.py      # Value Iteration solver
│   ├── q_learning.py           # Q-Learning agent
│   └── random_agent.py         # Random baseline agent
└── visualization/
    └── plots.py                # Plotting utilities
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/include07/TP_RL.git
cd TP_RL

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. Environment Configuration

In the sidebar, configure your grid world:
- **Grid Size**: Choose dimensions (4x4 to 10x10)
- **Start Position**: Set agent's initial location
- **Goal Position**: Define target location
- **Barriers**: Add obstacles using format `row,col;row,col`
- **Rewards**: Customize reward values for goals, barriers, and movement

Click **"Create Environment"** to initialize.

### 2. Algorithm Selection

Choose from three algorithms:

#### Value Iteration
- Model-based planning algorithm
- Requires full environment knowledge
- Computes optimal policy through iterative updates
- Parameters:
  - **Discount Factor (γ)**: Future reward importance (0-1)
  - **Max Iterations**: Convergence limit

#### Q-Learning
- Model-free learning algorithm
- Learns through interaction
- Balances exploration vs exploitation
- Parameters:
  - **Learning Rate (α)**: Update step size
  - **Discount Factor (γ)**: Future reward importance
  - **Exploration Rate (ε)**: Probability of random actions
  - **Episodes**: Number of training episodes

#### Random Baseline
- Selects random actions
- Useful for performance comparison
- No training required

### 3. Training

Click **"Train Agent"** to start training. The application will:
- Initialize the selected algorithm
- Run training iterations/episodes
- Display progress in real-time

### 4. Results Analysis

Explore multiple visualization tabs:

**Value Iteration:**
- Policy visualization with directional arrows
- State value heatmaps
- Convergence plots
- Policy evaluation metrics

**Q-Learning:**
- Training progress dashboard (rewards, steps, exploration)
- Learned policy visualization
- Q-value heatmaps
- Interactive demo episodes

**Random Agent:**
- Performance statistics
- Sample trajectory visualization

## 🎮 Example Scenarios

### Simple Navigation
```
Grid: 5x5
Start: (0, 0)
Goal: (4, 4)
Barriers: None
```

### Maze Challenge
```
Grid: 6x6
Start: (0, 0)
Goal: (5, 5)
Barriers: (2,2), (3,3), (1,4), (4,1)
```

### Narrow Path
```
Grid: 8x8
Start: (0, 0)
Goal: (7, 7)
Barriers: Create walls forcing specific path
```

## 🔧 Customization

### Modifying Default Settings

Edit `settings.py` to change defaults:

```python
ENV_CONFIG = {
    'grid_dimensions': 6,
    'start_location': (0, 0),
    'target_locations': [(5, 5)],
    'barrier_locations': [(2, 2), (3, 3)],
    'max_episode_steps': 150
}

REWARDS = {
    'target_reached': 15.0,
    'barrier_hit': -8.0,
    'movement_cost': -0.2,
    'boundary_penalty': -1.0
}
```

### Adding New Environments

Extend `GridEnvironment` class in `core/environment.py`:

```python
class CustomEnvironment(GridEnvironment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add custom logic
```

### Creating New Algorithms

Implement new agents in `algorithms/` directory following the base structure:

```python
class NewAgent:
    def __init__(self, environment):
        self.env = environment
    
    def choose_action(self, state):
        # Action selection logic
        pass
    
    def learn(self):
        # Training logic
        pass
```

## 📊 Understanding the Algorithms

### Value Iteration

Uses Bellman optimality equation:
```
V(s) = max_a [R(s,a) + γ * V(s')]
```

Iteratively updates state values until convergence. Guaranteed to find optimal policy for known MDPs.

### Q-Learning

Learns action-value function:
```
Q(s,a) ← Q(s,a) + α[r + γ * max_a' Q(s',a') - Q(s,a)]
```

Explores environment using ε-greedy strategy. Converges to optimal Q-function with sufficient exploration.

## 🎨 Visualization Features

- **Grid Display**: Shows current environment layout
- **Policy Arrows**: Directional indicators for learned behavior
- **Value Heatmaps**: Color-coded state values
- **Training Curves**: Real-time learning progress
- **Success Metrics**: Performance statistics
- **Trajectory Plots**: Agent path visualization

## ⚡ Performance Tips

- Start with smaller grids (5x5 or 6x6) for faster training
- Reduce max iterations/episodes for quick experiments
- Use Value Iteration for small, fully-known environments
- Use Q-Learning for larger or partially observable scenarios
- Adjust learning rate if Q-Learning converges too slowly
- Increase exploration if agent gets stuck in local optima

## 🐛 Troubleshooting

**Training takes too long:**
- Reduce grid size or number of episodes
- Lower max iterations for Value Iteration
- Simplify environment (fewer barriers)

**Agent doesn't reach goal:**
- Check if goal is reachable (not blocked)
- Increase training episodes for Q-Learning
- Adjust reward structure (higher goal reward)
- Verify barrier positions don't block path

**Convergence issues:**
- Increase discount factor (γ)
- Adjust learning rate (α) for Q-Learning
- Check reward values are appropriate

## 📚 References

- Sutton & Barto: "Reinforcement Learning: An Introduction"
- Bellman, R.: "Dynamic Programming"
- Watkins, C.J.: "Q-Learning"

## 🤝 Contributing

This is an educational project. Feel free to:
- Experiment with different configurations
- Add new algorithms
- Enhance visualizations
- Improve documentation

## 📝 License

Educational project for learning reinforcement learning concepts.

## 👤 Author

Developed as part of TP_RL coursework.

---

**Happy Learning! 🎓🤖**
