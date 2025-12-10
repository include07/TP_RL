"""
Visualization utilities for grid environment and agent performance
"""
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from settings import DIRECTION_SYMBOLS, DIRECTIONS


def plot_grid_with_policy(env, policy=None, values=None, title="Grid World"):
    """
    Create matplotlib visualization of grid with optional policy arrows
    
    Parameters:
        env: GridEnvironment instance
        policy: Optional policy array
        values: Optional value function
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create grid
    for i in range(env.size + 1):
        ax.axhline(i, color='black', linewidth=1)
        ax.axvline(i, color='black', linewidth=1)
    
    # Color cells based on values if provided
    if values is not None:
        value_grid = values.reshape(env.size, env.size)
        im = ax.imshow(value_grid, cmap='coolwarm', alpha=0.6, 
                      extent=[0, env.size, env.size, 0])
        plt.colorbar(im, ax=ax, label='State Value')
    
    # Mark special cells
    for row in range(env.size):
        for col in range(env.size):
            pos = (row, col)
            
            if pos in env.targets:
                ax.add_patch(plt.Rectangle((col, row), 1, 1, 
                           facecolor='green', alpha=0.7))
                ax.text(col + 0.5, row + 0.5, 'G', 
                       ha='center', va='center', fontsize=20, fontweight='bold')
            
            elif pos in env.barriers:
                ax.add_patch(plt.Rectangle((col, row), 1, 1, 
                           facecolor='red', alpha=0.7))
                ax.text(col + 0.5, row + 0.5, 'X', 
                       ha='center', va='center', fontsize=20, fontweight='bold')
            
            elif pos == tuple(env.initial_position):
                ax.add_patch(plt.Rectangle((col, row), 1, 1, 
                           facecolor='blue', alpha=0.4))
                ax.text(col + 0.5, row + 0.5, 'S', 
                       ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Draw policy arrows
    if policy is not None:
        direction_keys = list(DIRECTIONS.keys())
        for row in range(env.size):
            for col in range(env.size):
                pos = (row, col)
                if pos not in env.targets and pos not in env.barriers:
                    state = row * env.size + col
                    action = policy[state]
                    symbol = DIRECTION_SYMBOLS[direction_keys[action]]
                    ax.text(col + 0.5, row + 0.5, symbol, 
                           ha='center', va='center', fontsize=16)
    
    ax.set_xlim(0, env.size)
    ax.set_ylim(env.size, 0)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    
    return fig


def plot_trajectory(env, trajectory, title="Episode Trajectory"):
    """
    Visualize agent's path through grid
    
    Parameters:
        env: GridEnvironment instance
        trajectory: List of (row, col) positions
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw grid
    for i in range(env.size + 1):
        ax.axhline(i, color='gray', linewidth=0.5)
        ax.axvline(i, color='gray', linewidth=0.5)
    
    # Mark special cells
    for pos in env.targets:
        ax.add_patch(plt.Rectangle((pos[1], pos[0]), 1, 1, 
                   facecolor='green', alpha=0.5))
    
    for pos in env.barriers:
        ax.add_patch(plt.Rectangle((pos[1], pos[0]), 1, 1, 
                   facecolor='red', alpha=0.5))
    
    # Draw trajectory
    if trajectory:
        rows, cols = zip(*trajectory)
        rows = [r + 0.5 for r in rows]
        cols = [c + 0.5 for c in cols]
        
        # Plot path
        ax.plot(cols, rows, 'b-', linewidth=2, alpha=0.7, marker='o', 
               markersize=6, markerfacecolor='blue')
        
        # Mark start and end
        ax.plot(cols[0], rows[0], 'go', markersize=15, label='Start')
        ax.plot(cols[-1], rows[-1], 'ro', markersize=15, label='End')
    
    ax.set_xlim(0, env.size)
    ax.set_ylim(env.size, 0)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    
    return fig


def create_training_dashboard(metrics, title="Training Progress"):
    """
    Create interactive plotly dashboard for training metrics
    
    Parameters:
        metrics: Dictionary with training data
        title: Dashboard title
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Episode Rewards', 'Episode Lengths', 
                       'Exploration Rate', 'Success Rate (Moving Avg)'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    episodes = list(range(len(metrics['rewards'])))
    
    # Rewards
    fig.add_trace(
        go.Scatter(x=episodes, y=metrics['rewards'], 
                  mode='lines', name='Reward',
                  line=dict(color='royalblue', width=1)),
        row=1, col=1
    )
    
    # Steps
    fig.add_trace(
        go.Scatter(x=episodes, y=metrics['steps'], 
                  mode='lines', name='Steps',
                  line=dict(color='orange', width=1)),
        row=1, col=2
    )
    
    # Epsilon
    if 'epsilon' in metrics:
        fig.add_trace(
            go.Scatter(x=episodes, y=metrics['epsilon'], 
                      mode='lines', name='Epsilon',
                      line=dict(color='green', width=2)),
            row=2, col=1
        )
    
    # Success rate (moving average)
    if 'success' in metrics:
        window = 50
        success_rate = [np.mean(metrics['success'][max(0, i-window):i+1]) 
                       for i in range(len(metrics['success']))]
        fig.add_trace(
            go.Scatter(x=episodes, y=success_rate, 
                      mode='lines', name='Success Rate',
                      line=dict(color='red', width=2)),
            row=2, col=2
        )
    
    fig.update_xaxes(title_text="Episode", row=1, col=1)
    fig.update_xaxes(title_text="Episode", row=1, col=2)
    fig.update_xaxes(title_text="Episode", row=2, col=1)
    fig.update_xaxes(title_text="Episode", row=2, col=2)
    
    fig.update_yaxes(title_text="Reward", row=1, col=1)
    fig.update_yaxes(title_text="Steps", row=1, col=2)
    fig.update_yaxes(title_text="Epsilon", row=2, col=1)
    fig.update_yaxes(title_text="Success Rate", row=2, col=2)
    
    fig.update_layout(
        title_text=title,
        showlegend=False,
        height=700
    )
    
    return fig


def create_value_heatmap(env, values, title="State Value Heatmap"):
    """
    Create interactive heatmap of state values
    
    Parameters:
        env: GridEnvironment instance
        values: Value function array
        title: Heatmap title
    """
    value_grid = values.reshape(env.size, env.size)
    
    # Create annotations for special cells
    annotations = []
    for row in range(env.size):
        for col in range(env.size):
            pos = (row, col)
            text = ""
            if pos in env.targets:
                text = "GOAL"
            elif pos in env.barriers:
                text = "BLOCK"
            
            if text:
                annotations.append(
                    dict(x=col, y=row, text=text, showarrow=False,
                        font=dict(color='white', size=12, family='Arial Black'))
                )
    
    fig = go.Figure(data=go.Heatmap(
        z=value_grid,
        colorscale='Viridis',
        text=np.round(value_grid, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Value")
    ))
    
    fig.update_layout(
        title=title,
        xaxis=dict(title="Column", dtick=1),
        yaxis=dict(title="Row", dtick=1, autorange='reversed'),
        width=600,
        height=600
    )
    
    fig.update_layout(annotations=annotations)
    
    return fig


def plot_convergence(history, title="Convergence Plot"):
    """
    Plot convergence history for iterative algorithms
    
    Parameters:
        history: List of delta values per iteration
        title: Plot title
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(history))),
        y=history,
        mode='lines',
        line=dict(color='purple', width=2),
        name='Max Change'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Iteration",
        yaxis_title="Maximum Value Change",
        yaxis_type="log",
        height=400
    )
    
    return fig


def compare_agents(results_dict):
    """
    Create comparison visualization for multiple agents
    
    Parameters:
        results_dict: Dictionary mapping agent names to results
    """
    agents = list(results_dict.keys())
    metrics = ['success_rate', 'avg_reward', 'avg_steps']
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Success Rate', 'Average Reward', 'Average Steps')
    )
    
    for idx, metric in enumerate(metrics):
        values = [results_dict[agent][metric] for agent in agents]
        
        fig.add_trace(
            go.Bar(x=agents, y=values, name=metric,
                  marker_color=['blue', 'orange', 'green'][:len(agents)]),
            row=1, col=idx+1
        )
    
    fig.update_layout(
        title_text="Agent Performance Comparison",
        showlegend=False,
        height=400
    )
    
    return fig
