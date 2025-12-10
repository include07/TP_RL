"""
Interactive Reinforcement Learning Lab
Streamlit application for training and visualizing RL agents
"""
import streamlit as st
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.environment import GridEnvironment
from algorithms.value_iteration import ValueIterationSolver
from algorithms.q_learning import QLearner
from algorithms.random_agent import RandomAgent
from visualization.plots import (
    plot_grid_with_policy, plot_trajectory, create_training_dashboard,
    create_value_heatmap, plot_convergence, compare_agents
)
from settings import ENV_CONFIG, REWARDS, VI_PARAMS, QL_PARAMS

# Page configuration
st.set_page_config(
    page_title="RL Training Lab",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'environment' not in st.session_state:
    st.session_state.environment = None
if 'trained_agent' not in st.session_state:
    st.session_state.trained_agent = None
if 'agent_type' not in st.session_state:
    st.session_state.agent_type = None
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False


def main():
    st.markdown('<h1 class="main-header">🤖 Reinforcement Learning Training Lab</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Interactive platform for experimenting with reinforcement learning algorithms.
    Configure the grid environment, select an algorithm, and watch the agent learn!
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Environment settings
        st.subheader("Environment Setup")
        
        grid_size = st.slider("Grid Size", 4, 10, ENV_CONFIG['grid_dimensions'])
        
        # Start position
        st.write("**Start Position**")
        col1, col2 = st.columns(2)
        start_row = col1.number_input("Row", 0, grid_size-1, 0, key='start_row')
        start_col = col2.number_input("Col", 0, grid_size-1, 0, key='start_col')
        
        # Goal position
        st.write("**Goal Position**")
        col3, col4 = st.columns(2)
        goal_row = col3.number_input("Row", 0, grid_size-1, grid_size-1, key='goal_row')
        goal_col = col4.number_input("Col", 0, grid_size-1, grid_size-1, key='goal_col')
        
        # Barriers
        st.write("**Barriers (comma-separated pairs)**")
        barrier_input = st.text_input(
            "Format: row,col;row,col", 
            value=f"{min(2, grid_size-1)},{min(2, grid_size-1)};{min(3, grid_size-1)},{min(3, grid_size-1)}"
        )
        
        # Parse barriers
        barriers = []
        if barrier_input:
            try:
                for pair in barrier_input.split(';'):
                    if pair.strip():
                        r, c = map(int, pair.split(','))
                        if 0 <= r < grid_size and 0 <= c < grid_size:
                            barriers.append((r, c))
            except:
                st.warning("Invalid barrier format")
        
        # Reward structure
        st.subheader("Reward Settings")
        target_reward = st.number_input("Goal Reward", 1.0, 50.0, REWARDS['target_reached'])
        barrier_penalty = st.number_input("Barrier Penalty", -20.0, -1.0, REWARDS['barrier_hit'])
        step_cost = st.number_input("Step Cost", -1.0, 0.0, REWARDS['movement_cost'])
        
        # Create environment button
        if st.button("🌍 Create Environment", use_container_width=True):
            rewards = {
                'target_reached': target_reward,
                'barrier_hit': barrier_penalty,
                'movement_cost': step_cost,
                'boundary_penalty': -1.0
            }
            
            st.session_state.environment = GridEnvironment(
                grid_size=grid_size,
                targets=[(goal_row, goal_col)],
                barriers=barriers,
                start_pos=(start_row, start_col),
                rewards=rewards
            )
            st.session_state.training_complete = False
            st.success("✅ Environment created!")
        
        st.divider()
        
        # Algorithm selection
        st.subheader("Algorithm Selection")
        algorithm = st.selectbox(
            "Choose Algorithm",
            ["Value Iteration", "Q-Learning", "Random Baseline"]
        )
        
        # Algorithm-specific parameters
        if algorithm == "Value Iteration":
            st.write("**VI Parameters**")
            gamma_vi = st.slider("Discount Factor (γ)", 0.0, 1.0, VI_PARAMS['gamma'])
            max_iter = st.number_input("Max Iterations", 100, 5000, VI_PARAMS['max_iter'])
            
        elif algorithm == "Q-Learning":
            st.write("**Q-Learning Parameters**")
            alpha = st.slider("Learning Rate (α)", 0.01, 1.0, QL_PARAMS['alpha'])
            gamma_ql = st.slider("Discount Factor (γ)", 0.0, 1.0, QL_PARAMS['gamma'])
            epsilon_start = st.slider("Initial Exploration (ε)", 0.5, 1.0, QL_PARAMS['epsilon_start'])
            episodes = st.number_input("Training Episodes", 100, 2000, QL_PARAMS['episodes'])
        
        # Train button
        if st.button("🚀 Train Agent", use_container_width=True):
            if st.session_state.environment is None:
                st.error("❌ Create environment first!")
            else:
                with st.spinner("Training in progress..."):
                    env = st.session_state.environment
                    
                    if algorithm == "Value Iteration":
                        agent = ValueIterationSolver(env, gamma=gamma_vi, max_iterations=max_iter)
                        agent.solve()
                        
                    elif algorithm == "Q-Learning":
                        agent = QLearner(env, alpha=alpha, gamma=gamma_ql, 
                                       epsilon_start=epsilon_start)
                        agent.learn(num_episodes=episodes)
                        
                    else:  # Random
                        agent = RandomAgent(env)
                    
                    st.session_state.trained_agent = agent
                    st.session_state.agent_type = algorithm
                    st.session_state.training_complete = True
                    
                st.success("✅ Training complete!")
    
    # Main content area
    if st.session_state.environment is not None:
        env = st.session_state.environment
        
        # Display environment
        st.markdown('<p class="section-header">📍 Environment Overview</p>', 
                   unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Grid Size", f"{env.size}×{env.size}")
        col2.metric("Barriers", len(env.barriers))
        col3.metric("Max Steps", env.max_steps)
        
        # Visualize grid
        st.pyplot(plot_grid_with_policy(env, title="Environment Layout"))
        
        # Show training results
        if st.session_state.training_complete and st.session_state.trained_agent is not None:
            agent = st.session_state.trained_agent
            agent_name = st.session_state.agent_type
            
            st.markdown(f'<p class="section-header">🎯 {agent_name} Results</p>', 
                       unsafe_allow_html=True)
            
            # Algorithm-specific visualizations
            if agent_name == "Value Iteration":
                tab1, tab2, tab3 = st.tabs(["📊 Policy", "🔥 Values", "📈 Convergence"])
                
                with tab1:
                    st.subheader("Learned Policy")
                    policy = agent.get_policy()
                    st.pyplot(plot_grid_with_policy(env, policy=policy, 
                             title="Optimal Policy (Value Iteration)"))
                
                with tab2:
                    st.subheader("State Values")
                    values = agent.get_state_values()
                    st.plotly_chart(create_value_heatmap(env, values, 
                                   "State Value Function"), use_container_width=True)
                
                with tab3:
                    st.subheader("Convergence History")
                    st.plotly_chart(plot_convergence(agent.iteration_history, 
                                   "Value Iteration Convergence"), use_container_width=True)
                
                # Evaluation
                st.markdown('<p class="section-header">📊 Policy Evaluation</p>', 
                           unsafe_allow_html=True)
                
                if st.button("🧪 Test Policy (50 episodes)"):
                    with st.spinner("Testing..."):
                        results = agent.evaluate_policy(50)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Success Rate", f"{results['success_rate']:.1%}")
                    col2.metric("Avg Reward", f"{results['avg_reward']:.2f}")
                    col3.metric("Avg Steps", f"{results['avg_steps']:.1f}")
            
            elif agent_name == "Q-Learning":
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Training", "📊 Policy", "🔥 Q-Values", "🎬 Demo"])
                
                with tab1:
                    st.subheader("Training Progress")
                    metrics = agent.get_training_metrics()
                    st.plotly_chart(create_training_dashboard(metrics, 
                                   "Q-Learning Training Metrics"), use_container_width=True)
                
                with tab2:
                    st.subheader("Learned Policy")
                    policy = agent.get_policy()
                    values = agent.get_state_values()
                    st.pyplot(plot_grid_with_policy(env, policy=policy, values=values,
                             title="Learned Policy (Q-Learning)"))
                
                with tab3:
                    st.subheader("Q-Values")
                    q_values = agent.get_q_values()
                    state_values = np.max(q_values, axis=1)
                    st.plotly_chart(create_value_heatmap(env, state_values, 
                                   "Maximum Q-Values per State"), use_container_width=True)
                
                with tab4:
                    st.subheader("Test Episode")
                    if st.button("▶️ Run Demo Episode"):
                        with st.spinner("Running..."):
                            state = env.reset()
                            steps = 0
                            
                            while steps < env.max_steps:
                                action = agent.choose_action(state, explore=False)
                                state, reward, done, info = env.execute_action(action)
                                steps += 1
                                if done:
                                    break
                            
                            trajectory = env.get_trajectory()
                            st.pyplot(plot_trajectory(env, trajectory, 
                                     f"Episode: {steps} steps, Success: {info['success']}"))
                            
                            if info['success']:
                                st.success(f"✅ Goal reached in {steps} steps!")
                            else:
                                st.warning(f"⏱️ Episode ended after {steps} steps")
            
            else:  # Random Agent
                st.markdown('<p class="section-header">📊 Random Agent Performance</p>', 
                           unsafe_allow_html=True)
                
                if st.button("🎲 Test Random Agent (100 episodes)"):
                    with st.spinner("Running random trials..."):
                        results = agent.evaluate_performance(100)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Success Rate", f"{results['success_rate']:.1%}")
                    col2.metric("Avg Reward", f"{results['avg_reward']:.2f}")
                    col3.metric("Avg Steps", f"{results['avg_steps']:.1f}")
                    
                    # Show a sample trajectory
                    if results['trajectories']:
                        st.pyplot(plot_trajectory(env, results['trajectories'][0], 
                                 "Sample Random Episode"))
    
    else:
        st.info("👈 Configure and create an environment in the sidebar to begin")
        
        # Show example
        st.markdown('<p class="section-header">💡 Quick Start Guide</p>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        1. **Configure Environment**: Set grid size, start/goal positions, and barriers
        2. **Set Rewards**: Define reward structure for goals, barriers, and steps
        3. **Create Environment**: Click "Create Environment" button
        4. **Choose Algorithm**: Select Value Iteration, Q-Learning, or Random baseline
        5. **Train Agent**: Click "Train Agent" and watch it learn!
        6. **Analyze Results**: Explore visualizations and test the learned policy
        """)


if __name__ == "__main__":
    main()
