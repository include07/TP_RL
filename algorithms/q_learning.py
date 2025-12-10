"""
Q-Learning - Temporal Difference Algorithm
"""
import numpy as np
from settings import QL_PARAMS


class QLearner:
    """Q-Learning agent with epsilon-greedy exploration"""
    
    def __init__(self, environment, alpha=None, gamma=None, 
                 epsilon_start=None, epsilon_end=None, epsilon_decay=None):
        """
        Initialize Q-Learning agent
        
        Parameters:
            environment: GridEnvironment instance
            alpha: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Exploration decay rate
        """
        self.env = environment
        self.alpha = alpha or QL_PARAMS['alpha']
        self.gamma = gamma or QL_PARAMS['gamma']
        self.epsilon = epsilon_start or QL_PARAMS['epsilon_start']
        self.epsilon_min = epsilon_end or QL_PARAMS['epsilon_end']
        self.decay_rate = epsilon_decay or QL_PARAMS['epsilon_decay']
        
        # Initialize Q-table
        self.q_table = np.zeros((self.env.num_states, self.env.num_actions))
        
        # Training metrics
        self.training_rewards = []
        self.training_steps = []
        self.epsilon_values = []
        self.success_episodes = []
        self.q_snapshots = []
        
        self.trained = False
    
    def choose_action(self, state, explore=True):
        """
        Select action using epsilon-greedy strategy
        
        Parameters:
            state: Current state
            explore: Whether to use exploration
        """
        if explore and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.env.num_actions)
        else:
            # Exploit: best known action
            return np.argmax(self.q_table[state])
    
    def learn(self, num_episodes=None):
        """
        Train agent using Q-learning
        
        Parameters:
            num_episodes: Number of training episodes
        """
        episodes = num_episodes or QL_PARAMS['episodes']
        print(f"Training Q-Learner for {episodes} episodes...")
        
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            steps = 0
            success = False
            
            while steps < self.env.max_steps:
                # Choose action
                action = self.choose_action(state, explore=True)
                
                # Execute action
                next_state, reward, done, info = self.env.execute_action(action)
                
                # Q-learning update
                self._update_q_value(state, action, reward, next_state, done)
                
                episode_reward += reward
                state = next_state
                steps += 1
                
                if done:
                    success = info['success']
                    break
            
            # Decay exploration
            self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_rate)
            
            # Record metrics
            self.training_rewards.append(episode_reward)
            self.training_steps.append(steps)
            self.epsilon_values.append(self.epsilon)
            self.success_episodes.append(1 if success else 0)
            
            # Snapshot Q-values periodically
            if episode % 100 == 0:
                self.q_snapshots.append(np.max(self.q_table, axis=1).copy())
                success_rate = np.mean(self.success_episodes[-100:]) if len(self.success_episodes) >= 100 else 0
                print(f"  Episode {episode}: reward={episode_reward:.2f}, steps={steps}, "
                      f"epsilon={self.epsilon:.3f}, success_rate={success_rate:.2%}")
        
        self.trained = True
        print("Training complete!")
    
    def _update_q_value(self, state, action, reward, next_state, done):
        """Apply Q-learning update rule"""
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        
        # TD update
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
    
    def get_q_values(self):
        """Return Q-table"""
        return self.q_table.copy()
    
    def get_state_values(self):
        """Extract state values from Q-table"""
        return np.max(self.q_table, axis=1)
    
    def get_policy(self):
        """Extract greedy policy from Q-table"""
        return np.argmax(self.q_table, axis=1)
    
    def evaluate_performance(self, num_episodes=50):
        """Test learned policy without exploration"""
        results = {
            'success_count': 0,
            'total_rewards': [],
            'episode_lengths': [],
            'trajectories': []
        }
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            steps = 0
            
            while steps < self.env.max_steps:
                action = self.choose_action(state, explore=False)
                state, reward, done, info = self.env.execute_action(action)
                
                episode_reward += reward
                steps += 1
                
                if done:
                    if info['success']:
                        results['success_count'] += 1
                    break
            
            results['total_rewards'].append(episode_reward)
            results['episode_lengths'].append(steps)
            results['trajectories'].append(self.env.get_trajectory())
        
        results['success_rate'] = results['success_count'] / num_episodes
        results['avg_reward'] = np.mean(results['total_rewards'])
        results['avg_steps'] = np.mean(results['episode_lengths'])
        
        return results
    
    def get_training_metrics(self):
        """Return training history"""
        return {
            'rewards': self.training_rewards.copy(),
            'steps': self.training_steps.copy(),
            'epsilon': self.epsilon_values.copy(),
            'success': self.success_episodes.copy(),
            'q_snapshots': self.q_snapshots.copy()
        }
