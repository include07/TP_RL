"""
Random Agent - Baseline for comparison
"""
import numpy as np


class RandomAgent:
    """Agent that selects random actions"""
    
    def __init__(self, environment):
        """
        Initialize random agent
        
        Parameters:
            environment: GridEnvironment instance
        """
        self.env = environment
        self.trained = False  # Random agent doesn't need training
    
    def choose_action(self, state):
        """Select random action"""
        return np.random.randint(self.env.num_actions)
    
    def evaluate_performance(self, num_episodes=100):
        """Test random policy"""
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
                action = self.choose_action(state)
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
