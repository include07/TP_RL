"""
Value Iteration - Dynamic Programming Algorithm
"""
import numpy as np
from settings import VI_PARAMS


class ValueIterationSolver:
    """Solves MDP using value iteration algorithm"""
    
    def __init__(self, environment, gamma=None, theta=None, max_iterations=None):
        """
        Initialize solver
        
        Parameters:
            environment: GridEnvironment instance
            gamma: Discount factor
            theta: Convergence threshold
            max_iterations: Maximum iterations
        """
        self.env = environment
        self.gamma = gamma or VI_PARAMS['gamma']
        self.theta = theta or VI_PARAMS['theta']
        self.max_iterations = max_iterations or VI_PARAMS['max_iter']
        
        self.value_function = np.zeros(self.env.num_states)
        self.optimal_policy = np.zeros(self.env.num_states, dtype=int)
        
        self.iteration_history = []
        self.value_snapshots = []
        self.trained = False
    
    def solve(self):
        """Run value iteration to convergence"""
        print(f"Starting Value Iteration (max {self.max_iterations} iterations)...")
        
        for iteration in range(self.max_iterations):
            previous_values = self.value_function.copy()
            max_change = 0
            
            # Update each state
            for state in self.env.get_valid_states():
                action_values = self._compute_action_values(state, previous_values)
                
                # Bellman optimality backup
                self.value_function[state] = np.max(action_values)
                self.optimal_policy[state] = np.argmax(action_values)
                
                # Track maximum change
                change = abs(self.value_function[state] - previous_values[state])
                max_change = max(max_change, change)
            
            # Record progress
            self.iteration_history.append(max_change)
            
            if iteration % 50 == 0:
                self.value_snapshots.append(self.value_function.copy())
                print(f"  Iteration {iteration}: max change = {max_change:.6f}")
            
            # Check convergence
            if max_change < self.theta:
                print(f"Converged after {iteration + 1} iterations!")
                self.trained = True
                break
        
        if not self.trained:
            print(f"Reached max iterations ({self.max_iterations})")
            self.trained = True
        
        return self.value_function, self.optimal_policy
    
    def _compute_action_values(self, state, values):
        """Calculate Q-values for all actions in given state"""
        action_values = np.zeros(self.env.num_actions)
        
        for action in range(self.env.num_actions):
            next_state, reward, terminal = self.env.compute_transition(state, action)
            
            if terminal:
                action_values[action] = reward
            else:
                action_values[action] = reward + self.gamma * values[next_state]
        
        return action_values
    
    def select_action(self, state):
        """Choose action according to optimal policy"""
        if not self.trained:
            raise ValueError("Solver must be trained before selecting actions")
        return self.optimal_policy[state]
    
    def get_state_values(self):
        """Return value function"""
        return self.value_function.copy()
    
    def get_policy(self):
        """Return optimal policy"""
        return self.optimal_policy.copy()
    
    def evaluate_policy(self, num_episodes=50):
        """Test learned policy"""
        results = {
            'success_count': 0,
            'total_rewards': [],
            'episode_lengths': []
        }
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            steps = 0
            
            while steps < self.env.max_steps:
                action = self.select_action(state)
                state, reward, done, info = self.env.execute_action(action)
                
                episode_reward += reward
                steps += 1
                
                if done:
                    if info['success']:
                        results['success_count'] += 1
                    break
            
            results['total_rewards'].append(episode_reward)
            results['episode_lengths'].append(steps)
        
        results['success_rate'] = results['success_count'] / num_episodes
        results['avg_reward'] = np.mean(results['total_rewards'])
        results['avg_steps'] = np.mean(results['episode_lengths'])
        
        return results
