"""Gridworld MDP - Policy Iteration Algorithm

Gridworld is a classic reinforcement learning environment where an agent navigates
a 2D grid to reach goal states while avoiding obstacles.

Environment Description:
- A 2D grid with rows and columns
- Deterministic transitions (no randomness in movement)
- Terminal states with fixed rewards (positive for goals, negative for obstacles)
- Living cost (step cost) for each non-terminal state
- Four actions: UP, DOWN, LEFT, RIGHT

Algorithm: Policy Iteration
- Policy Evaluation: Compute value function for current policy
- Policy Improvement: Extract greedy policy from value function
- Repeat until policy converges
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class GridworldMDP:
    """Gridworld MDP Environment.
    
    A deterministic grid-based environment where the agent navigates from
    start position(s) to goal position(s) while managing a living cost.
    """
    
    # Constants for actions to make code readable
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    
    def __init__(self, rows, cols, terminal_states, step_cost=-0.1, gamma=0.99):
        """Initialize the Gridworld MDP.
        
        Args:
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            terminal_states: Dict mapping (row, col) tuples to their rewards
            step_cost: Reward for each non-terminal state (typically negative)
            gamma: Discount factor
        """
        self.rows = rows
        self.cols = cols
        self.gamma = gamma
        self.step_cost = step_cost  # Reward for non-terminal states
        self.terminal_states = terminal_states  # dict: state -> reward
        
        # Map actions to direction changes: (delta_row, delta_col)
        self.actions = {
            self.UP: (-1, 0),
            self.DOWN: (1, 0),
            self.LEFT: (0, -1),
            self.RIGHT: (0, 1)
        }
    
    def in_bounds(self, r, c):
        """Check if position (r, c) is within grid bounds.
        
        Args:
            r: Row coordinate
            c: Column coordinate
            
        Returns:
            True if position is valid, False otherwise
        """
        return 0 <= r < self.rows and 0 <= c < self.cols
    
    def next_state(self, state, action):
        """Get the next state resulting from taking an action.
        
        Args:
            state: Current position (row, col)
            action: Action index (0-3)
            
        Returns:
            Next state (row, col). If move goes out of bounds, returns current state.
        """
        dr, dc = self.actions[action]
        r, c = state
        nr, nc = r + dr, c + dc
        
        # If out of bounds, stay in same state
        if not self.in_bounds(nr, nc):
            return state
        return (nr, nc)
    
    def is_terminal(self, state):
        """Check if a state is terminal (goal or obstacle).
        
        Args:
            state: Position (row, col)
            
        Returns:
            True if state is terminal, False otherwise
        """
        return state in self.terminal_states
    
    def reward(self, state):
        """Get the reward for being in a state.
        
        Args:
            state: Position (row, col)
            
        Returns:
            Reward value (positive for goals, negative for step cost)
        """
        if self.is_terminal(state):
            return self.terminal_states[state]
        return self.step_cost


def initialize_value_function(env):
    """Initialize the value function (V matrix).
    
    Args:
        env: GridworldMDP environment
        
    Returns:
        NumPy array of shape (rows, cols) initialized to zeros
    """
    return np.zeros((env.rows, env.cols))


def initialize_policy(env):
    """Initialize the policy randomly.
    
    Args:
        env: GridworldMDP environment
        
    Returns:
        Dictionary mapping non-terminal states to random action indices
    """
    policy = {}
    for r in range(env.rows):
        for c in range(env.cols):
            state = (r, c)
            if not env.is_terminal(state):
                policy[state] = np.random.randint(0, 4)
    return policy


def policy_evaluation(env, V, policy, theta=1e-6):
    """Evaluate a given policy by computing its value function.
    
    Iteratively update V(s) for the given policy until convergence using
    the Bellman Expectation Equation:
    V(s) = R(s') + gamma * V(s')
    
    Args:
        env: GridworldMDP environment
        V: Value function (NumPy array)
        policy: Current policy (dict: state -> action)
        theta: Convergence threshold
        
    Returns:
        Updated value function
    """
    iteration = 0
    while True:
        delta = 0
        for r in range(env.rows):
            for c in range(env.cols):
                state = (r, c)
                if env.is_terminal(state):
                    V[r, c] = 0  # Terminal states have 0 value
                    continue
                
                # Get action from current policy
                action = policy[state]
                
                # Deterministic transition
                next_s = env.next_state(state, action)
                
                # Reward for entering next_s
                r_val = env.reward(next_s)
                
                # Bellman Expectation Update (Deterministic environment)
                # V(s) = R(s') + gamma * V(s')
                new_v = r_val + env.gamma * V[next_s[0], next_s[1]]
                
                delta = max(delta, abs(V[r, c] - new_v))
                V[r, c] = new_v
        
        iteration += 1
        if delta < theta:
            break
    
    return V


def policy_improvement(env, V):
    """Extract the greedy policy from the value function.
    
    For each state, compute Q(s,a) for all actions and select the one with
    maximum Q-value. This implements the greedy policy improvement step:
    π'(s) = argmax_a Q(s,a) = argmax_a [R(s') + gamma * V(s')]
    
    Args:
        env: GridworldMDP environment
        V: Current value function (NumPy array)
        
    Returns:
        New policy (dict: state -> best action)
    """
    new_policy = {}
    
    for r in range(env.rows):
        for c in range(env.cols):
            state = (r, c)
            if env.is_terminal(state):
                new_policy[state] = 0  # Action doesn't matter for terminal states
                continue
            
            action_values = []
            # Evaluate all 4 actions
            for a in range(4):  # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
                next_s = env.next_state(state, a)
                reward = env.reward(next_s)
                # Q(s,a) = R(s') + gamma * V(s')
                q_val = reward + env.gamma * V[next_s[0], next_s[1]]
                action_values.append(q_val)
            
            # Select action with maximum Q-value (greedy)
            best_action = np.argmax(action_values)
            new_policy[state] = best_action
    
    return new_policy


def policy_iteration(env):
    """Run the complete Policy Iteration algorithm.
    
    Alternates between:
    1. Policy Evaluation: Compute V^pi(s) for current policy
    2. Policy Improvement: Extract greedy policy from V
    
    Repeat until policy converges (doesn't change).
    
    Args:
        env: GridworldMDP environment
        
    Returns:
        Tuple of (final value function, final policy)
    """
    V = initialize_value_function(env)
    policy = initialize_policy(env)
    
    iteration = 0
    while True:
        # Step 1: Policy Evaluation
        V = policy_evaluation(env, V, policy)
        
        # Step 2: Policy Improvement
        new_policy = policy_improvement(env, V)
        
        # Check for convergence
        if new_policy == policy:
            print(f"✓ Policy Iteration converged in {iteration} iterations")
            break
        
        policy = new_policy
        iteration += 1
    
    return V, policy


def print_policy_grid(env, policy):
    """Visualize the policy as a grid with directional symbols.
    
    Symbols:
    - '^' = UP
    - 'v' = DOWN
    - '<' = LEFT
    - '>' = RIGHT
    - '*' = Terminal state
    
    Args:
        env: GridworldMDP environment
        policy: Policy dict (state -> action)
    """
    symbols = {0: '^', 1: 'v', 2: '<', 3: '>'}
    
    print(f"\n{'='*50}")
    print(f"Policy Grid (Step Cost: {env.step_cost}):")
    print(f"{'='*50}")
    for r in range(env.rows):
        row_str = "|"
        for c in range(env.cols):
            state = (r, c)
            if env.is_terminal(state):
                row_str += " * |"
            elif state in policy:
                action = policy[state]
                row_str += f" {symbols.get(action, '?')} |"
            else:
                row_str += " ? |"
        print(row_str)
    print(f"{'='*50}")


def print_value_grid(V):
    """Visualize the value function as a heatmap.
    
    Args:
        V: Value function (NumPy array)
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(V, annot=True, fmt='.2f', cmap='RdYlGn', cbar_kws={'label': 'Value'})
    plt.title('Value Function Heatmap V(s)')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.tight_layout()
    return plt


def policy_agent_1(state):
    """Random Agent: Takes random actions.
    
    Args:
        state: Current position (row, col)
        
    Returns:
        Random action index (0-3)
    """
    return np.random.randint(0, 4)


def policy_agent_2(state):
    """Heuristic Agent: Moves toward the goal (right and down).
    
    Simple heuristic: If not at right edge, prefer RIGHT. Otherwise, prefer DOWN.
    
    Args:
        state: Current position (row, col)
        
    Returns:
        Action index favoring RIGHT and DOWN
    """
    r, c = state
    if c < 3:  # Not at right edge
        return 3  # RIGHT
    elif r > 0:  # Can go UP toward top-right
        return 0  # UP
    else:
        return 3  # RIGHT


def run_episode(env, policy_fn, start_state=(0, 0), max_steps=100):
    """Simulate one full episode following a policy.
    
    Args:
        env: GridworldMDP environment
        policy_fn: Function that takes state and returns action
        start_state: Starting position
        max_steps: Maximum steps before timeout
        
    Returns:
        Tuple of (total_reward, steps_taken, success_boolean)
    """
    state = start_state
    total_reward = 0
    steps = 0
    success = False
    
    while steps < max_steps:
        if env.is_terminal(state):
            # Check if it was a success (positive reward terminal state)
            if env.terminal_states.get(state, 0) > 0:
                success = True
            break
        
        # Get action from policy
        action = policy_fn(state)
        next_s = env.next_state(state, action)
        
        # Collect reward for entering next state
        r = env.reward(next_s)
        
        total_reward += r
        state = next_s
        steps += 1
    
    return total_reward, steps, success


if __name__ == "__main__":
    print("\n" + "#" * 50)
    print("# Gridworld MDP - Policy Iteration")
    print("#" * 50)
    
    # Example: Create a 4x4 gridworld
    # Goal at (3, 3) with reward +1
    # Obstacle at (2, 2) with reward -1
    # Living cost: -0.1 per step
    
    terminal_states = {
        (3, 3): 1.0,   # Goal: positive reward
        (2, 2): -1.0   # Obstacle: negative reward
    }
    
    env = GridworldMDP(
        rows=4,
        cols=4,
        terminal_states=terminal_states,
        step_cost=-0.1,
        gamma=0.99
    )
    
    print(f"\n✓ Environment initialized")
    print(f"  Grid size: {env.rows}x{env.cols}")
    print(f"  Terminal states: {terminal_states}")
    print(f"  Step cost: {env.step_cost}")
    print(f"  Discount factor: {env.gamma}")
    
    # Run Policy Iteration
    print(f"\n→ Running Policy Iteration...")
    V, policy = policy_iteration(env)
    
    # Display results
    print_policy_grid(env, policy)
    
    print(f"\n✓ Value Function (V-matrix):")
    print(V)
    
    # Test optimal policy
    print(f"\n→ Testing optimal policy (10 episodes from (0,0)):")
    total_rewards = []
    successes = 0
    
    for _ in range(10):
        reward, steps, success = run_episode(
            env,
            lambda s: policy[s] if s in policy else 0,
            start_state=(0, 0),
            max_steps=100
        )
        total_rewards.append(reward)
        if success:
            successes += 1
    
    print(f"\n✓ Results:")
    print(f"  Average Reward: {np.mean(total_rewards):.3f}")
    print(f"  Successful Episodes: {successes}/10")
    print(f"  Max Reward: {np.max(total_rewards):.3f}")
    print(f"  Min Reward: {np.min(total_rewards):.3f}")
    
    # Test random agent for comparison
    print(f"\n→ Testing random policy (10 episodes from (0,0)) for comparison:")
    total_rewards_random = []
    successes_random = 0
    
    for _ in range(10):
        reward, steps, success = run_episode(
            env,
            policy_agent_1,
            start_state=(0, 0),
            max_steps=100
        )
        total_rewards_random.append(reward)
        if success:
            successes_random += 1
    
    print(f"\n✓ Random Agent Results:")
    print(f"  Average Reward: {np.mean(total_rewards_random):.3f}")
    print(f"  Successful Episodes: {successes_random}/10")
    
    print(f"\n✨ Improvement: {np.mean(total_rewards) - np.mean(total_rewards_random):.3f} reward per episode")
