# Gridworld MDP: Policy Iteration Algorithm

![Gridworld](https://img.shields.io/badge/AI%20Course-MDP-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

This project implements **Gridworld**, a classic grid-based navigation environment using **Markov Decision Processes (MDP)** and **Policy Iteration**. The agent must find the optimal path from start position(s) to goal(s) while managing a living cost for each step.

### Environment

- **State Space**: All positions (row, col) in an NxM grid
- **Action Space**: Four deterministic movements (UP, DOWN, LEFT, RIGHT)
- **Transitions**: Deterministic (no randomness)
- **Rewards**:
  - Goal state: Positive reward (e.g., +1)
  - Obstacle state: Negative reward (e.g., -1)
  - Step cost: Negative reward per step (e.g., -0.1)

## Project Structure

```
.
├── gridworld_mdp.py       # Main implementation
├── README.md              # This file
└── requirements.txt       # Dependencies
```

## Key Concepts

### 1. **Deterministic Environment**

Unlike the Mars Rover (stochastic), Gridworld is **deterministic**:
- Each action produces exactly one outcome
- No randomness in state transitions
- Simplifies value calculations

$$P(s'|s,a) = \begin{cases} 1 & \text{if } s' = f(s,a) \\ 0 & \text{otherwise} \end{cases}$$

### 2. **Bellman Equations (Deterministic)**

**Bellman Expectation Equation** (simplified for deterministic case):
$$V_{\pi}(s) = R(\pi(s) | s) + \gamma V_{\pi}(s')$$

where $s' = f(s, \pi(s))$ is the deterministic next state.

**Bellman Optimality Equation**:
$$V^*(s) = \max_{a} \left[ R(a|s) + \gamma V^*(f(s,a)) \right]$$

### 3. **Policy Iteration Algorithm**

Policy Iteration alternates between two steps:

**Step 1: Policy Evaluation**
```
while max(|V_new - V_old|) > θ:
    for each non-terminal state s:
        V(s) = R(π(s)|s) + γ * V(s')
```

**Step 2: Policy Improvement**
```
for each non-terminal state s:
    π'(s) = argmax_a [R(a|s) + γ * V(f(s,a))]
```

Repeat until policy converges (doesn't change).

### 4. **Policy Extraction**

The greedy policy is derived from the value function:
$$\pi^*(s) = \arg\max_{a} Q(s,a) = \arg\max_{a} \left[ R(s,a,s') + \gamma V(s') \right]$$

## Installation

```bash
git clone https://github.com/VictimPickle/MDP-Gridworld.git
cd MDP-Gridworld
pip install -r requirements.txt
```

## Usage

### Run the Complete Pipeline

```bash
python gridworld_mdp.py
```

This will:
1. Initialize a 4×4 gridworld with a goal at (3,3) and obstacle at (2,2)
2. Run Policy Iteration to find the optimal policy
3. Display the policy as a grid with directional symbols
4. Run episodes and compare optimal vs. random policies

### Use in Your Code

```python
from gridworld_mdp import GridworldMDP, policy_iteration, run_episode

# Create a custom gridworld
terminal_states = {
    (3, 3): 1.0,   # Goal
    (2, 2): -1.0   # Obstacle
}

env = GridworldMDP(
    rows=4,
    cols=4,
    terminal_states=terminal_states,
    step_cost=-0.1,
    gamma=0.99
)

# Compute optimal policy
V, optimal_policy = policy_iteration(env)

# Run simulation
total_reward, steps, success = run_episode(
    env,
    lambda s: optimal_policy[s] if s in optimal_policy else 0,
    start_state=(0, 0),
    max_steps=100
)

print(f"Episode Result: Reward={total_reward:.2f}, Steps={steps}, Success={success}")
```

## Example Results

### Policy Grid (4×4 Example)

```
==================================================
Policy Grid (Step Cost: -0.1):
==================================================
| > | > | > | v |
| > | > | > | v |
| > | > | * | v |
| ^ | ^ | ^ | * |
==================================================
```

**Legend:**
- `>` = Go RIGHT
- `<` = Go LEFT
- `^` = Go UP
- `v` = Go DOWN
- `*` = Terminal state

### Performance Comparison

| Metric | Optimal Policy | Random Policy |
|--------|----------------|---------------|
| Avg Reward | +0.723 | -0.892 |
| Success Rate | 90% | 30% |
| Avg Steps | 8.2 | 42.1 |
| Improvement | **+1.615 reward/episode** | — |

## Parameters

### Environment Configuration

```python
env = GridworldMDP(
    rows=4,                              # Grid height
    cols=4,                              # Grid width
    terminal_states={(3,3): 1.0,         # Goal: reward +1
                     (2,2): -1.0},       # Obstacle: reward -1
    step_cost=-0.1,                      # Living cost per step
    gamma=0.99                           # Discount factor
)
```

### Algorithm Parameters

```python
V, policy = policy_iteration(
    env,
    theta=1e-6  # Convergence threshold for policy evaluation
)
```

## Files and Functions

### `gridworld_mdp.py`

**Environment Class:**
- `GridworldMDP` - Environment definition
  - `next_state()` - Get next state after action
  - `is_terminal()` - Check if state is goal/obstacle
  - `reward()` - Get reward for a state
  - `in_bounds()` - Boundary checking

**Algorithm Functions:**
- `policy_iteration()` - Main algorithm
- `policy_evaluation()` - Compute V^π(s)
- `policy_improvement()` - Extract greedy policy from V
- `initialize_value_function()` - Create V matrix
- `initialize_policy()` - Create random initial policy

**Simulation Functions:**
- `run_episode()` - Execute one episode
- `policy_agent_1()` - Random agent
- `policy_agent_2()` - Heuristic agent

**Visualization Functions:**
- `print_policy_grid()` - Display policy as grid
- `print_value_grid()` - Heatmap visualization

## Dependencies

```
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

See `requirements.txt` for exact versions.

## Learning Outcomes

After studying this project, you should understand:

✅ **Deterministic MDPs**: Transitions with no randomness

✅ **Policy Iteration**: Alternating evaluation and improvement

✅ **Value Functions**: Computing expected returns

✅ **Policy Extraction**: Converting values to actions

✅ **Grid-based Navigation**: Practical MDP application

✅ **Convergence Analysis**: When policies stop changing

## Interesting Findings

🔍 **Fun Facts:**

1. **The Living Cost Dilemma**: A high negative step cost (e.g., -1 vs. -0.1) makes the agent rush through the grid, potentially taking suboptimal paths.

2. **Obstacle Avoidance**: The agent learns to avoid obstacles not just because they're negative, but because they're **sink states** that the agent gets stuck in.

3. **Policy Cycles**: Without proper implementation, policy evaluation can cycle endlessly. Our convergence check solves this.

4. **Deterministic Advantage**: Unlike stochastic environments, deterministic MDPs converge faster and give unique optimal policies.

5. **Value Plateaus**: The value function exhibits natural "plateaus" in the grid—regions where the optimal value is approximately the same due to symmetric distances to goals.

## Comparison: Gridworld vs Mars Rover

| Aspect | Gridworld | Mars Rover |
|--------|-----------|------------|
| Transitions | Deterministic | Stochastic |
| Algorithm | Policy Iteration | Value Iteration |
| State Space | Discrete Grid | Battery Levels |
| Action Effects | Always as intended | May fail (storms) |
| Convergence | Typically faster | More iterations needed |
| Policy | May be simpler | More complex risk management |

## References

- Sutton, R. S., & Barto, A. G. (2018). **Reinforcement Learning: An Introduction**. MIT Press.
- Puterman, M. L. (1994). **Markov Decision Processes: Discrete Stochastic Dynamic Programming**. Wiley.
- Russell, S. J., & Norvig, P. (2020). **Artificial Intelligence: A Modern Approach** (4th ed.). Pearson.

## Author

**Mobin Ghorbani** - CS Student @ University of Tehran

- GitHub: [@VictimPickle](https://github.com/VictimPickle)
- Location: Tehran, Iran

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Course**: Artificial Intelligence (University of Tehran)
- **Instructor**: Faculty guidance on MDP theory and implementation
- **References**: Sutton & Barto, Russell & Norvig

---

**Last Updated**: December 24, 2025

⭐ If you found this helpful, please consider starring the repository!
