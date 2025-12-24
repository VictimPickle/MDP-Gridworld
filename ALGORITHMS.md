# Algorithms and Theoretical Foundation

## Policy Iteration Algorithm

### Overview

**Policy Iteration** is a dynamic programming algorithm that finds the optimal policy by alternating between:
1. **Policy Evaluation**: Computing the value function for the current policy
2. **Policy Improvement**: Extracting a greedy policy from the value function

This process repeats until the policy converges (stops changing).

### Algorithm Pseudocode

```
Function POLICY_ITERATION(env):
    π ← random initial policy
    V ← 0 for all states
    
    repeat:
        // Step 1: Policy Evaluation
        V ← POLICY_EVAL(env, π, V)
        
        // Step 2: Policy Improvement
        π_old ← π
        π ← POLICY_IMPROVE(env, V)
        
        // Check convergence
        if π == π_old:
            break
    
    return V, π

---

Function POLICY_EVAL(env, π, V, theta=1e-6):
    repeat:
        delta ← 0
        
        for each state s:
            if s is terminal:
                V[s] ← 0
                continue
            
            old_v ← V[s]
            action ← π(s)
            next_s ← f(s, action)  // Deterministic transition
            reward ← R(next_s)     // Reward for entering next_s
            
            V[s] ← reward + gamma * V[next_s]
            delta ← max(delta, |old_v - V[s]|)
        
        if delta < theta:
            break
    
    return V

---

Function POLICY_IMPROVE(env, V):
    π_new ← empty policy
    
    for each state s:
        if s is terminal:
            π_new[s] ← None
            continue
        
        best_action ← None
        best_q ← -∞
        
        for each action a:
            next_s ← f(s, a)      // Deterministic transition
            reward ← R(next_s)     // Reward for entering next_s
            q_value ← reward + gamma * V[next_s]
            
            if q_value > best_q:
                best_q ← q_value
                best_action ← a
        
        π_new[s] ← best_action
    
    return π_new
```

### Key Characteristics

| Aspect | Value Iteration | Policy Iteration |
|--------|-----------------|------------------|
| **Structure** | Single loop combining eval & improve | Two nested loops |
| **Updates** | Direct to V* | Via policy intermediary |
| **Convergence** | When values stabilize (delta < theta) | When policy stops changing |
| **Best For** | Stochastic environments | Deterministic environments |
| **Policy Updates** | Implicit (extracted after convergence) | Explicit (checked each iteration) |

### Advantages of Policy Iteration

1. **Explicit Policy**: We see which actions are chosen at each step
2. **Convergence Check**: Simpler criterion (policy equality)
3. **Fewer Iterations**: Often converges in fewer outer iterations
4. **Deterministic Advantage**: Shines in deterministic environments like Gridworld

## Bellman Equations for Deterministic MDPs

### Bellman Expectation Equation (Deterministic)

For a deterministic policy π, the value function satisfies:

$$V_{\pi}(s) = R(s') + \gamma V_{\pi}(s')$$

where $s' = f(s, \pi(s))$ is the deterministic next state.

**Interpretation**: Value = immediate reward + discounted future value

### Bellman Optimality Equation (Deterministic)

The optimal value function satisfies:

$$V^*(s) = \max_{a} \left[ R(f(s,a)) + \gamma V^*(f(s,a)) \right]$$

**Interpretation**: Optimal value = max over all actions of (reward + future value)

### Q-Value (Deterministic)

For deterministic transitions:

$$Q(s,a) = R(f(s,a)) + \gamma V(f(s,a))$$

**Interpretation**: Expected return for action a in state s

## Gridworld Environment Details

### State Space

$$S = \{(r,c) : 0 \leq r < \text{rows}, 0 \leq c < \text{cols}\}$$

Example: 4×4 grid has 16 states

### Action Space

$$A = \{\text{UP}, \text{DOWN}, \text{LEFT}, \text{RIGHT}\}$$

Each action produces a deterministic movement.

### Transitions

For each state and action:

$$s' = (r', c') = (r + \Delta r, c + \Delta c)$$

where $(\Delta r, \Delta c)$ depends on the action:
- UP: $(-1, 0)$
- DOWN: $(+1, 0)$
- LEFT: $(0, -1)$
- RIGHT: $(0, +1)$

**Boundary Condition**: If $(r', c')$ is out of bounds, $s' = s$ (stay in place)

### Reward Function

$$R(s) = \begin{cases}
r_{\text{terminal}} & \text{if } s \text{ is terminal} \\
r_{\text{step}} & \text{otherwise}
\end{cases}$$

Typical values:
- Goal terminal state: $r_{\text{terminal}} = +1.0$
- Obstacle terminal state: $r_{\text{terminal}} = -1.0$
- Step cost: $r_{\text{step}} = -0.1$

## Example Walkthrough

### Problem Setup

4×4 Gridworld:
- Goal at (3, 3) with reward +1
- Obstacle at (2, 2) with reward -1
- Step cost: -0.1
- Starting at (0, 0)

### Policy Evaluation Step

For state (0, 0) with policy π(0,0) = RIGHT:

1. **Deterministic next state**: $(0, 1)$
2. **Reward**: $R((0,1)) = -0.1$ (non-terminal, step cost)
3. **Value update** (assuming $V(0,1) = -0.3$ from previous iteration):

$$V(0,0) \leftarrow -0.1 + 0.99 \times (-0.3) = -0.1 - 0.297 = -0.397$$

### Policy Improvement Step

For state (0, 0) with current $V$ function, evaluate all 4 actions:

| Action | Next State | Reward | V[next] | Q-value |
|--------|-----------|--------|---------|----------|
| UP     | (0,0)     | -0.1   | -0.397  | -0.497  |
| DOWN   | (1,0)     | -0.1   | -0.3    | -0.397  |
| LEFT   | (0,0)     | -0.1   | -0.397  | -0.497  |
| RIGHT  | (0,1)     | -0.1   | -0.3    | -0.397  |

**Best action**: DOWN or RIGHT (both give -0.397)

## Convergence Properties

### Convergence Guarantee

Policy Iteration is guaranteed to converge if:
1. The state space is finite
2. The action space is finite
3. The discount factor $\gamma < 1$

### Convergence Speed

**Typical behavior for Gridworld:**
- Iteration 1: Random policy → slightly better policy
- Iteration 2-3: Major improvements as optimal paths emerge
- Iteration 4+: Fine-tuning, policy mostly stable
- Total: Usually 5-10 iterations for small grids

### Why Policy Iteration Works Well for Deterministic Environments

1. **Simpler Value Updates**: No stochastic expectations, just direct value computation
2. **Policy Stability**: Deterministic policies are easier to track
3. **Quick Convergence**: Policies tend to stabilize quickly
4. **Clearer Semantics**: Easy to visualize and understand chosen actions

## Implementation Details

### 1. State Representation

```python
class GridState(tuple):
    """Represents a position in the grid."""
    def __new__(cls, r, c):
        return super().__new__(cls, (r, c))
```

### 2. Terminal State Handling

```python
# Terminal states have fixed values
if state in terminal_states:
    V[state] = 0  # No future transitions
else:
    # Normal Bellman update
    next_state = env.next_state(state, action)
    V[state] = R(next_state) + gamma * V[next_state]
```

### 3. Out-of-Bounds Handling

```python
def next_state(state, action):
    r, c = state
    dr, dc = actions[action]
    nr, nc = r + dr, c + dc
    
    # Check bounds
    if 0 <= nr < rows and 0 <= nc < cols:
        return (nr, nc)
    else:
        return (r, c)  # Stay in place
```

### 4. Policy Equality Check

```python
# Check if policy converged
if new_policy == old_policy:
    converged = True

# This compares all state-action mappings
# Equivalent to: all(new_policy[s] == old_policy[s] for all s)
```

## Performance Optimization

### 1. NumPy Vectorization

```python
# Instead of nested loops for all states:
for each state s:
    for each action a:
        Q[s,a] = ...

# Vectorized approach:
Q = np.zeros((rows, cols, 4))
for a in range(4):
    Q[:,:,a] = rewards + gamma * V[next_states]
```

### 2. Sparse Policy Representation

```python
# Only store non-terminal states
policy = {}
for r in range(rows):
    for c in range(cols):
        if not is_terminal((r,c)):
            policy[(r,c)] = best_action
```

## Debugging Tips

### Check 1: Value Function Sanity

```python
# Values should be bounded by discounted terminal rewards
assert min_value <= min(V.values()) <= max(V.values()) <= max_reward / (1 - gamma)
```

### Check 2: Policy Consistency

```python
# Improved policy should be greedy w.r.t. value function
for s in non_terminal_states:
    a_greedy = argmax_a Q(s,a)
    assert policy[s] == a_greedy  # After improvement
```

### Check 3: Convergence Behavior

```python
# Track delta and policy changes
print(f"Iteration {i}: delta={delta:.6f}, policy_stable={policy_stable}")
```

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
- Puterman, M. L. (1994). *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. Wiley.
- Russell, S. J., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
