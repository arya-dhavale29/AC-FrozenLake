import numpy as np

def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.nstates, dtype=np.float64)
    
    for iteration in range(max_iterations):
        delta = 0
        for state in range(env.nstates):
            v = value[state]
            action = policy[state]
            
            # Calculate new value using Bellman equation
            new_value = 0
            for next_state in range(env.nstates):
                new_value += env.p(next_state, state, action) * (
                    env.r(next_state, state, action) + gamma * value[next_state]
                )
            
            value[state] = new_value
            delta = max(delta, abs(v - value[state]))
        
        if delta < theta:
            break
    
    return value

def policy_improvement(env, value, gamma):
    policy = np.zeros(env.nstates, dtype=int)
    
    for state in range(env.nstates):
        action_values = np.zeros(env.nactions)
        
        for action in range(env.nactions):
            for next_state in range(env.nstates):
                action_values[action] += env.p(next_state, state, action) * (
                    env.r(next_state, state, action) + gamma * value[next_state]
                )
        
        policy[state] = np.argmax(action_values)
    
    return policy

def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.nstates, dtype=int)
    else:
        policy = np.array(policy, dtype=int)
    
    for iteration in range(max_iterations):
        # Policy Evaluation
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        
        # Policy Improvement
        new_policy = policy_improvement(env, value, gamma)
        
        # Check if policy is stable
        if np.array_equal(policy, new_policy):
            break
        
        policy = new_policy
    
    return policy, value

def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.nstates, dtype=np.float64)
    else:
        value = np.array(value, dtype=np.float64)
    
    for iteration in range(max_iterations):
        delta = 0
        
        for state in range(env.nstates):
            v = value[state]
            action_values = np.zeros(env.nactions)
            
            for action in range(env.nactions):
                for next_state in range(env.nstates):
                    action_values[action] += env.p(next_state, state, action) * (
                        env.r(next_state, state, action) + gamma * value[next_state]
                    )
            
            value[state] = np.max(action_values)
            delta = max(delta, abs(v - value[state]))
        
        if delta < theta:
            break
    
    # Extract optimal policy
    policy = policy_improvement(env, value, gamma)
    
    return policy, value