def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=float)

    for _ in range(max_iterations):
        delta = 0.0
        new_value = value.copy()

        for s in range(env.n_states):
            a = policy[s]

            v = 0.0
            for ns in range(env.n_states):
                p = env.p(ns, s, a)
                r = env.r(ns, s, a)
                v += p * (r + gamma * value[ns])

            delta = max(delta, abs(v - value[s]))
            new_value[s] = v

        value = new_value
        if delta < theta:
            break

    return value


def policy_improvement(env, value, gamma):
    policy = np.zeros(env.n_states, dtype=int)

    for s in range(env.n_states):
        action_values = np.zeros(env.n_actions)

        for a in range(env.n_actions):
            q = 0.0
            for ns in range(env.n_states):
                p = env.p(ns, s, a)
                r = env.r(ns, s, a)
                q += p * (r + gamma * value[ns])
            action_values[a] = q

        policy[s] = np.argmax(action_values)

    return policy


def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)

    for _ in range(max_iterations):
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        new_policy = policy_improvement(env, value, gamma)

        if np.array_equal(new_policy, policy):
            break

        policy = new_policy

    return policy, value


def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states, dtype=float)
    else:
        value = np.array(value, dtype=float)

    for _ in range(max_iterations):
        delta = 0.0
        new_value = value.copy()

        for s in range(env.n_states):
            action_values = np.zeros(env.n_actions)

            for a in range(env.n_actions):
                q = 0.0
                for ns in range(env.n_states):
                    p = env.p(ns, s, a)
                    r = env.r(ns, s, a)
                    q += p * (r + gamma * value[ns])
                action_values[a] = q

            best_v = np.max(action_values)
            delta = max(delta, abs(best_v - value[s]))
            new_value[s] = best_v

        value = new_value
        if delta < theta:
            break

    policy = policy_improvement(env, value, gamma)

    return policy, value
