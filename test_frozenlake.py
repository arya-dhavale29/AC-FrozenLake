# test_frozenlake.py
import numpy as np
from frozenlake import FrozenLake

# ---- Configure: set path to p.npy and lake layout used in p.npy ----
PNPY_PATH = "p.npy"   # change if p.npy is elsewhere

# This is the standard small 4x4 lake used in many assignments.
# Ensure this matches the map that p.npy was generated from.
lake = [
    ['&', '.', '.', '.'],
    ['.', '#', '.', '#'],
    ['.', '.', '.', '#'],
    ['#', '.', '.', '$']
]

# If you used a different slip when you implemented FrozenLake, set it here.
# The p.npy file encodes the true probabilities for some slip value (typically e.g. 0.1).
# If you're not sure, try default_slip=0.1 first.
default_slip = 0.1

# maxsteps and seed don't affect p(), but we pass them for constructor compatibility
env = FrozenLake(lake, slip=default_slip, maxsteps=100, seed=0)

# Load the ground truth p array
try:
    P = np.load(PNPY_PATH)  # expected shape: (n_nextstates, n_states, n_actions)
except FileNotFoundError:
    raise SystemExit(f"p.npy not found at '{PNPY_PATH}'. Put p.npy in the project root or update PNPY_PATH.")

print("Loaded p.npy with shape:", P.shape)
n_next, n_states, n_actions = P.shape

# Sanity checks
if n_states != env.nstates:
    raise SystemExit(f"Mismatch: p.npy has {n_states} states but env.nstates == {env.nstates}. Check lake or p.npy.")

if n_actions != env.nactions:
    raise SystemExit(f"Mismatch: p.npy has {n_actions} actions but env.nactions == {env.nactions}. Check action ordering.")

# Compare env.p() to p.npy
tol = 1e-8
max_diff = 0.0
mismatch_count = 0
total_entries = n_next * n_states * n_actions

print("Comparing env.p(next,state,action) to p.npy ...")
for s in range(n_states):
    for a in range(n_actions):
        # Check probabilities sum to ~1 in both
        p_np = P[:, s, a]
        sumnp = p_np.sum()
        if not np.isclose(sumnp, 1.0, atol=1e-6):
            print(f"WARNING: p.npy probabilities for state {s}, action {a} sum to {sumnp:.8f} (expected 1.0)")

        for ns in range(n_next):
            p_env = env.p(ns, s, a)
            p_ref = float(p_np[ns])
            diff = abs(p_env - p_ref)
            if diff > tol:
                mismatch_count += 1
                if diff > max_diff:
                    max_diff = diff
                print(f"Mismatch: s={s}, a={a}, ns={ns} | env.p={p_env:.8f} ref={p_ref:.8f} diff={diff:.8e}")
            # optionally you can assert closeness here
print("Done comparing.")
print(f"Total entries checked: {total_entries}")
print(f"Total mismatches (diff>{tol}): {mismatch_count}")
print(f"Maximum absolute difference observed: {max_diff:.8e}")

# Extra: check each (s,a) row sums in your env
print("\nChecking env row sums for each (state,action):")
for s in range(n_states):
    for a in range(n_actions):
        row_sum = sum(env.p(ns, s, a) for ns in range(n_next))
        if not np.isclose(row_sum, 1.0, atol=1e-6):
            print(f"  Row sum not 1.0 for s={s}, a={a}: {row_sum:.8f}")
print("Row-sum check complete.")
