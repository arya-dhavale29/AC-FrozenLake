import numpy as np
import contextlib


# Configures numpy print options
@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


class EnvironmentModel:
    def __init__(self, nstates, nactions, seed=None):
        self.nstates = nstates
        self.nactions = nactions
        self.randomstate = np.random.RandomState(seed)

    def p(self, nextstate, state, action):
        raise NotImplementedError()

    def r(self, nextstate, state, action):
        raise NotImplementedError()

    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.nstates)]
        nextstate = self.randomstate.choice(self.nstates, p=p)
        reward = self.r(nextstate, state, action)
        return nextstate, reward


class Environment(EnvironmentModel):
    def __init__(self, nstates, nactions, maxsteps, pi, seed=None):
        EnvironmentModel.__init__(self, nstates, nactions, seed)
        self.maxsteps = maxsteps
        self.pi = pi
        if self.pi is None:
            self.pi = np.full(nstates, 1.0 / nstates)

    def reset(self):
        self.nsteps = 0
        self.state = self.randomstate.choice(self.nstates, p=self.pi)
        return self.state

    def step(self, action):
        if action < 0 or action >= self.nactions:
            raise Exception("Invalid action.")
        self.nsteps += 1
        done = self.nsteps >= self.maxsteps
        self.state, reward = self.draw(self.state, action)
        return self.state, reward, done

    def render(self, policy=None, value=None):
        raise NotImplementedError()


class FrozenLake(Environment):
    def __init__(self, lake, slip, maxsteps, seed=None):
        """
        lake: a matrix of characters: '&', '.', '#', '$'
        slip: probability of slipping
        maxsteps: maximum episode length
        """
        self.lake = np.array(lake)
        self.lakeflat = self.lake.reshape(-1)
        self.slip = slip

        nstates = self.lake.size + 1 # +1 absorbing
        nactions = 4

        # initial distribution: starts at '&'
        pi = np.zeros(nstates)
        pi[np.where(self.lakeflat == '&')[0]] = 1.0

        self.absorbingstate = nstates - 1

        Environment.__init__(self, nstates, nactions, maxsteps, pi, seed=seed)

    def step(self, action):
        state, reward, done = Environment.step(self, action)
        done = (state == self.absorbingstate) or done
        return state, reward, done

    # IMPLEMENT p FUNCTION
    def p(self, nextstate, state, action):
        # If in absorbing state → stay there
        if state == self.absorbingstate:
            return 1.0 if nextstate == self.absorbingstate else 0.0

        tile = self.lakeflat[state]

        # If tile is terminal (hole or goal) → transition to absorbing
        if tile in ['#', '$']:
            return 1.0 if nextstate == self.absorbingstate else 0.0

        # Possible moves (UP, LEFT, DOWN, RIGHT)
        rows, cols = self.lake.shape
        r, c = divmod(state, cols)

        moves = {
            0: (-1, 0), # UP
            1: (0, -1), # LEFT
            2: (1, 0), # DOWN
            3: (0, 1) # RIGHT
        }

        def move(pos_r, pos_c, action):
            dr, dc = moves[action]
            nr, nc = pos_r + dr, pos_c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                return state # hits wall → stays
            return nr * cols + nc

        intended = move(r, c, action)

        # Slipping: choose random action uniformly
        prob = 0.0
        for a in range(4):
            target = move(r, c, a)
            p_a = (1 - self.slip) if a == action else (self.slip / 3)
            if target == nextstate:
                prob += p_a

        # If falling into a terminal tile, we go to absorbing
        if nextstate == self.absorbingstate:
            # hole or goal at intended?
            term_prob = 0.0
            for a in range(4):
                target = move(r, c, a)
                t = self.lakeflat[target]
                p_a = (1 - self.slip) if a == action else (self.slip / 3)
                if t in ['#', '$']:
                    term_prob += p_a
            return term_prob

        return prob

    # ------------------------------------------------------------
    # TODO: IMPLEMENT r FUNCTION
    # ------------------------------------------------------------
    def r(self, nextstate, state, action):
        # reward only when moving into goal tile
        if state == self.absorbingstate:
            return 0.0

        # nextstate = absorbing because goal reached
        if nextstate == self.absorbingstate:
            tile = self.lakeflat[state]
            rows, cols = self.lake.shape
            r, c = divmod(state, cols)
            moves = {
                0: (-1, 0), # UP
                1: (0, -1), # LEFT
                2: (1, 0), # DOWN
                3: (0, 1) # RIGHT
            }
            dr, dc = moves[action]
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                tile = self.lake[nr, nc]
                return 1.0 if tile == '$' else 0.0
            return 0.0

        return 0.0

    # ------------------------------------------------------------

    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lakeflat)
            if self.state < self.absorbingstate:
                lake[self.state] = '@'
            print(lake.reshape(self.lake.shape))
        else:
            actions = ['^', '<', 'v', '>']
            print("Lake:")
            print(self.lake)
            print("Policy:")
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))
            print("Value:")
            with printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))


def play(env):
    actions = ['w', 'a', 's', 'd']
    state = env.reset()
    env.render()
    done = False
    while not done:
        c = input("\nMove: ")
        if c not in actions:
            raise Exception("Invalid action")
        state, r, done = env.step(actions.index(c))
        env.render()
        print("Reward:", r)

        