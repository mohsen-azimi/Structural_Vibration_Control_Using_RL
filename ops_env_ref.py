"""
Notes:
"""


import math
from functools import partial
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from scipy.integrate import solve_ivp


from ops_SDOF import test_ops_model, ops_initiateGravity, ops_initiateTH, ops_continueTH


class opsEnv():
    """
    Description:
        Cantiliver column vibration

	Observations:
        Type: Box(3)
        Num	Observation               Min           Max
        0	Floor1 Position (in)     -Inf           Inf

    Actions:
        Type: Box(1)
        Num	Action                     Min           Max
        0	Force on Floor1            -.1           .1     (force in kip)

    Reward:
        The reward is calculated each time step and is a negative cost.
        The cost function is the
          (i) First floor x-position/drift


    Starting State:
        Each episode, the system starts from t=0.0 sec (one step, using 'ops_initiateTH')

    Episode Termination:
        Episode ends after 100 timesteps.

    Solved Requirements:
        To be determined by comparison with the ideal controller.
    """


    def __init__(self, description="ops_cantiliver_RC",
                 goal_state=(0.0),
                 initial_state='goal',
                 measurement_error=None,  # Not implemented yet
                 n_steps=10
                 ):

        self.description = description

        # Physical attributes of system
        self.max_force = 0.1

        # Set initial state and goal state
        self.goal_state = np.array(goal_state, dtype=np.float32)
        if initial_state == 'goal':
            self.initial_state = self.goal_state.copy()
        else:
            self.initial_state = np.array(initial_state, dtype=np.float32)


        # Details of simulation
        self.dt = 0.01   # seconds between state updates
        self.n_steps = n_steps
        self.time_step = 0

        # Maximum and minimum floor position
        self.x_threshold = 1.0  # limit for lateral displ {change to drift}

        # Thresholds for observation bounds
        high = np.array([
            self.x_threshold],
            dtype=np.float32)

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(np.float32(-self.max_force),
                                       np.float32(self.max_force),
                                       shape=(1,), dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def cost_function(self, state, goal_state):
        """Evaluates the cost based on the current state y and
        the goal state.
        """

        return ((state[0] - self.goal_state[0])**2)

    def ops_initiate(self):
        self.myops = ops_initiateGravity()
        # self.myops.ops_initiateTH()
        # self.myops.ops_initiateTH()
        # self.myops.ops_continueTH()
        self.myops.ops_draw2D()


    def step(self, u):
        u = np.clip(u, -self.max_force, self.max_force)[0]

        # self.myops.ops_continueTH()

        # Simple state update (test; get it from opensees)
        self.state = np.array([
            15.0],
            dtype=np.float32)


        reward = -self.cost_function(self.state, self.goal_state)


        self.time_step += 1
        done = True if self.time_step >= self.n_steps else False

        return self.state, reward, done, {}

    def reset(self):

        self.state = self.initial_state.copy()
        assert self.state.shape[0] == 3

    def render(self, mode='human', close=False):
        screen_width = 600
        screen_height = 400


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


