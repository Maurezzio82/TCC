import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from scipy.integrate import solve_ivp
from random import uniform

class SwingUpPendulum(gym.Env):
    metadata = {'render.modes': ['console']}
    
    def __init__(self):
        super(SwingUpPendulum, self).__init__()
        
        # System parameters
        self.m = 1.0    # mass
        self.L = 1.0    # pendulum length
        self.b = 0.01   # viscous friction coefficient
        self.g = 9.8    # gravitational acceleration

        # Time step
        self.dt = 0.05

        # Define action and observation space
        # Control input u is bounded
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)
        
        # Observation: position and velocity
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        self.state = None
        self.current_it = 0                         #current iteration updated in the reset and step functions
        self.max_it = 800 * 0.05/self.dt


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_it = 0
        angle = uniform(0.2, 0.2)
        self.state = np.array([angle, 0.0], dtype=np.float32)  # initial position and velocity
        return self.state, {}

    def dynamics(self, t, y, u):
        x1, x2 = y
        dx1 = x2
        dx2 = (u-self.b*x2)/(self.m*self.L**2)-self.g*math.sin(x1)/self.L
        return [dx1, dx2]

    def step(self, action):
        self.current_it += 1
        u = np.clip(action[0], self.action_space.low[0], self.action_space.high[0])
        
        # Integrate ODE
        sol = solve_ivp(self.dynamics, [0, self.dt], self.state, args=(u,), t_eval=[self.dt])
        
        self.state = sol.y[:, -1]

        # WrapToPi
        positiveInput = self.state[0] > 0
        self.state[0] = self.state[0] % 2*np.pi
        if self.state[0]>0 and positiveInput:
            self.state[0] = 2*np.pi

        # Reward 
        reward = -((np.pi-np.abs(self.state[0]))**2 + 0.2*self.state[1]**2)

        if self.current_it >=self.max_it:
            truncated = True
            if -reward < 0.05:
                terminated = True
            else:
                terminated = False
        else:
            truncated = False
            truncated = False    

        return self.state, reward, terminated, truncated, {}

    def render(self):
        print(f"Position: {self.state[0]:.2f}, Velocity: {self.state[1]:.2f}")

    def close(self):
        pass





"""
        t = state[0]
        t_dot = state[1]
        x_1 = np.array([1.0, 1.0, t**2, t, t_dot**2, t_dot, (u*t)**2, u*t, (u*t_dot)**2, u*t_dot], dtype=np.float32) #feature vector for the first stage
 """