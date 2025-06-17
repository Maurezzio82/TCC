import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
from random import uniform

def wrap_to_pi(theta):
    return ((theta + np.pi) % (2 * np.pi)) - np.pi

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
        self.dt = 0.025

        # Action and observation space
        # Control input u is bounded
        self.valid_actions = np.array([-5,0,5])
        self.action_space = spaces.Discrete(len(self.valid_actions))
        
        # Observation: sin and cos of the angle and angular velocity
        self.observation_space = spaces.Box(
            low=np.array([-1, -np.inf], dtype=np.float32),
            high=np.array([1, np.inf], dtype=np.float32),
            dtype=np.float32)
        
        self.reached_upright = False                              
        self.state = None
        self.current_it = 0                         #current iteration updated in the reset and step functions
        self.max_it = 15/self.dt                    #max 15s of simulation time (for dt = 0.025, max_it = 800)



    def reset(self, seed=None, options=None, start_upright = False):
        super().reset(seed=seed)
        self.current_it = 0
        angle = uniform(-0.2, 0.2)
        if start_upright:
            self.reached_upright = 2
            angle = np.pi + uniform(-0.1, 0.1)
                   
        self.state = np.array([angle, 0.0], dtype=np.float32)  # initial position and velocity
        self.reached_upright = False
        
        sin_theta = np.sin(angle)
        cos_theta = np.cos(angle)
        self.observation = np.array([sin_theta, cos_theta, 0.0], dtype=np.float32)
        
        return self.observation, {}

    def dynamics(self, t, y, u):
        x1, x2 = y
        dx1 = x2
        dx2 = (u-self.b*x2)/(self.m*self.L**2)-self.g*np.sin(x1)/self.L
        return [dx1, dx2]

    def step(self, action_idx):
        self.current_it += 1
        action = self.valid_actions[action_idx]
        u = float(action)
            
        # Integrate ODE
        sol = solve_ivp(self.dynamics, [0, self.dt], self.state, args=(u,), t_eval=[self.dt])

        self.state = sol.y[:, -1]
        #print(type(self.state))

        terminated = False
        truncated = False

        # WrapToPi
        self.state[0] = wrap_to_pi(self.state[0])
        
        theta, theta_dot = self.state
        theta_error = wrap_to_pi(theta - np.pi)


        upright_now = abs(theta_error) < 0.1 and abs(theta_dot) < 0.5        
        
        if upright_now:
            self.reached_upright = True

        # Reward System
        
        # Directional shaping: encourage velocity toward upright
        direction_bonus = -0.5 * theta_dot * np.tanh(10 * theta_error)

        # Smooth reward shaping toward upright
        shaping_reward = 100 / (1 + (theta_error / 0.2)**2)

        # Cost terms
        cost = (theta_error)**2 + 0.2 * theta_dot**2 + 0.1 * u**2
        reward = shaping_reward + direction_bonus - cost

        if self.reached_upright and abs(theta_error) >= np.pi/2:
            terminated = True

        if self.current_it >= self.max_it:
            truncated = True
        
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        self.observation = np.array([sin_theta, cos_theta, theta_dot], dtype=np.float32)
        return self.observation, reward, terminated, truncated, {}

    def render(self):
        print(f"Position: {self.state[0]:.2f}, Velocity: {self.state[1]:.2f}")

    def close(self):
        pass