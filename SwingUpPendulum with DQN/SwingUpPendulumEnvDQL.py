import gymnasium as gym
from gymnasium import spaces
import numpy as np
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
        self.dt = 0.025

        # Action and observation space
        # Control input u is bounded
        self.valid_actions = np.array([-9,0,9])
        #self.valid_actions = np.arange(-9, 10)
        self.action_space = spaces.Discrete(len(self.valid_actions))
        
        # Observation: angle and angular velocity
        self.observation_space = spaces.Box(
            low=np.array([0.0, -np.inf], dtype=np.float32),
            high=np.array([2 * np.pi, np.inf], dtype=np.float32),
            dtype=np.float32
        )
        
        self.stage = 1                              
        self.state = None
        self.current_it = 0                         #current iteration updated in the reset and step functions
        self.max_it = 20/self.dt                    #max 20s of simulation time (for dt = 0.025, max_it = 800)



    def reset(self, seed=None, options=None, Stage2 = False):
        super().reset(seed=seed)
        self.current_it = 0
        angle = uniform(-0.2, 0.2)           
        if Stage2:
            self.stage = 2
            angle = np.pi + uniform(-0.1, 0.1)
                   
        self.state = np.array([angle, 0.0], dtype=np.float32)  # initial position and velocity
        self.stage = 1
        
        return self.state, {}

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

        # WrapToPi
        positiveInput = self.state[0] > 0
        self.state[0] = self.state[0] % (2*np.pi)
        if self.state[0]==0 and positiveInput:
            self.state[0] = 2*np.pi


        terminated = False
        truncated = False

        objective = abs(self.state[0]-np.pi) < 0.1 and abs(self.state[1]) < 0.5        
        
        if objective:
            self.stage = 2

        # Reward System
        if self.stage == 1:
            reward = -(np.abs(u)*(np.pi-self.state[0])**2 + 0.2*self.state[1]**2)
            if self.state[0] > np.pi/2 and self.state[0] < 3*np.pi/2:
                reward += 50
            
            if abs(self.state[0]-np.pi) < 0.1:
                reward += 50
        
        if self.stage == 2:
            reward = 100 - (np.abs(u)*(np.pi-self.state[0])**2 + 0.2*self.state[1]**2)
            
            PendulumFell = np.abs(self.state[0]-np.pi) >= np.pi/2
            
            if PendulumFell:
                terminated = True
                reward -= 10

        if self.current_it >= self.max_it:
            truncated = True

        return self.state, reward, terminated, truncated, {}

    def render(self):
        print(f"Position: {self.state[0]:.2f}, Velocity: {self.state[1]:.2f}")

    def close(self):
        pass


""" 

env = SwingUpPendulum()
env.reset()

n_actions = env.action_space.n

for actionidx in range(n_actions):
    x = env.valid_actions[actionidx]
    if x == 0:
        notorqueindex = actionidx
        break

import time

# Teste do ambiente
env = SwingUpPendulum()  

# Reset
obs, info = env.reset()
done = False
truncated = False

elapsed_time = time.perf_counter()

thetas = []
theta_dots = []
torques = []

for episodes in range(1):
    env.reset()
    for step in range(800):
        #env.render() 
        action = env.action_space.sample()  # Take a random action
        obs, reward, done, truncated, info = env.step(notorqueindex)

        #print(f"Step: {step}, Action: {action}, Obs: {obs}, Reward: {reward}")
        
        thetas.append(obs[0])
        theta_dots.append(obs[1])
        
        if done or truncated:
            #print("Episode ended.")
            break
        
import matplotlib.pyplot as plt


plt.figure(figsize=(10, 4))
plt.plot(thetas, label='θ')
plt.plot(theta_dots, label='ω')
plt.xlabel('Time step')
plt.ylabel('State value')
plt.title('State Variables Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

env.close()

elapsed_time = time.perf_counter() - elapsed_time

print(elapsed_time)
 """