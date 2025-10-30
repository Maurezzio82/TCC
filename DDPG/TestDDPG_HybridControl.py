# Test environment for the network trained for the swing up task

from CartPoleEnv import CartPole
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# This function is only for the purposes of having the angle
# value go from 0 to 2pi instead of from -pi to pi
def wrap_to_2pi(theta):
    return theta % (2 * np.pi)

#================================== Actor-Critic NN Module ==================================#

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class Actor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)

class QFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class ActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,128),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = Actor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = QFunction(obs_dim, act_dim, hidden_sizes, activation)

#============================================================================================#
# LQR gain

K = np.array([-15.1744, -18.2369, -134.4547, -36.9102])
# ganhos de x, xd, th, thd


#============================================================================================#
name = input("Enter network name:\n")
PATH = "Trained_Networks/DDPG/Cartpole/" + name + ".pth"

ac = torch.load(PATH, weights_only=False)
env = CartPole(gamma = 0.99, simul_time=20)
act_max = env.action_space.high[0]

import time

# Load the trained model (assuming Q_net is already trained)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the environment
state, _ = env.reset(start_upright=True)

#env.state[0] = 2.0

# Convert state to tensor
state_nn = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

x_values = []
x_dots = []
thetas = []
theta_dots = []
F_us = []

# Run a test episode
done = False
sim_max_it = env.max_it/5
it = 0


print('Simulating...')

while not done and it < sim_max_it:
    it += 1
    x, x_dot, sin_th, cos_th, th_dot = state
    th = np.asin(sin_th)
    state_LQR = np.array([x, x_dot, th, th_dot], dtype=np.float32)
    LQR_u = -(K*state_LQR).sum()
    
    with torch.no_grad():  # No gradient needed for testing
        action_NN = ac.pi(state_nn).cpu().numpy().astype(np.float32).squeeze(0)
    
    Linearity_factor = np.exp(-(2.6*th)**6)
        
    action = LQR_u*Linearity_factor + action_NN.item()*(1-Linearity_factor)
    action = np.clip(action, -act_max, act_max)



    # Take action
    next_state, reward, terminated, truncated, _ = env.step(action)

    # Update state
    state = next_state
    state_nn = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    #save states and actions for plotting
    x_values.append(env.state[0])
    x_dots.append(env.state[1])
    thetas.append(wrap_to_2pi(env.state[2]))
    theta_dots.append(env.state[3])
    F_us.append(5*action/act_max)

    # Render environment
    time.sleep(0.02)  # Small delay to slow down rendering
    
    # Check if episode is over
    done = terminated or truncated
    env.render()
print('Simulation completed\n')
input("Press enter to close")
env.close()

plt.figure(figsize=(10, 4))
plt.plot(x_values, label = 'x')
plt.plot(x_dots, label = 'v')
plt.plot(thetas, label='θ')
plt.plot(theta_dots, label='ω')
plt.plot(F_us, label = 'F=u')
plt.xlabel('Time step')
plt.ylabel('State value')
plt.title('State Variables Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
