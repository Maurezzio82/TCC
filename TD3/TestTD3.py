# Test environment for the network trained for the swing up task

from CartPoleEnv import CartPole
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def wrap_to_2pi(theta):
    return theta % (2 * np.pi)
def deg2rad(theta):
    return theta*np.pi/180

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
        # For TD3, use two Q-functions (twin critics)
        self.q1 = QFunction(obs_dim, act_dim, hidden_sizes, activation)  #added line: [twin critic q1]
        self.q2 = QFunction(obs_dim, act_dim, hidden_sizes, activation)  #added line: [twin critic q2]


#============================================================================================#


name = input("Enter network name:\n")
PATH = "Trained_Networks/TD3/Cartpole/" + name + ".pth"

ac = torch.load(PATH, weights_only=False)
env = CartPole(gamma = 0.99, simul_time=60)
act_max = env.action_space.high[0]

import time

# Load the trained model (assuming Q_net is already trained)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the environment
state, _ = env.reset(start_upright=False)
env.state[0] = 0.0
# Convert state to tensor
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

x_values = []
x_dots = []
thetas = []
theta_dots = []
F_us = []

# Run a test episode
done = False
sim_max_it = env.max_it/4
it = 0

print('Simulating...')
while not done and it < sim_max_it:
    it += 1
    with torch.no_grad():  # No gradient needed for testing
        action = ac.pi(state).cpu().numpy().astype(np.float32).squeeze(0)

    # Take action
    next_state, reward, terminated, truncated, _ = env.step(action)

    # Update state
    state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
    
    #save states and actions for plotting
    x_values.append(env.state[0])
    x_dots.append(env.state[1])
    thetas.append(wrap_to_2pi(env.state[2]))
    theta_dots.append(env.state[3])
    F_us.append(action/act_max)

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
