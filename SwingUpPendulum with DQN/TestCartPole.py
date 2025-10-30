# Test environment for the network trained for the swing up task

from CartPoleEnv import SwingUpCartPole
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# This function is only for the purposes of having the angle
# value go from 0 to 2pi instead of from -pi to pi
def wrap_to_2pi(theta):
    return theta % (2 * np.pi)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, n_l1 = 256, n_l2 = 128):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, n_l1)
        self.layer2 = nn.Linear(n_l1, n_l2)
        self.layer3 = nn.Linear(n_l2, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


name = input("Enter network name:\n")
PATH = "Trained_Networks/DQN/Cartpole/" + name + ".pth"

Q_net = torch.load(PATH, weights_only=False)
env = SwingUpCartPole(gamma = 0.99, simul_time=60)

import time

# Load the trained model (assuming Q_net is already trained)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the environment
state, _ = env.reset(start_upright=True)
env.state[0] = 2

# Convert state to tensor
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

x_values = []
x_dots = []
thetas = []
theta_dots = []
F_us = []

# Run a test episode
done = False
sim_max_it = round(env.max_it/6)
it = 0

print('Simulating...')
while not done and it < sim_max_it:
    it += 1
    with torch.no_grad():  # No gradient needed for testing
        action = Q_net(state).max(1).indices.view(1, 1).item()

    # Take action
    next_state, reward, terminated, truncated, _ = env.step(action)

    # Update state
    state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
    
    #save states and actions for plotting
    x_values.append(env.state[0])
    x_dots.append(env.state[1])
    thetas.append(wrap_to_2pi(env.state[2]))
    theta_dots.append(env.state[3])
    F_us.append(env.valid_actions[action]/10)

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
