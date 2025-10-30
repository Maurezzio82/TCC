from CartPoleTestEnv import CartPole
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn

from modules import TD3_ActorCritic
from modules import DDPG_ActorCritic
from modules import DQN


def wrap_to_2pi(theta):
    return theta % (2 * np.pi)

def deg2rad(theta):
    return theta*np.pi/180

# Load the trained model (assuming Q_net is already trained)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#initializing environment

env = CartPole(gamma = 0.99)

state, _ = env.reset(start_upright=False)

observation_space = env.observation_space
action_space = env.action_space
act_max = env.action_space.high[0]

hidden_sizes = [128, 128]


# LQR gain
K = np.array([-15.1744, -18.2369, -134.4547, -36.9102]) 
# ganhos de x, xd, θ, θd

# ======= loading agents ========= #
PATH = "Trained_Networks/PerformanceTest/Cartpole/DDPG.pth"
state_dict = torch.load(PATH, weights_only=False)
ddpg_ac = DDPG_ActorCritic(observation_space,action_space, hidden_sizes, nn.ReLU).to(device)
ddpg_ac.load_state_dict(state_dict)

PATH = "Trained_Networks/PerformanceTest/Cartpole/TD3.pth"
state_dict = torch.load(PATH, weights_only=False)
td3_ac = TD3_ActorCritic(observation_space,action_space, hidden_sizes, nn.ReLU).to(device)
td3_ac.load_state_dict(state_dict)

PATH = "Trained_Networks/PerformanceTest/Cartpole/DQN.pth"
state_dict = torch.load(PATH, weights_only=False)
MaxForce_DQN = 65.0
ForceStep = 5.0
valid_actions = np.arange(-MaxForce_DQN, MaxForce_DQN+ForceStep, ForceStep, dtype=np.float32)
# the number of valid actions must match that of the environment the DQN was trained for
dqn = DQN(len(state), len(valid_actions), 128, 128)
dqn.load_state_dict(state_dict)


def simulation_loop(angles, agent, sim_max_it = 300):
    total_reward_list= []
    global valid_actions
    #global env


    for start_angle in angles:
        total_reward = 0
        state, _ = env.reset(start_upright=True, angle=deg2rad(start_angle))
        done = False
        it = 0
        while not done and it < sim_max_it:
            it += 1
            with torch.no_grad():  # No gradient needed for testing
                if agent == None:
                    x, x_dot, sin_th, cos_th, th_dot = state
                    th = np.asin(sin_th)
                    state_LQR = np.array([x, x_dot, th, th_dot], dtype=np.float32)
                    LQR_u = -(K*state_LQR).sum()
                    action = np.clip(LQR_u, -act_max, act_max)
                elif type(agent) == DQN:
                    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action_index = agent(state).max(1).indices.view(1, 1).item()
                    action = valid_actions[action_index]
                else:
                    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action = agent.pi(state).cpu().numpy().astype(np.float32).squeeze(0)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Update state
            state = next_state
            

            total_reward += reward.item()
            
            # Check if episode is over
            done = terminated or truncated
            
        total_reward_list.append(total_reward)
    
    return np.array(total_reward_list)


max_angle = 45.0
angle_step = 2.0
angle_range = np.arange(-max_angle, max_angle+angle_step, angle_step)

lqr_rewards = simulation_loop(angle_range, None)

dqn_rewards = simulation_loop(angle_range, dqn)

ddpg_rewards = simulation_loop(angle_range, ddpg_ac)

td3_rewards = simulation_loop(angle_range, td3_ac)


import matplotlib.pyplot as plt
from cycler import cycler

# --- Configure Matplotlib style ---
plt.style.use('seaborn-v0_8-whitegrid')  # clean base style
plt.rcParams.update({
    "axes.prop_cycle": cycler('color', ['#1f77b4', '#ff7f0e', '#9467bd']),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "grid.alpha": 0.3,
    "lines.linewidth": 2,
})
#, '#2ca02c'
# --- Create figure ---
fig, ax = plt.subplots(figsize=(8, 5))

# Plot each agent on the same axis (simpler, cleaner, easier to read)
ax.plot(angle_range, lqr_rewards, label='LQR')
ax.plot(angle_range, dqn_rewards, label='DQN')
#ax.plot(angle_range, ddpg_rewards, label='DDPG')
ax.plot(angle_range, td3_rewards, label='TD3')

# --- Labels and title ---
ax.set_xlabel('Initial Pole Angle (degrees)')
ax.set_ylabel('Total Reward per Episode')
ax.set_title('CartPole Control Performance Comparison')

# --- Grid, legend, layout ---
ax.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.6)
ax.legend(frameon=True, loc='best')
plt.tight_layout()

# --- Show plot ---
plt.show()
