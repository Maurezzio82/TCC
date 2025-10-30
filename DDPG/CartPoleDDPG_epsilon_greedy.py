import os
os.environ["QT_LOGGING_RULES"] = "qt.qpa.*=false"

from CartPoleEnv import CartPole
import numpy as np
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward):
        # State
        state = torch.as_tensor(state, dtype=torch.float32, device=device).flatten().unsqueeze(0)

        # Action
        action = torch.as_tensor(action, dtype=torch.float32, device=device).flatten().unsqueeze(0)

        # Reward
        reward = torch.as_tensor([reward], dtype=torch.float32, device=device)

        # Next state
        if next_state is not None:
            next_state = torch.as_tensor(next_state, dtype=torch.float32, device=device).flatten().unsqueeze(0)

        self.memory.append(Transition(state, action, next_state, reward))


    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
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


    
num_episodes = 3000
    
BATCH_SIZE = 128
GAMMA = 0.99        #already defined above to create the environment
TAU = 0.001         #target network update parameter
LR_Q = 1e-4           #Learning Rate of Q
LR_mu = 1e-3            #Learning Rate of μ
EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = num_episodes

# Stop criterion hyper parameters
REWARD_THRESHOLD = 550
WINDOW = 20
EPSILON = 5

env = CartPole(gamma = GAMMA)


# Get action_space
action_space = env.action_space
# Get the state space
observation_space = env.observation_space
# Set sizes of the hidden layers
hidden_sizes = [128, 128]

# get the dimension and limits of the action and observation space
obs_dim = observation_space.shape[0]
act_dim = action_space.shape[0]
act_limit = action_space.high[0]

main_ac = ActorCritic(observation_space,action_space, hidden_sizes, nn.ReLU).to(device)
target_ac = ActorCritic(observation_space,action_space, hidden_sizes, nn.ReLU).to(device)
target_ac.load_state_dict(main_ac.state_dict())


for p in target_ac.parameters():
    p.requires_grad = False

critic_optimizer = optim.AdamW(main_ac.q.parameters(), lr=LR_Q, amsgrad=True)
actor_optimizer  = optim.AdamW(main_ac.pi.parameters(), lr=LR_mu, amsgrad=True)
memory = ReplayMemory(50000)

episodes_done = 0
def select_action(o, Explore=False):
    global episodes_done
    
    # o may already be a tensor or numpy array; ensure it's a float32 tensor on device
    if isinstance(o, np.ndarray):
        obs_t = torch.as_tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
    elif isinstance(o, torch.Tensor):
        obs_t = o.to(device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
    else:
        obs_t = torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
    
    
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * episodes_done / EPS_DECAY) if Explore else 0
    
    if np.random.random() > eps_threshold:
        with torch.no_grad():
            a = main_ac.pi(obs_t).cpu().numpy().astype(np.float32).squeeze(0)         
    else:
        a = env.action_space.sample()
    
    return torch.tensor(a, dtype=torch.float32, device=device).unsqueeze(0)


episode_reward = []

def plot_reward(show_result=False):
    plt.figure(1)
    reward_t = torch.tensor(episode_reward, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(reward_t.numpy(), label='Reward')

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
            
def optimize_model():
    # Ensure enough samples
    if len(memory) < BATCH_SIZE:
        return

    # Sample batch and unpack
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Prepare tensors on device
    state_batch = torch.cat(batch.state)        # [B, obs_dim]
    action_batch = torch.cat(batch.action)      # [B, act_dim]
    reward_batch = torch.cat(batch.reward).unsqueeze(1)  # [B,1]

    
    # next states mask & tensor
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device, dtype=torch.bool)
    if any(non_final_mask):
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)
    else:
        non_final_next_states = torch.empty((0, state_batch.shape[1]), device=device)

    # ----- Critic update -----
    # Q(s,a) from current critic
    q_vals = main_ac.q(state_batch, action_batch).unsqueeze(1)  # shape [B,1]

    # Compute target Q: r + gamma * Q_target(s', pi_target(s')) for non-terminal s'
    with torch.no_grad():
        target_q = torch.zeros((BATCH_SIZE, 1), device=device)
        if non_final_next_states.size(0) > 0:
            next_actions = target_ac.pi(non_final_next_states)        # [N_nonfinal, act_dim]
            q_next = target_ac.q(non_final_next_states, next_actions).unsqueeze(1)  # [N,1]
            # place into target_q only where non-final
            target_q[non_final_mask, :] = q_next
        target_q = reward_batch + (GAMMA * target_q)

    # Critic loss: MSE between current Q and target_q
    critic_loss = F.mse_loss(q_vals, target_q)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_value_(main_ac.q.parameters(), 100.0)
    critic_optimizer.step()

    # ----- Actor update -----
    # Actor objective: maximize Q(s, pi(s)) => minimize -mean(Q(s, pi(s)))
    actor_actions = main_ac.pi(state_batch)  # [B, act_dim]
    actor_loss = -main_ac.q(state_batch, actor_actions).mean()

    actor_optimizer.zero_grad() 
    actor_loss.backward()
    torch.nn.utils.clip_grad_value_(main_ac.pi.parameters(), 100.0)
    actor_optimizer.step()

    return q_vals.mean().item()         # mean of q values is returned for logging

    
print("Training Started")
print(f"Total Number of Episodes: {num_episodes}")

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    total_reward = 0
    state, info = env.reset(start_upright=True)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    episode_steps = 0
    mean_q_values = [] 

    # Monitor how often random actions are being chosen
    
    for t in count():
        action = select_action(state, Explore=True)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        q_val = optimize_model()

        if q_val is not None:
            mean_q_values.append(q_val)
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_ac_state_dict = target_ac.state_dict()
        main_ac_state_dict = main_ac.state_dict()
        for key in main_ac_state_dict:
            target_ac_state_dict[key] = main_ac_state_dict[key]*TAU + target_ac_state_dict[key]*(1-TAU)
        target_ac.load_state_dict(target_ac_state_dict)

        total_reward += reward.item()
        if done:
            #if env.reached_upright:
                #print('Pendulum is upright')
                #print(total_reward.item())
            episode_reward.append(total_reward)
            plot_reward()
            episodes_done += 1
            episode_steps = t
            break
        
    avg_q = np.mean(mean_q_values) if len(mean_q_values) > 0 else 0.0

    if i_episode % 5 == 0:
        # You can recompute eps_threshold here using steps_done
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * episodes_done / EPS_DECAY)
        print(f"Episode {i_episode+1:4d} | "
              f"Eps: {eps_threshold:.3f} | Replay size: {len(memory):5d} | "
              f"Episode length: {episode_steps:3d} | Total reward: {total_reward:.2f} | "
              f"Mean Q-value: {avg_q:7.3f}")
        
    if len(episode_reward) >= 2 * WINDOW:
        recent_avg = np.mean(episode_reward[-WINDOW:])
        previous_avg = np.mean(episode_reward[-2*WINDOW:-WINDOW])
        if abs(recent_avg - previous_avg) < EPSILON and recent_avg >= REWARD_THRESHOLD:
            print(f"\nStopping early at episode {i_episode}: "
                  f"Average reward {recent_avg:.2f} stabilized (≥ {REWARD_THRESHOLD}).")
            break

print('Complete')
plot_reward(show_result=True)
plt.ioff()
plt.show()



import time

# Load the trained model (assuming main_ac is already trained)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the environment
state, _ = env.reset(start_upright=True)

# Convert state to tensor
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

x_values = []
x_dots = []
thetas = []
theta_dots = []
F_us = []

# Run a test episode
done = False
sim_max_it = env.max_it
it = 0

print("Simulating...")
while not done and it < sim_max_it:
    it += 1
    with torch.no_grad():  # No gradient needed for testing
        action = select_action(state)       # No noise during Testing

    # Take action
    next_state, reward, terminated, truncated, _ = env.step(action)

    # Update state
    state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
    
    #save states and actions for plotting
    x_values.append(env.state[0])
    x_dots.append(env.state[1])
    thetas.append(env.state[2])
    theta_dots.append(env.state[3])
    F_us.append(action[0]/act_limit)

    # Render environment
    time.sleep(0.02)  # Small delay to slow down rendering
    
    # Check if episode is over
    done = terminated or truncated


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
print("done")

env.close()  # Close environment after testing

if input("Do you wish to save the policy network? y/n") == 'y':
    name = input("Select a name with which to save the policy net:\n")
    torch.save(target_ac,"Trained_Networks/DDPG/Cartpole/" + name + "_target.pth")
    torch.save(main_ac,"Trained_Networks/DDPG/Cartpole/" + name + "_main.pth")
    torch.save(target_ac.state_dict(),"Trained_Networks/DDPG/Cartpole/state_dicts/" + name + "_target.pth")
    torch.save(main_ac.state_dict(),"Trained_Networks/DDPG/Cartpole/state_dicts/" + name + "_main.pth")