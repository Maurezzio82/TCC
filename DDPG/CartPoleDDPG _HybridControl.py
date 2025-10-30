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

num_episodes = 1500
exploration_episodes = 500
    
    
BATCH_SIZE = 128
GAMMA = 0.99        #already defined above to create the environment
TAU = 0.005         #target network update parameter
LR = 5e-3           #Learning Rate of Q and μ
NOISE_START = 5.0
NOISE_END = 0.1

env = CartPole(gamma = GAMMA)

NOISE_DECAY = env.max_it*num_episodes

# Get action_space
action_space = env.action_space
# Get the state space
observation_space = env.observation_space
# Set sizes of the hidden layers
hidden_sizes = [64, 32]

# get the dimension and limits of the action and observation space
obs_dim = observation_space.shape[0]
act_dim = action_space.shape[0]
act_limit = action_space.high[0]

if input("Train WIP Actor Critic?(y/n)") == 'y':
    main_ac = torch.load("Trained_Networks/DDPG/Cartpole/WIP_main.pth", weights_only=False).to(device)
    target_ac = torch.load("Trained_Networks/DDPG/Cartpole/WIP_target.pth", weights_only=False).to(device)
    WIP = True
else:
    main_ac = ActorCritic(observation_space,action_space, hidden_sizes, nn.ReLU).to(device)
    target_ac = ActorCritic(observation_space,action_space, hidden_sizes, nn.ReLU).to(device)
    target_ac.load_state_dict(main_ac.state_dict())
    WIP = False

if WIP:
    exploration_episodes = 0

for p in target_ac.parameters():
    p.requires_grad = False

critic_optimizer = optim.AdamW(main_ac.q.parameters(), lr=LR, amsgrad=True)
actor_optimizer  = optim.AdamW(main_ac.pi.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(o, Add_Noise=False):
    # o may already be a tensor or numpy array; ensure it's a float32 tensor on device
    if isinstance(o, np.ndarray):
        obs_t = torch.as_tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
    elif isinstance(o, torch.Tensor):
        obs_t = o.to(device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
    else:
        obs_t = torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        a = main_ac.pi(obs_t).cpu().numpy().astype(np.float32).squeeze(0)

    # add gaussian exploration noise and clip to valid action range
    noise_scale = NOISE_END + (NOISE_START - NOISE_END) * \
        math.exp(-1. * steps_done / NOISE_DECAY) if Add_Noise else 0
    
    added_noise = noise_scale * np.random.randn(*a.shape)
    
    
    
    #if steps_done % 100 == 0:
    #   SNR = np.abs(a/added_noise)
    #    print(f"Steps done so far: {steps_done}")
    #    print(f"SNR: {SNR}") 
        
    a += added_noise

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


def hybrid_action(state, action_NN):
    x, x_dot, sin_th, cos_th, th_dot = state
    th = np.asin(sin_th)
    state_LQR = np.array([x, x_dot, th, th_dot], dtype=np.float32)
    LQR_u = -(K*state_LQR).sum()
    Linearity_factor = np.exp(-(2.6*th)**6)     # determines how much the LQR can "take the wheel"

    action = LQR_u*Linearity_factor + action_NN.item()*(1-Linearity_factor)
    action = np.clip(action, -act_limit, act_limit)
   
    return action 
    
    
    
    
    
    
# LQR gain obtained using MATLAB

K = np.array([-2.5813, -5.9623, -90.6496, -30.4344])
# fatores de x, xd, th, thd

    
    
print("Training Started")
print(f"Total Number of Episodes: {num_episodes}")
print(f"Number of Exploratory Episodes: {exploration_episodes}\n")

for i_episode in range(num_episodes):    
    #set exploration flag
    exploration_flag = i_episode < exploration_episodes
    
    #variable for ploting total reward x episode number
    total_reward = 0
    
    #Initialize the environment and get its state
    state, info = env.reset()
    state_NN = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    for t in count():
        
        action_NN = env.action_space.sample() if exploration_flag else select_action(state, Add_Noise=True)
        #no hybrid action during training (?)
        action = action_NN.item() #hybrid_action(state, action_NN)   
        
        observation, reward, terminated, truncated, _ = env.step(action)
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Store the transition in memory
        memory.push(state_NN, action_NN, next_state, reward)

        # Move to the next state
        state = observation

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        steps_done += 1
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_ac_state_dict = target_ac.state_dict()
        main_ac_state_dict = main_ac.state_dict()
        for key in main_ac_state_dict:
            target_ac_state_dict[key] = main_ac_state_dict[key]*TAU + target_ac_state_dict[key]*(1-TAU)
        target_ac.load_state_dict(target_ac_state_dict)

        total_reward += reward
        if done:
            if env.reached_upright:
                print('Pendulum reached upright position')
                print(total_reward.item())
            episode_reward.append(total_reward)
            plot_reward()
            break

print('Complete')
plot_reward(show_result=True)
plt.ioff()
plt.show()



import time

# Load the trained model (assuming main_ac is already trained)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the environment
state, _ = env.reset()

# Convert state to tensor
state_NN = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

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
        action_NN = select_action(state_NN)       # No noise during Testing

    action = hybrid_action(state, action_NN)
    
    # Take action
    next_state, reward, terminated, truncated, _ = env.step(action)

    # Update state
    state_NN = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
    
    #save states and actions for plotting
    x_values.append(env.state[0])
    x_dots.append(env.state[1])
    thetas.append(env.state[2])
    theta_dots.append(env.state[3])
    F_us.append(action/act_limit)

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

x = input("Do you wish to save the policy network? y/n")

if input("Do you wish to save the policy network? y/n") == 'y':
    name = input("Select a name with which to save the policy net:\n")
    torch.save(target_ac,"Trained_Networks/DDPG/Cartpole/" + name + "_target.pth")
    torch.save(main_ac,"Trained_Networks/DDPG/Cartpole/" + name + "_main.pth")
    torch.save(target_ac.state_dict(),"Trained_Networks/DDPG/Cartpole/state_dicts/" + name + "_target.pth")
    torch.save(main_ac.state_dict(),"Trained_Networks/DDPG/Cartpole/state_dicts/" + name + "_main.pth")