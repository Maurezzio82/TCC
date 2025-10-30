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
        # For TD3, use two Q-functions (twin critics)
        self.q1 = QFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = QFunction(obs_dim, act_dim, hidden_sizes, activation)

#============================================================================================#

num_episodes = 4000
exploration_episodes = 1000
    
    
BATCH_SIZE = 256
GAMMA = 0.99        #already defined above to create the environment
TAU = 0.001         #target network update parameter
LR_Q = 1e-4           #Learning Rate of Q
LR_mu = 1e-3        #Learning Rate of μ
NOISE_START = 0.3   #This represents the start multiple of act_lim that is available as noise
NOISE_END = 0.02   
POLICY_DELAY = 2    #Number of critic updates per policy update (2 or 3 allowed)

# TD3 specific hyperparams
TARGET_POLICY_NOISE = 0.2    #Stddev of target policy smoothing noise
TARGET_NOISE_CLIP = 0.5      #Clipping range for target policy noise
EXPL_NOISE = 0.1             #Exploration noise stddev added to actions during data collection
# NOTE: EXPL_NOISE and NOISE_START interplay — EXPL_NOISE is additional gaussian noise (std) used at action selection.

START_UPRIGHT = False           #set this flag false if you want to train the network to do the swing up task
REWARD_THRESHOLD = 565 if START_UPRIGHT else 450
WINDOW = 20
EPSILON = 5

env = CartPole(gamma = GAMMA)

NOISE_DECAY = num_episodes*env.max_it/15

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


# Declaring the AC objects
main_ac = ActorCritic(observation_space,action_space, hidden_sizes, nn.ReLU).to(device)
target_ac = ActorCritic(observation_space,action_space, hidden_sizes, nn.ReLU).to(device)
target_ac.load_state_dict(main_ac.state_dict())



for p in target_ac.parameters():
    p.requires_grad = False

# For TD3 we need separate optimizers for two critics and one actor
critic1_optimizer = optim.AdamW(main_ac.q1.parameters(), lr=LR_Q, amsgrad=True)  #optimizer for q1
critic2_optimizer = optim.AdamW(main_ac.q2.parameters(), lr=LR_Q, amsgrad=True)  #optimizer for q2
actor_optimizer  = optim.AdamW(main_ac.pi.parameters(), lr=LR_mu, amsgrad=True)
# If you prefer single optimizer for both critics, you could combine parameters (not done here).
memory = ReplayMemory(50000)

steps_done = 0
td3_update_step = 0  #counter for delayed policy updates

def select_action(o, Add_Noise=False):
    # o may already be a tensor or numpy array; ensure it's a float32 tensor on device
    global steps_done
    
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
    # For TD3 we use EXPL_NOISE (std) scaled by act_limit, and also keep your decaying noise scale if Add_Noise True
    noise_scale = NOISE_END + (NOISE_START - NOISE_END) * \
        math.exp(-1. * steps_done / NOISE_DECAY) if Add_Noise else 0
    steps_done += 1
    if Add_Noise:
        a += noise_scale * np.random.randn(*a.shape) * act_limit
        a += EXPL_NOISE * np.random.randn(*a.shape) * act_limit  
    a = np.clip(a, -act_limit, act_limit)

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
    # Q(s,a) from current critics
    q1_vals = main_ac.q1(state_batch, action_batch).unsqueeze(1)  # shape [B,1] q1 current
    q2_vals = main_ac.q2(state_batch, action_batch).unsqueeze(1)  # shape [B,1] q2 current

    # Compute target Q: r + gamma * min(Q1_target(s', pi_target(s') + noise), Q2_target(...))
    with torch.no_grad():
        target_q = torch.zeros((BATCH_SIZE, 1), device=device)  # placeholder
        if non_final_next_states.size(0) > 0:
            # target policy smoothing: add clipped noise to target action
            next_actions = target_ac.pi(non_final_next_states)        # [N_nonfinal, act_dim]  [target actor output]
            # sample noise
            noise = (torch.randn_like(next_actions) * TARGET_POLICY_NOISE).clamp(-TARGET_NOISE_CLIP, TARGET_NOISE_CLIP)  #[sample and clip target noise]
            next_actions = (next_actions + noise).clamp(-act_limit, act_limit)  #[apply smoothing noise and clip actions]
            q1_next = target_ac.q1(non_final_next_states, next_actions).unsqueeze(1)  # [N,1] [q1 target]
            q2_next = target_ac.q2(non_final_next_states, next_actions).unsqueeze(1)  # [N,1] [q2 target]
            min_q_next = torch.min(q1_next, q2_next)  # [N,1] [take min of twin critics]
            # place into target_q only where non-final
            target_q[non_final_mask, :] = min_q_next
        target_q = reward_batch + (GAMMA * target_q)

    # Critic losses: MSE between current Qs and target_q
    critic1_loss = F.mse_loss(q1_vals, target_q)  # [critic1 loss]
    critic2_loss = F.mse_loss(q2_vals, target_q)  # [critic2 loss]

    # Update critic 1
    critic1_optimizer.zero_grad()
    critic1_loss.backward()
    torch.nn.utils.clip_grad_value_(main_ac.q1.parameters(), 100.0)  # [clip grads q1]
    critic1_optimizer.step()

    # Update critic 2
    critic2_optimizer.zero_grad()
    critic2_loss.backward()
    torch.nn.utils.clip_grad_value_(main_ac.q2.parameters(), 100.0)  #[clip grads q2]
    critic2_optimizer.step()

    # ----- Delayed Actor update -----
    global td3_update_step  #[use global delayed update counter]
    td3_update_step += 1    #[increment update counter]
    actor_loss_val = None   #[placeholder for returned actor loss or None]
    if td3_update_step % POLICY_DELAY == 0:  #[perform actor update every POLICY_DELAY steps]
        # Actor objective: maximize Q(s, pi(s)) => minimize -mean(Q(s, pi(s)))
        actor_actions = main_ac.pi(state_batch)  # [B, act_dim]
        # Use q1 (it is common to use either q1 or min, Spinning Up uses q1)
        actor_loss = -main_ac.q1(state_batch, actor_actions).mean()  #[actor loss uses q1]
        actor_optimizer.zero_grad() 
        actor_loss.backward()
        torch.nn.utils.clip_grad_value_(main_ac.pi.parameters(), 100.0)
        actor_optimizer.step()
        actor_loss_val = actor_loss.item()  #[store actor loss scalar]

        # Soft update (τ = TAU) for all target networks: actor and both critics
        target_ac_state_dict = target_ac.state_dict()
        main_ac_state_dict = main_ac.state_dict()
        for key in main_ac_state_dict:
            target_ac_state_dict[key] = main_ac_state_dict[key]*TAU + target_ac_state_dict[key]*(1-TAU)
        target_ac.load_state_dict(target_ac_state_dict)

    # Return some helpful logging values
    mean_q_current = 0.5 * (q1_vals.mean().item() + q2_vals.mean().item())  #[return average of q1 and q2]
    return mean_q_current, actor_loss_val         # mean of q values and actor loss (if updated) are returned for logging

    
print("Training Started")
print(f"Total Number of Episodes: {num_episodes}")
print(f"Number of Exploratory Episodes: {exploration_episodes}\n")

for i_episode in range(num_episodes):    
    exploration_flag = i_episode < exploration_episodes
    
    total_reward = 0
    episode_steps = 0
    mean_q_values = [] 
    mean_actor_loss = []

    state, info = env.reset(start_upright=START_UPRIGHT)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    upright_flag = False
    
    for t in count():
        action = env.action_space.sample() if exploration_flag else select_action(state, Add_Noise=True)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Store transition
        memory.push(state, action, next_state, reward)
        state = next_state

        # One optimization step
        log_data = optimize_model()  
        
        if log_data is not None:
            q_val, actor_loss = log_data
            mean_q_values.append(q_val)
            mean_actor_loss.append(actor_loss)

        total_reward += reward.item()

        if done:
            episode_reward.append(total_reward)
            episode_steps = t
            plot_reward()
            upright_flag = env.reached_upright
            break

    # Compute mean Q and actor loss for printing
    avg_q = np.mean(mean_q_values) if len(mean_q_values) > 0 else 0.0
    
    valid_losses = [loss for loss in mean_actor_loss if loss is not None]
    avg_loss = np.mean(valid_losses) if len(valid_losses) > 0 else 0.0


    if i_episode % 5 == 0:
        noise_scale = NOISE_END + (NOISE_START - NOISE_END) * \
            math.exp(-1. * steps_done / NOISE_DECAY)
        # logged information:
        if START_UPRIGHT:
            print(f"Episode {i_episode:4d} | Steps: {steps_done:6d} | "
                f"Noise Scale: {noise_scale:.3f} | Replay size: {len(memory):5d} | "
                f"Episode length: {episode_steps:3d} | Total reward: {total_reward:7.2f} | "
                f"Mean Q-value: {avg_q:7.3f} | Mean Actor Loss: {avg_loss:7.3f}")
        else:
            print(f"Episode {i_episode:4d} | Steps: {steps_done:6d} | "
                f"Noise Scale: {noise_scale:.3f} | Replay size: {len(memory):5d} | "
                f"Reached Upright: {upright_flag} | Total reward: {total_reward:7.2f} | "
                f"Mean Q-value: {avg_q:7.3f} | Mean Actor Loss: {avg_loss:7.3f}")
    

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
state, _ = env.reset(start_upright=START_UPRIGHT)

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
    torch.save(target_ac,"Trained_Networks/TD3/Cartpole/" + name + "_target.pth")
    torch.save(main_ac,"Trained_Networks/TD3/Cartpole/" + name + "_main.pth")
    torch.save(target_ac.state_dict(),"Trained_Networks/TD3/Cartpole/state_dicts/" + name + "_target.pth")
    torch.save(main_ac.state_dict(),"Trained_Networks/TD3/Cartpole/state_dicts/" + name + "_main.pth")
