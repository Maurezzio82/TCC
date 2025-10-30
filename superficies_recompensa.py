import numpy as np
import matplotlib.pyplot as plt

# ===== Reward definitions =====
def old_reward(x, x_dot, theta, theta_dot, u):
    cost = 3.0*x**2 + 0.1*x_dot**2 + 2*theta**2 + 0.2*theta_dot**2 + 0.01*u**2
    return 10 - cost/2

def new_reward(x, x_dot, theta, theta_dot, u):
    cost = 3.0*x**2 + 0.1*x_dot**2 + 2*theta**2 + 0.2*theta_dot**2 + 0.01*u**2
    reward = 1 - 0.02*cost
    return np.clip(reward, -1.0, 1.0)

# ===== Create grid =====
x_vals = np.linspace(-5, 5, 200)
theta_vals = np.linspace(-np.pi, np.pi, 200)
X, THETA = np.meshgrid(x_vals, theta_vals)

# ===== Compute rewards =====
R_old = old_reward(X, 0, THETA, 0, 0)
R_new = new_reward(X, 0, THETA, 0, 0)

# ===== Plot configuration =====
fig, axs = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': '3d'})

# Common limits
x_lim = (-5, 5)
theta_lim = (-np.pi, np.pi)
z_lim = (-1.5, 10.5)  # to compare on the same scale

# ===== Old reward surface =====
surf1 = axs[0].plot_surface(X, THETA, R_old, cmap='viridis', edgecolor='none')
axs[0].set_title("Old Reward: 10 - cost/2", fontsize=12)
axs[0].set_xlabel("x (cart position)")
axs[0].set_ylabel("θ (radians)")
axs[0].set_zlabel("Reward")
axs[0].set_xlim(x_lim)
axs[0].set_ylim(theta_lim)
axs[0].set_zlim(z_lim)
fig.colorbar(surf1, ax=axs[0], shrink=0.6)

# ===== New reward surface =====
surf2 = axs[1].plot_surface(X, THETA, R_new, cmap='plasma', edgecolor='none')
axs[1].set_title("New Reward: 1 - 0.02*cost (clipped)", fontsize=12)
axs[1].set_xlabel("x (cart position)")
axs[1].set_ylabel("θ (radians)")
axs[1].set_zlabel("Reward")
axs[1].set_xlim(x_lim)
axs[1].set_ylim(theta_lim)
axs[1].set_zlim(z_lim)
fig.colorbar(surf2, ax=axs[1], shrink=0.6)

plt.tight_layout()
plt.show()



"""
    
for i_episode in range(num_episodes):    
    #set exploration flag
    exploration_flag = i_episode < exploration_episodes
    #upright_to_start = random.choice([True,False]) if exploration_flag else False
    
    #variable for ploting total reward x episode number
    total_reward = 0
    episode_steps = 0
    #Initialize the environment and get its state
    state, info = env.reset(start_upright=True)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    for t in count():
        action = env.action_space.sample() if exploration_flag else select_action(state, Add_Noise=True)
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
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_ac_state_dict = target_ac.state_dict()
        main_ac_state_dict = main_ac.state_dict()
        for key in main_ac_state_dict:
            target_ac_state_dict[key] = main_ac_state_dict[key]*TAU + target_ac_state_dict[key]*(1-TAU)
        target_ac.load_state_dict(target_ac_state_dict)

        total_reward += reward.item()
        if done:
            #if env.reached_upright and not upright_to_start:
            #    print('Pendulum reached upright position')
            #    print(total_reward.item())
            episode_reward.append(total_reward)
            episode_steps = t
            plot_reward()
            break
    if i_episode % 20 == 0:
        # You can recompute eps_threshold here using steps_done
        noise_scale = NOISE_END + (NOISE_START - NOISE_END) * \
            math.exp(-1. * steps_done / NOISE_DECAY)
        print(f"Episode {i_episode:4d} | Steps: {steps_done:6d} | "
            f"Noise Scale: {noise_scale:.3f} | Replay size: {len(memory):5d} | "
            f"Episode length: {episode_steps:3d} | Total reward: {total_reward:.2f} | "
            f"Mean Q-Value: {env.reached_upright}")
"""
