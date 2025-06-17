# Application of Q learning for the swing up task in a simple pendulum system
# based on the paper by Alessio Ghio and Oscar E. Ramos 2019

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
from random import uniform
import torch
import matplotlib.pyplot as plt

def wrap_to_pi(theta):
    return ((theta + np.pi) % (2 * np.pi)) - np.pi

class SwingUpPendulum(gym.Env):
    def __init__(self):
        super().__init__()
        self.m = 1.0
        self.L = 1.0
        self.b = 0.01
        self.g = 9.8
        self.dt = 0.025

        self.valid_actions = np.array([-9.0, 0.0, 9.0])
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(
            low=np.array([0.0, -np.inf], dtype=np.float32),
            high=np.array([2 * np.pi, np.inf], dtype=np.float32),
            dtype=np.float32
        )

        self.state = None
        self.current_it = 0
        self.max_it = 800

    def reset(self, seed=None, options=None, Upright=False):
        super().reset(seed=seed)
        self.current_it = 0
        if Upright:
            angle = np.pi + uniform(-0.1, 0.1)
            self.valid_actions = np.array([-0.5, 0.5])
            self.stage = 2
            self.max_it = 300
        else:
            angle = uniform(-0.2, 0.2)
            self.valid_actions = np.array([-9.0, 0.0, 9.0])
            self.stage = 1
            self.max_it = 800

        self.state = np.array([angle, 0.0], dtype=np.float32)
        return self.state, {}

    def dynamics(self, t, y, u):
        x1, x2 = y
        dx1 = x2
        dx2 = (u - self.b * x2) / (self.m * self.L ** 2) - self.g * np.sin(x1) / self.L
        return [dx1, dx2]

    def step(self, action_idx):
        self.current_it += 1
        u = float(self.valid_actions[action_idx])
        sol = solve_ivp(self.dynamics, [0, self.dt], self.state, args=(u,), t_eval=[self.dt])
        self.state = sol.y[:, -1]
        self.state[0] = self.state[0] % (2 * np.pi)

        terminated = False
        truncated = False
        theta, theta_dot = self.state
        theta_error = wrap_to_pi(theta - np.pi)

        if self.stage == 1:
            reward = -(theta_error ** 2 + 0.2 * theta_dot ** 2)
            objective = abs(theta_error) < 0.01 and abs(theta_dot) < 0.05
            if objective:
                reward += 100
        else:
            reward = -10 * abs(theta_error)
            objective = abs(theta_error) < 0.05
            if abs(theta_error) >= np.pi / 2:
                terminated = True
                reward -= 10
            if objective:
                reward += 100

        if self.current_it >= self.max_it:
            truncated = True

        return self.state, reward, terminated, truncated, {}

# Feature maps

def featuremap1(state, action):
    t, t_dot = wrap_to_pi(state[0]), state[1]
    a = action
    return torch.tensor([
        1.0, 1.0, t ** 2, t, t_dot ** 2, t_dot,
        (a * t) ** 2, a * t, (a * t_dot) ** 2, a * t_dot
    ], dtype=torch.float32)

def featuremap2(state, action):
    t, t_dot = wrap_to_pi(state[0]), state[1]
    a = action
    return torch.tensor([
        1.0, t ** 2, t, t_dot ** 2, t_dot, a * (np.pi - t), a * t_dot
    ], dtype=torch.float32)

# Linear Q-function
class LinearQFunction(torch.nn.Module):
    def __init__(self, n_features, feature_fn):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1, bias=False)
        self.feature_fn = feature_fn

    def forward(self, state, action):
        x = self.feature_fn(state, action)
        return self.linear(x)

# TD update

def td_update(Q, state, action, reward, next_state, alpha, gamma, valid_actions):
    with torch.no_grad():
        q_val = Q(state, action)
        if next_state is None:
            target = reward
        else:
            next_qs = [Q(next_state, torch.tensor([a])) for a in valid_actions]
            target = reward + gamma * torch.max(torch.stack(next_qs))

        delta = target - q_val
        delta = torch.clamp(delta, -10.0, 10.0)
        x = Q.feature_fn(state, action)
        Q.linear.weight.data += alpha * delta * x.unsqueeze(0)

# Training stages

def train_stage(stage):
    env = SwingUpPendulum()
    Q = LinearQFunction(10 if stage == 1 else 7, featuremap1 if stage == 1 else featuremap2)
    Q.linear.weight.data.zero_()
    episodes = 2000 if stage == 1 else 5000
    alpha = 1e-6 if stage == 1 else 5e-4
    gamma = 0.99 if stage == 1 else 0.7
    EPS_START = 0.45
    decay = 0.999 if stage == 1 else 0.99
    valid_actions = np.array([-9.0, 0.0, 9.0]) if stage == 1 else np.array([-0.5, 0.5])
    upright = False if stage == 1 else True
    episode_lengths = []
    epsilon = EPS_START

    for ep in range(episodes):
        if stage == 1:
            epsilon = EPS_START  # reset epsilon each episode for stage 1 only

        state_np, _ = env.reset(Upright=upright)
        state = torch.tensor(state_np, dtype=torch.float32)

        for t in range(env.max_it):
            if np.random.rand() > epsilon:
                qs = [Q(state, torch.tensor([a])) for a in valid_actions]
                action_idx = torch.argmax(torch.stack(qs)).item()
            else:
                action_idx = np.random.randint(len(valid_actions))

            action_val = valid_actions[action_idx]
            next_state_np, reward, terminated, truncated, _ = env.step(action_idx)
            reward = torch.tensor([reward])

            if terminated or truncated:
                next_state = None
            else:
                next_state = torch.tensor(next_state_np, dtype=torch.float32)

            if stage == 1:
                theta, theta_dot = env.state
                theta_error = wrap_to_pi(theta - np.pi)
                if abs(theta_error) < 0.01 and abs(theta_dot) < 0.05:
                    print('Stage 2 has been reached')
            
            td_update(Q, state, torch.tensor([action_val]), reward, next_state,
                      alpha, gamma, valid_actions)
            
            state = next_state if next_state is not None else state
            if terminated or truncated:
                break

        episode_lengths.append(t + 1)
        epsilon *= decay

        if ep % (episodes // 20) == 0:
            print(f"Stage {stage} Progress: {ep / episodes * 100:.1f}%")

    return Q

if __name__ == '__main__':
    
    yn1 = input("Train weights of stage 1? (y/n)")
    yn2 = input("Train weights of stage 2? (y/n)")
    if yn1 == 'y':
        Q1 = train_stage(stage=1)
        torch.save(Q1,"LinearActionValueF_Stage1.pth")
        print("Training complete.")
    else:
        Q1 = torch.load("LinearActionValueF_Stage1.pth", weights_only=False)

    if yn2 == 'y':
        Q2 = train_stage(stage=2)
        torch.save(Q2,"LinearActionValueF_Stage2.pth")
        print("Training complete.")
    else:
        Q2 = torch.load("LinearActionValueF_Stage1.pth", weights_only=False)

    
    

    # ------------------------
    # Testing combined policy
    # ------------------------
    env = SwingUpPendulum()
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    stage2 = False

    thetas, theta_dots, torques = [], [], []
    valid_actions1 = np.array([-9.0,0.0,9.0])
    valid_actions2 = np.array([-0.5,0.5])

    for _ in range(1100):
        Q = Q2 if stage2 else Q1
        valid_actions = valid_actions2 if stage2 else valid_actions1
        qs = [Q(state, torch.tensor([a])) for a in valid_actions]
        action_idx = torch.argmax(torch.stack(qs)).item()

        next_state, reward, terminated, truncated, _ = env.step(action_idx)
        thetas.append(next_state[0])
        theta_dots.append(next_state[1])
        torques.append(valid_actions[action_idx]/9)

        #the conditions to transitioning to the second stage is looser here
        objective = abs(next_state[0] - np.pi) < 0.08 and abs(next_state[1])<5 
        if objective:
            stage2 = True
            env.stage = 2
            env.state = next_state

        if stage2 and abs(next_state[0] - np.pi) >= np.pi / 2:
            break

        if terminated or truncated:
            break

        state = torch.tensor(next_state, dtype=torch.float32)

    plt.figure(figsize=(10, 4))
    plt.plot(thetas, label='θ')
    plt.plot(theta_dots, label='ω')
    plt.plot(torques, label='τ')
    plt.xlabel('Time step')
    plt.ylabel('State value')
    plt.title('State Variables Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
