import os
import torch
import torch.nn as nn
import torch.nn.functional as F


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

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, n_l1 = 128, n_l2 = 128):
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


# === User configuration ===
SOURCE_DIR = "Trained_Networks/DQN/Pendulum"
TARGET_DIR = os.path.join(SOURCE_DIR, "state_dicts")

# === Make sure the target folder exists ===
os.makedirs(TARGET_DIR, exist_ok=True)

# === Loop through all .pth files ===
for file_name in os.listdir(SOURCE_DIR):
    if file_name.endswith(".pth"):
        file_path = os.path.join(SOURCE_DIR, file_name)
        print(f"Processing: {file_name}")

        try:
            # Load the full model (you'll define the proper classes yourself)
            model = torch.load(file_path, map_location="cpu", weights_only= False)

            # Extract and save the state_dict
            state_dict = model.state_dict()
            save_path = os.path.join(TARGET_DIR, file_name)
            torch.save(state_dict, save_path)

            print(f"✅ Saved state_dict to: {save_path}")

        except Exception as e:
            print(f"❌ Failed to process {file_name}: {e}")

print("\nDone! All valid .pth files converted to state_dicts.")
