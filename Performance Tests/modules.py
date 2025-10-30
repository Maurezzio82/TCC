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

class TD3_ActorCritic(nn.Module):

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
        
class DDPG_ActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,128),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = Actor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = QFunction(obs_dim, act_dim, hidden_sizes, activation)
        
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