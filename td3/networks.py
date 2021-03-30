import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_layer(layer, limit=0.1):
    layer.weight = nn.Parameter(
        data=torch.distributions.uniform.Uniform(-limit, limit).sample(layer.weight.shape), requires_grad=True)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hid_size1=400, hid_size2=300, init_layers=False):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hid_size1)
        self.fc2 = nn.Linear(hid_size1, hid_size2)
        self.fc3 = nn.Linear(hid_size2, action_dim)

        if init_layers:
            weight_limit = 1. / (state_dim * state_dim)
            initialize_layer(self.fc1, weight_limit)
            initialize_layer(self.fc2, weight_limit)
            initialize_layer(self.fc3, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hid_size1=400, hid_size2=300, init_layers=False):
        super(Critic, self).__init__()
        self.fc1_1 = nn.Linear(state_dim + action_dim, hid_size1)
        self.fc2_1 = nn.Linear(hid_size1, hid_size2)
        self.fc3_1 = nn.Linear(hid_size2, 1)

        self.fc1_2 = nn.Linear(state_dim + action_dim, hid_size1)
        self.fc2_2 = nn.Linear(hid_size1, hid_size2)
        self.fc3_2 = nn.Linear(hid_size2, 1)

        if init_layers:
            weight_limit = 1. / ((state_dim + action_dim) * (state_dim + action_dim))
            initialize_layer(self.fc1_1, weight_limit)
            initialize_layer(self.fc2_1, weight_limit)
            initialize_layer(self.fc3_1, 3e-3)
            initialize_layer(self.fc1_2, weight_limit)
            initialize_layer(self.fc2_2, weight_limit)
            initialize_layer(self.fc3_2, 3e-3)

    def Q1(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1_1(x))
        x = F.relu(self.fc2_1(x))
        x = self.fc3_1(x)
        return x

    def Q2(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1_2(x))
        x = F.relu(self.fc2_2(x))
        x = self.fc3_2(x)
        return x

    def forward(self, state, action):
        return self.Q1(state, action), self.Q2(state, action)
