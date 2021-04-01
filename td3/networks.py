import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hid_size1=400, hid_size2=300):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hid_size1)
        self.fc2 = nn.Linear(hid_size1, hid_size2)
        self.fc3 = nn.Linear(hid_size2, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hid_size1=400, hid_size2=300):
        super(Critic, self).__init__()
        self.fc1_1 = nn.Linear(state_dim + action_dim, hid_size1)
        self.fc2_1 = nn.Linear(hid_size1, hid_size2)
        self.fc3_1 = nn.Linear(hid_size2, 1)

        self.fc1_2 = nn.Linear(state_dim + action_dim, hid_size1)
        self.fc2_2 = nn.Linear(hid_size1, hid_size2)
        self.fc3_2 = nn.Linear(hid_size2, 1)

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
