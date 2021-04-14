import torch
import torch.nn.functional as F
import torch.optim as optim
from td3.buffer import Buffer
from td3.networks import Actor, Critic
from tools import device, project_folder


class TD3:
    def __init__(self, state_dim, action_dim, gamma=0.99, rho=0.005, min_action=-1, max_action=1, noise_std=0.1,
                 noise_clip=0.5, actor_lr=1e-3, critic_lr=1e-3, actor_wd=0, critic_wd=0):
        self.gamma = gamma
        self.rho = rho
        self.min_action = min_action
        self.max_action = max_action
        self.noise_std = noise_std
        self.noise_clip = noise_clip

        self.actor = Actor(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=actor_wd)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=critic_wd)

        self.actor.to(device)
        self.target_actor.to(device)
        self.critic.to(device)
        self.target_critic.to(device)

        self.target_actor.eval()
        self.target_critic.eval()

    def compute_target_action(self, next_state):
        with torch.no_grad():
            action = self.target_actor(next_state)
            noise = torch.empty_like(action).data.normal_(0, self.noise_std).to(device).clamp(-self.noise_clip, self.noise_clip)
            return (action + noise).clamp(self.min_action, self.max_action)

    def compute_actor_loss(self, state):
        loss = - self.critic.Q1(state, self.actor(state)).mean()
        return loss

    def compute_critic_loss(self, state, action, reward, next_state, terminal):
        predicted_q1, predicted_q2 = self.critic(state, action)
        target_action = self.compute_target_action(next_state)
        with torch.no_grad():
            target_q = torch.minimum(*self.target_critic(next_state, target_action))
            target_q = reward + self.gamma * target_q * terminal.logical_not()
        loss = F.mse_loss(predicted_q1, target_q) + F.mse_loss(predicted_q2, target_q)
        return loss

    def update_actor(self, state):
        actor_loss = self.compute_actor_loss(state)

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        return actor_loss.item()

    def update_critic(self, state, action, reward, next_state, terminal):
        critic_loss = self.compute_critic_loss(state, action, reward, next_state, terminal)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        return critic_loss.item()

    def update_target_actor(self):
        self.soft_update(self.actor, self.target_actor)

    def update_target_critic(self):
        self.soft_update(self.critic, self.target_critic)

    def soft_update(self, source_net, target_net):
        for source_param, target_param in zip(source_net.parameters(), target_net.parameters()):
            target_param.data.mul_(1 - self.rho)
            target_param.data.add_(self.rho * source_param.data)

    def update(self, batch, step, actor_delay):
        state, action, reward, next_state, terminal = batch
        state = torch.tensor(state, dtype=torch.float32, device=device)
        action = torch.tensor(action, dtype=torch.float32, device=device)
        reward = torch.tensor(reward, dtype=torch.float32, device=device).view(-1, 1)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
        terminal = torch.tensor(terminal, device=device).view(-1, 1)

        critic_loss = self.update_critic(state, action, reward, next_state, terminal)
        actor_loss = torch.zeros(1)

        if step % actor_delay == 0:
            actor_loss = self.update_actor(state)

            # self.update_target_actor()
            self.update_target_critic()

        return actor_loss, critic_loss

    def act(self, state):
        state = torch.from_numpy(state).to(device).float()
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy().flatten()

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def train(self):
        self.actor.train()
        self.critic.train()

    def save(self, env_name):
        torch.save(self.actor.state_dict(), project_folder + f'/params/{env_name}/actor.pkl')
        torch.save(self.critic.state_dict(), project_folder + f'/params/{env_name}/critic.pkl')
        print('=====Model saved=====')