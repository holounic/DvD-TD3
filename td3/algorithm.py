from utils.base import BaseActorCritic
from td3.networks import Actor, Critic
import torch.optim as optim
from utils.tools import device, soft_update, unpack_batch
from td3.buffer import Buffer
import torch
import torch.nn.functional as F
from diversity.loss import DiversityLoss


class TD3(BaseActorCritic):
    def __init__(self, state_dim, action_dim, gamma=0.99, rho=0.005, min_action=-1, max_action=1, noise_std=0.1,
                 noise_clip=0.5, actor_lr=1e-3, critic_lr=1e-3, actor_wd=0, critic_wd=0, buffer_size=int(1e5),
                 actor_dims=None, critic_dims=None):
        BaseActorCritic.__init__(self)

        self.gamma = gamma
        self.rho = rho
        self.min_action = min_action
        self.max_action = max_action
        self.noise_std = noise_std
        self.noise_clip = noise_clip

        self.actor = Actor(state_dim, action_dim, actor_dims)
        self.target_actor = Actor(state_dim, action_dim, actor_dims)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim, critic_dims)
        self.target_critic = Critic(state_dim, action_dim, critic_dims)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=actor_wd)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=critic_wd)

        self.actor.to(device)
        self.target_actor.to(device)
        self.critic.to(device)
        self.target_critic.to(device)

        self.target_actor.eval()
        self.target_critic.eval()

        self.buffer = Buffer(buffer_size)

    def compute_target_action(self, next_state):
        with torch.no_grad():
            action = self.target_actor(next_state)
            noise = torch.empty_like(action).data.normal_(0, self.noise_std).to(device).clamp(-self.noise_clip,
                                                                                              self.noise_clip)
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
        soft_update(self.actor, self.target_actor, self.rho)

    def update_target_critic(self):
        soft_update(self.critic, self.target_critic, self.rho)

    def update(self, batch_size, step, actor_delay):
        state, action, reward, next_state, terminal = self._unpack_batch(self.buffer.sample(batch_size))
        self._prepare_update(state, action, reward, next_state, terminal)

        critic_loss = self.update_critic(state, action, reward, next_state, terminal)
        actor_loss = torch.zeros(1)

        if step % actor_delay == 0:
            actor_loss = self.update_actor(state)

            self.update_target_actor()
            self.update_target_critic()

        return actor_loss, critic_loss

    def save_transition(self, state, action, reward, next_state, terminal):
        self.buffer.save_transition((state, action, reward, next_state, terminal))

    def _extract_action(self, x):
        return x

    def _get_name(self):
        return 'td3'


class DvDTD3:
    def __init__(self, state_dim, action_dim, population_size=5, diversity_loss=DiversityLoss(),
                 diversity_importance=0.3, gamma=0.99, rho=0.005, min_action=-1, max_action=1, noise_std=0.1,
                 noise_clip=0.5, actor_lr=1e-3, critic_lr=1e-3, actor_wd=0, critic_wd=0, buffer_size=int(1e6),
                 actor_dims=None, critic_dims=None):
        self.diversity_loss = diversity_loss
        self.diversity_importance = diversity_importance
        self.buffer = Buffer(buffer_size)
        self.gamma = gamma

        self.population = []
        population_params = []

        self.critic = Critic(state_dim, action_dim, critic_dims)
        self.target_critic = Critic(state_dim, action_dim, critic_dims)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=critic_wd)

        self.critic.to(device)
        self.target_critic.to(device)

        self.target_critic.eval()

        for _ in range(population_size):
            agent = TD3(state_dim, action_dim, gamma, rho, min_action, max_action, noise_std, noise_clip, actor_lr,
                        critic_lr, actor_wd, critic_wd, 1, actor_dims, critic_dims)
            population_params = population_params + list(agent.actor.parameters())
            agent.buffer = self.buffer
            agent.critic = self.critic
            agent.target_critic = self.target_critic
            agent.critic_optim = None
            self.population.append(agent)

        self.population_optim = optim.Adam(population_params, lr=actor_lr, weight_decay=actor_wd)

    def compute_population_loss(self, state):
        embeddings = [agent.actor(state).flatten() for agent in self.population]
        embeddings = torch.stack(embeddings)
        return self.diversity_loss(embeddings)

    def update_critic(self, states, actions, rewards, next_states, terminals):
        all_states = torch.cat(states)
        all_actions = torch.cat(actions)
        all_rewards = torch.cat(rewards)
        all_next_states = torch.cat(next_states)
        all_terminals = torch.cat(terminals)
        predicted_q1, predicted_q2 = self.critic(all_states, all_actions)
        target_actions = torch.cat([agent.compute_target_action(next_state) for agent, next_state in zip(self.population, next_states)])
        with torch.no_grad():
            target_q = torch.minimum(*self.target_critic(all_next_states, target_actions))
            target_q = all_rewards + self.gamma * target_q * all_terminals.logical_not()
        loss = F.mse_loss(predicted_q1, target_q) + F.mse_loss(predicted_q2, target_q)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        return loss

    def update(self, batch_size, step, actor_delay):
        critic_losses, actor_losses = 0, 0
        self.train()
        states, actions, rewards, next_states, terminals = \
            list(zip(*[unpack_batch(self.buffer.sample(batch_size)) for _ in range(len(self.population))]))
        critic_losses = self.update_critic(states, actions, rewards, next_states, terminals)

        if step % actor_delay == 0:
            for agent, state in zip(self.population, states):
                actor_losses = actor_losses + agent.compute_actor_loss(state)

        state = unpack_batch(self.buffer.sample(20))[0]
        if step % actor_delay == 0:
            actor_losses += self.diversity_importance * self.compute_population_loss(state)
            self.population_optim.zero_grad()
            actor_losses.backward()
            self.population_optim.step()

            for actor in self.population:
                actor.update_target_actor()
                actor.update_target_critic()
        return actor_losses, critic_losses

    def train(self):
        for agent in self.population:
            agent.train()

    def eval(self):
        for agent in self.population:
            agent.eval()
