import torch
from td3.algorithm import TD3
from td3.buffer import Buffer
from td3.networks import Critic
import torch.optim as optim
from diversity.loss import DiversityLoss
from tools import device


class Population:
    def __init__(self, state_dim, action_dim, population_size=3, diversity_loss=DiversityLoss(), gamma=0.99, rho=0.005,
                 min_action=-1, max_action=1, noise_std=0.1, noise_clip=0.5, actor_lr=1e-3, critic_lr=1e-3,
                 actor_wd=0, critic_wd=0, buffer_size=int(1e6), shared_critic=False):
        actor_params = []
        self.buffer = Buffer(buffer_size)
        self.diversity_loss = diversity_loss
        self.learners = []
        shared_q = None
        if shared_critic:
            shared_q = Critic(state_dim, action_dim)

        for _ in range(population_size):
            learner = TD3(state_dim, action_dim, gamma, rho, min_action, max_action, noise_std,
                        noise_clip, actor_lr, critic_lr, actor_wd, critic_wd)
            actor_params = actor_params + list(learner.actor.parameters())
            if shared_critic:
                learner.critic = shared_q
                learner.target_critic.load_state_dict(learner.critic.state_dict())
                learner.critic_optim = optim.Adam(learner.critic.parameters(), lr=critic_lr, weight_decay=critic_wd)
                learner.critic.to(device)
                learner.target_critic.to(device)
                learner.target_critic.eval()
            self.learners.append(learner)

        self.diversity_optim = optim.Adam(actor_params, lr=actor_lr, weight_decay=actor_wd)

    def compute_diversity_loss(self, state):
        embeddings = []
        for learner in self.learners:
            embedding = learner.actor(state).flatten()
            embeddings.append(embedding)
        return self.diversity_loss(torch.stack(embeddings))

    def update_population(self, state):
        loss = 0.5 * self.compute_diversity_loss(state)
        self.diversity_optim.zero_grad()
        loss.backward()
        self.diversity_optim.step()
        return loss.item()

    def update(self, batch_size, step, actor_delay):
        for learner in self.learners:
            learner.train()
            learner.update(self.buffer.sample(batch_size), step, actor_delay)

        embedding_state = self.buffer.sample(batch_size)[0]
        embedding_state = torch.tensor(embedding_state).to(device).float()
        self.update_population(embedding_state)

        if step % actor_delay == 0:
            for learner in self.learners:
                learner.update_target_actor()

    def save_transition(self, state, action, reward, next_state, terminal):
        self.buffer.save_transition((state, action, reward, next_state, terminal))
