import numpy as np
from tools import make_env, writer
from utils.test import test
from td3.noise import Noise
from td3.algorithm import DvDTD3
from bandits.algorithm import BetaBernoulliBandit


def log_train_info(actor_loss, critic_loss, score_mean, score_std, step):
    print('===============================')
    print(f'Step: {step}')
    print(f'Actor loss: {actor_loss}')
    print(f'Critic loss: {critic_loss}')
    print(f'Mean score: {score_mean}')
    print(f'Score std: {score_std}')

    writer.add_scalar('Loss/actor', actor_loss, step)
    writer.add_scalar('Loss/critic', critic_loss, step)
    writer.add_scalar('Score/mean', score_mean, step)
    writer.add_scalar('Score/std', score_std, step)
    writer.flush()


def train(env_name, agent: DvDTD3, min_action=-1, max_action=1, timesteps=int(5e5), start_train=int(1e4),
          use_bandit=True, actor_delay=2, num_tests=15, batch_size=100, noise=None, save_every=int(5e4),
          log_every=int(1e3)):
    actor_loss_accum, critic_loss_accum, episodes = 0, 0, 0

    population_size = len(agent.population)
    envs, states, next_states = [], [], []

    for i in range(population_size):
        env = make_env(env_name)
        env.seed(i)
        envs.append(env)
        states.append(env.reset())

    if noise is None:
        noise = Noise()

    if use_bandit:
        bandit = BetaBernoulliBandit()
    else:
        bandit = None

    for step in range(timesteps):
        for env, member, state in zip(envs, agent.population, states):
            if step > start_train:
                action = member.act(state)
                action = np.clip(action + noise.sample(action.shape), min_action, max_action)
            else:
                action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            member.save_transition(state, action, reward, next_state, done)
            if done:
                next_state = env.reset()
            next_states.append(next_state)
        states = next_states
        next_states = []

        if step > start_train:
            agent.train()
            actor_loss, critic_loss = agent.update(batch_size, step, actor_delay)
            actor_loss_accum += actor_loss
            critic_loss_accum += critic_loss

            agent.eval()
            reward_mean, reward_std = 0, 0

            cur_num_tests = 0
            if step % log_every == 0:
                cur_num_tests = num_tests
            elif use_bandit:
                cur_num_tests = 1

            for member in agent.population:
                cur_reward_mean, cur_reward_std = test(member, env_name, cur_num_tests)
                reward_mean += cur_reward_mean
                reward_std += cur_reward_std

            if step % log_every == 0:
                log_train_info(actor_loss_accum * actor_delay / (log_every * population_size),
                               critic_loss_accum / (population_size * log_every), reward_mean / population_size,
                               reward_std / population_size, step)
                actor_loss_accum, critic_loss_accum = 0, 0
            if use_bandit:
                bandit.update_dist(reward_mean)
                agent.diversity_importance = bandit.sample()

            if step % save_every == 0 or step == timesteps - 1:
                agent.save(env_name)
