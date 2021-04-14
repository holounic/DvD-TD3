import numpy as np
from tools import make_env, writer
from utils.test import test
from td3.noise import Noise


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


def acquire_knowledge(env, agent, state, noise, min_action, max_action, step, start_train):
    if step > start_train:
        action = agent.act(state)
        action = np.clip(action + noise.sample(action.shape), min_action, max_action)
    else:
        action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    return state, action, reward, next_state, done


def train(env_name, population, min_action=-1, max_action=1, timesteps=int(5e5), start_train=int(1e4), actor_delay=int(2),
          num_tests=10, batch_size=100, noise=None, save_every=int(5e4), log_every=int(5e3)):
    population_size = len(population.learners)
    envs, states, next_states = [], [], []

    for i in range(population_size):
        env = make_env(env_name)
        env.seed(i)
        envs.append(env)
        states.append(env.reset())

    actor_loss_accum, critic_loss_accum, episodes = 0, 0, 0

    if noise is None:
        noise = Noise()

    for step in range(timesteps):
        for (agent, env, state) in zip(population.learners, envs, states):
            state, action, reward, next_state, done = \
                acquire_knowledge(env, agent, state, noise, min_action, max_action, step, start_train)
            population.save_transition(state, action, reward, next_state, done)
            if done:
                next_states.append(env.reset())
            else:
                next_states.append(next_state)
        states = next_states
        next_states = []

        if step > start_train:
            population.update(batch_size, step, actor_delay)

            if step % log_every == 0 or step == timesteps - 1:
                population.learners[0].eval()
                reward_mean, reward_std = test(population.learners[0], env_name, num_tests)
                log_train_info(actor_loss_accum / actor_delay, critic_loss_accum / actor_delay, reward_mean, reward_std, step)
                actor_loss_accum, critic_loss_accum = 0, 0

            # if step % save_every == 0 or step == timesteps - 1:
            #     agent.save(env_name)