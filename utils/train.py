import numpy as np
from tools import make_env, writer
from utils.test import test


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


def train(env_name, agent, timesteps=int(5e6), test_every=int(1e3), start_train=int(1e4), actor_delay=int(1e3),
          num_tests=10, batch_size=128, noise_clip=0.2):
    env = make_env(env_name)
    done = False
    state = env.reset()
    actor_loss_accum, critic_loss_accum, episodes = 0, 0, 0

    for step in range(timesteps):
        if done:
            state = env.reset()
            done = False
            episodes += 1
        action = agent.act(state)
        action = np.clip(action + np.clip(np.random.randn(*action.shape), -noise_clip, noise_clip), -1., 1.)
        next_state, reward, done, _ = env.step(action)
        agent.save_transition(state, action, reward, next_state, done)
        state = next_state

        if step > start_train:
            agent.train()
            batch = agent.buffer.sample(batch_size)
            actor_loss, critic_loss = agent.update(batch, step, actor_delay)
            actor_loss_accum += actor_loss
            critic_loss_accum += critic_loss

            if step % test_every == 0 or step == timesteps - 1:
                agent.eval()
                reward_mean, reward_std = test(agent, env_name, num_tests)
                log_train_info(actor_loss_accum / test_every, critic_loss_accum / test_every, reward_mean, reward_std, step)
                actor_loss_accum, critic_loss_accum = 0, 0
