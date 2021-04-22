import torch
import gym
import pybullet_envs
import numpy as np
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
writer = SummaryWriter()
project_folder = __file__[:-15]


def make_env(env_name):
    return gym.make(env_name)


def get_env_info(env_name):
    result = {}
    env = make_env(env_name)
    result['env'] = env
    result['state_dim'] = env.observation_space.shape[0]
    result['action_dim'] = env.action_space.shape[0]
    return result


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


def unpack_batch(batch):
    state, action, reward, next_state, terminal = batch
    state = torch.tensor(np.array(state), dtype=torch.float32, device=device)
    action = torch.tensor(np.array(action), dtype=torch.float32, device=device)
    reward = torch.tensor(np.array(reward), dtype=torch.float32, device=device).view(-1, 1)
    next_state = torch.tensor(np.array(next_state), dtype=torch.float32, device=device)
    terminal = torch.tensor(np.array(terminal), device=device).view(-1, 1)
    return state, action, reward, next_state, terminal


def soft_update(source_net, target_net, rho):
    for source_param, target_param in zip(source_net.parameters(), target_net.parameters()):
        target_param.data.mul_(1 - rho)
        target_param.data.add_(rho * source_param.data)
