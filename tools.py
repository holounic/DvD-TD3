import torch
import gym
import pybullet_envs
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
writer = SummaryWriter()
project_folder = __file__[:-9]


def make_env(env_name):
    return gym.make(env_name)


def get_env_info(env_name):
    result = {}
    env = make_env(env_name)
    result['env'] = env
    result['state_dim'] = env.observation_space.shape[0]
    result['action_dim'] = env.action_space.shape[0]
    return result
