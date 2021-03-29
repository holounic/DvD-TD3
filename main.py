import json
from tools import get_env_info
from td3.algorithm import TD3
from utils.train import train

with open('config.json', 'r') as source:
    config = json.load(source)

if __name__ == '__main__':
    env_name = config['env_name']
    ddpg_kwargs = config['ddpg_kwargs']
    train_kwargs = config['train_kwargs']

    env_info = get_env_info(env_name)

    agent = TD3(env_info['state_dim'], env_info['action_dim'], **ddpg_kwargs)
    train(env_name, agent, **train_kwargs)