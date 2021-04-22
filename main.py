import json
from tools import get_env_info
from td3.algorithm import DvDTD3
from td3.noise import ScheduledNoise, Noise
from utils.train import train

with open('config.json', 'r') as source:
    config = json.load(source)

if __name__ == '__main__':
    env_name = config['env_name']
    algo_kwargs = config['algo_kwargs']
    train_kwargs = config['train_kwargs']

    env_info = get_env_info(env_name)
    noise = Noise(mean=0, std=0.2, clip=0.5)
    agent = DvDTD3(env_info['state_dim'], env_info['action_dim'], **algo_kwargs)
    train(env_name, agent, noise=noise, **train_kwargs)