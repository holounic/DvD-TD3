import numpy as np
from tools import make_env


def play_episode(agent, env_name):
    env = make_env(env_name)
    done = False
    state = env.reset()
    reward_accum = 0
    while not done:
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        reward_accum += reward
    return reward_accum


def test(agent, env_name, num_tests=10):
    rewards = []
    for _ in range(num_tests):
        rewards.append(play_episode(agent, env_name))
    return np.mean(rewards).item(), np.std(rewards).item()
