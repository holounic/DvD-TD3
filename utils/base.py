import torch
from utils.tools import device, project_folder, unpack_batch


class BaseActorCritic:
    def __init__(self):
        self.actor, self.critic = None, None

    def _extract_action(self, x):
        raise NotImplementedError()

    def _get_name(self):
        return None

    def _prepare_update(self, state, action, reward, next_state, terminal):
        pass

    def _unpack_batch(self, batch):
        return unpack_batch(batch)

    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def act(self, state):
        state = torch.from_numpy(state).to(device).float()
        with torch.no_grad():
            action = self._extract_action(self.actor(state))
        return action.cpu().numpy().flatten()

    def save(self, env_name):
        torch.save(self.actor.state_dict(), project_folder + f'/params/{env_name}/{self._get_name()}/actor.pkl')
        torch.save(self.critic.state_dict(), project_folder + f'/params/{env_name}/{self._get_name()}/critic.pkl')
        print('=====Model saved=====')