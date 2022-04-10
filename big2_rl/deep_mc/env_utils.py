"""
Here, we wrap the original environment to make it easier
to use. When a game is finished, instead of manually resetting
the environment, we do it automatically.
"""
import torch


def _format_observation(obs, device):
    """
    A utility function to process the `obs` dict containing the features returned by Env.get_obs()
    and moves the batched groups of features to CUDA.
    """
    position = obs['position']
    if not device == "cpu":
        device = 'cuda:' + str(device)
    device = torch.device(device)
    x_batch = torch.from_numpy(obs['x_batch']).to(device)
    z_batch = torch.from_numpy(obs['z_batch']).to(device)
    x_no_action = torch.from_numpy(obs['x_no_action'])
    z = torch.from_numpy(obs['z'])
    obs = {'x_batch': x_batch,
           'z_batch': z_batch,
           'legal_actions': obs['legal_actions'],
           }
    return position, obs, x_no_action, z


class Environment:
    def __init__(self, env, device):
        """
        Initialize this environment wrapper. There are 3 layers of abstraction: Environment -> Env -> GameEnv
        1. Environment encapsulates automatic resetting of Env (and subsequently GameEnv) updating of episodic reward
        (episodic reward is not necessarily one deal's worth), and device used.
        2. Env provides interface for DummyAgent to act
        3. GameEnv encodes actual game logic such as computing of player rewards, keeping track of infosets, etc
        """
        self.env = env
        self.device = device
        self.episode_return = None

    def initial(self):
        initial_position, initial_obs, x_no_action, z = _format_observation(self.env.reset(), self.device)
        self.episode_return = torch.zeros(1, 1)  # initialise a 1x1 tensor of zeros to be episode return.
        # episode return is either 1x1 tensor of 0s or a dict of string:int corresponding to position and their rewards
        initial_done = torch.ones(1, 1, dtype=torch.bool)  # 1x1 bool tensor denotes whether current round is over

        return initial_position, initial_obs, dict(
            done=initial_done,
            episode_return=self.episode_return,
            obs_x_no_action=x_no_action,
            obs_z=z,
        )

    def step(self, action):
        obs, reward, done, _ = self.env.step(action)

        self.episode_return = reward  # add reward if game is over
        episode_return = self.episode_return

        if done:  # reset reward back to 1x1 tensor of 0's for the next game
            obs = self.env.reset()
            self.episode_return = torch.zeros(1, 1)

        position, obs, x_no_action, z = _format_observation(obs, self.device)
        done = torch.tensor(done).view(1, 1)

        return position, obs, dict(
            done=done,  # 1x1 tensor of bool
            episode_return=episode_return,  # 1x1 tensor of episode_return for given deal (should be 0 if done=False)
            obs_x_no_action=x_no_action,  # observation set of x_no_action as described in env.get_obs
            obs_z=z,  # observation set of z as described in env.get_obs
        )

    def close(self):
        self.env.close()
