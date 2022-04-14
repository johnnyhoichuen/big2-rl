import torch
from torch import nn
from big2_rl.env.env import get_obs


class PPONetwork(nn.Module):

    def __init__(self, obs_dim, act_dim, name):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.name = name

        self.dense1 = nn.Linear(obs_dim, 512)
        self.dense2a = nn.Linear(512, 256)
        self.dense2b = nn.Linear(512, 256)
        self.dense3b = nn.Linear(256, 1)
        self.dense3a = nn.Linear(256, act_dim)
        self.dense4a = nn.Linear(act_dim, act_dim)
        self.dense4b = nn.Linear(act_dim, act_dim)

    def forward(self, z, x, return_value=False, flags=None):
        y = 0  # input state
        a = 0  # available actions
        y = torch.relu(self.dense1(y))
        y_1 = torch.relu(self.dense3a(torch.relu(self.dense2a(y))))
        y_2 = torch.softmax([y_1, y_2])


class PPOAgent:
    def __init__(self, model_path):
        """
        Loads model's pretrained weights from a given model path.
        """
        self.model = PPONetwork(sess, 412, 1695, "trainNet")
        params = joblib.load("modelParameters136500")
        self.model.loadParams(params)

    def act(self, infoset):
        """
        Given an infoset available to this agent, takes the z_batch and x_batch features (historical actions
        and current state + action) and computes forward pass of model to get the suggested legal action.
        However, if only one action is legal (pass), then take that action.
        """
        if len(infoset.legal_actions) == 1:  # default case
            return infoset.legal_actions[0]
        obs = get_obs(infoset)
        z_batch = torch.from_numpy(obs['z_batch']).float()
        x_batch = torch.from_numpy(obs['x_batch']).float()
        if torch.cuda.is_available():
            z_batch, x_batch = z_batch.cuda(), x_batch.cuda()
        y_pred = self.model.forward(z_batch, x_batch, return_value=True)['values'].detach().cpu().numpy()

        best_action_index = np.argmax(y_pred, axis=0)[0]
        best_action = infoset.legal_actions[best_action_index]
        return best_action
