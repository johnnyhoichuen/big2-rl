import torch
import numpy as np

from big2_rl.env.env import get_obs
from big2_rl.deep_mc.model import Big2ModelResNet, Big2Model


class DMCAgent:
    def __init__(self, model_path, model_type):
        """
        Loads model's pretrained weights from a given model path.
        """
        if model_type == 'residual':
            self.model = Big2ModelResNet()
        elif model_type == 'conv':
            self.model = Big2Model()  # TODO
        else:
            self.model = Big2Model()

        model_state_dict = self.model.state_dict()
        if torch.cuda.is_available():
            pretrained_weights = torch.load(model_path, map_location='cuda:0')
        else:
            pretrained_weights = torch.load(model_path, map_location='cpu')
        # replace default state dict with pretrained weights)
        if model_path.endswith(".ckpt"):
            model_state_dict.update(pretrained_weights)
        else:
            model_state_dict.update(pretrained_weights["model_state_dict"])

        self.model.load_state_dict(model_state_dict)
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

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

    def recommend(self, infoset):
        """
        Given an infoset, takes z_batch and x_batch features and computes forward pass of model to get top k moves
        according to estimated Q(s,a). Returns a list of tuples [(action1: value1), (action2: value2), ...] where action
        is an int corresponding to the index of the appropriate legal action and value is an int
        Unless: If only one action legal, then return value as 'N/A'
        """
        if len(infoset.legal_actions) == 1:  # default case
            return [(0, 'N/A')]
        obs = get_obs(infoset)
        z_batch = torch.from_numpy(obs['z_batch']).float()
        x_batch = torch.from_numpy(obs['x_batch']).float()
        if torch.cuda.is_available():
            z_batch, x_batch = z_batch.cuda(), x_batch.cuda()
        y_pred = self.model.forward(z_batch, x_batch, return_value=True)['values'].detach().cpu().numpy()

        # http://seanlaw.github.io/2020/01/03/finding-top-or-bottom-k-in-a-numpy-array/
        top_k = min(5, y_pred.shape[0])  # number of results to return. Adjust '5'.
        idx = np.argpartition(y_pred, -top_k, axis=0)[-top_k:]
        idx_sorted = idx[np.argsort(y_pred[idx], axis=None)][::-1].reshape(top_k,)
        actions_and_values = []
        for action_ind in idx_sorted:
            value = y_pred[action_ind][0]
            actions_and_values.append((action_ind, value))
        return actions_and_values
