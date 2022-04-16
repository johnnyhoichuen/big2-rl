import torch
from torch import nn
from big2_rl.env.env import get_obs
import numpy as np
import joblib


# a Pytorch re-implementation of Henry Charlesworth's network in Big2PPO
class PPONetworkPytorch(nn.Module):

    def __init__(self, obs_dim, act_dim, device=0):
        super().__init__()
        if not device == "cpu":  # move device to right parameter
            device = 'cuda:' + str(device)
            self.to(torch.device(device))

        self.dense1 = nn.Linear(obs_dim, 512)
        self.dense2a = nn.Linear(512, 256)
        self.dense2b = nn.Linear(512, 256)
        self.dense3b = nn.Linear(256, 1)
        self.dense3a = nn.Linear(256, act_dim)

    def forward(self, y, a):  # y = input state, a = available actions (apply as mask to output of layer 3a)
        y = torch.relu(self.dense1(y))
        y_1 = torch.relu(self.dense3a(torch.relu(self.dense2a(y))))
        value = torch.relu(self.dense3b(torch.relu(self.dense2b(y))))
        out_action = torch.nn.Softmax(a + y_1)
        # see: http://aaai-rlg.mlanctot.info/papers/AAAI19-RLG-Paper02.pdf
        # input_state_value: value of the given input state. A 1*1 tensor.
        # out_action: probability weighting of each potentially allowable move. A 1695*1 tensor.
        # We ensure no illegal moves can be made by having a (Available actions) come as a mask with all legal actions 0
        # and all illegal actions -inf.
        return {'input_state_value': value, 'out_action': out_action}


class PPOAgent:
    def __init__(self):
        """
        Loads model's pretrained weights from a given model path.
        """
        obs_dim, act_dim = 412, 1695

        # load the Big2PPO model's pretrained parameters
        self.model_pytorch = PPONetworkPytorch(obs_dim, act_dim)
        params = joblib.load("modelParameters136500")  # directly downloaded from Charlesworth's github repo
        for i, np_param in enumerate(params):
            param = torch.from_numpy(np_param)
            if i == 0:
                self.model_pytorch.dense1.weight = torch.nn.Parameter(param)
            elif i == 1:
                self.model_pytorch.dense1.bias = torch.nn.Parameter(param)
            elif i == 2:
                self.model_pytorch.dense2a.weight = torch.nn.Parameter(param)
            elif i == 3:
                self.model_pytorch.dense2a.bias = torch.nn.Parameter(param)
            elif i == 4:
                self.model_pytorch.dense3a.weight = torch.nn.Parameter(param)
            elif i == 5:
                self.model_pytorch.dense3a.bias = torch.nn.Parameter(param)
            elif i == 6:
                self.model_pytorch.dense2b.weight = torch.nn.Parameter(param)
            elif i == 7:
                self.model_pytorch.dense2b.bias = torch.nn.Parameter(param)
            elif i == 8:
                self.model_pytorch.dense3b.weight = torch.nn.Parameter(param)
            elif i == 9:
                self.model_pytorch.dense3b.bias = torch.nn.Parameter(param)
            #print("{}, shape: {}" .format(i, param.shape))

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
        # TODO convert z_batch, x_batch to y, a
        if torch.cuda.is_available():
            y, a = y.cuda(), a.cuda()
        pred_action = self.model_pytorch.forward(y, a)['out_action'].detach().cpu().numpy()
        best_action_index = np.argmax(pred_action, axis=0)[0]
        best_action = infoset.legal_actions[best_action_index]
        return best_action
