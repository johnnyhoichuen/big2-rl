import torch
from torch import nn
from big2_rl.env.env import get_ppo_state, get_ppo_action
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
        self.action_softmax = nn.Softmax(dim=0)

    def forward(self, y, a):  # y = input state, a = available actions (apply as mask to output of layer 3a)
        y = self.dense1(y)
        y = torch.relu(y)
        y_1 = torch.relu(self.dense3a(torch.relu(self.dense2a(y))))
        value = torch.relu(self.dense3b(torch.relu(self.dense2b(y))))
        out_action = self.action_softmax(a + y_1)
        # see: http://aaai-rlg.mlanctot.info/papers/AAAI19-RLG-Paper02.pdf
        # input_state_value: value of the given input state. A 1*1 tensor.
        # out_action: probability weighting of each potentially allowable move. A 1695*1 tensor.
        # We ensure no illegal moves can be made by having a (Available actions) come as a mask with all legal actions 0
        # and all illegal actions -inf.
        return dict(input_state_value=value, out_action=out_action)


class PPOAgent:
    def __init__(self):
        """
        Loads model's pretrained weights from a given model path.
        """
        self.obs_dim = 412
        self.act_dim = 1695

        # load the Big2PPO model's pretrained parameters
        self.model_pytorch = PPONetworkPytorch(self.obs_dim, self.act_dim)

        params = joblib.load("modelParameters136500")  # directly downloaded from Charlesworth's github repo
        for i, np_param in enumerate(params):
            param = torch.from_numpy(np_param)
            if i % 2 == 0:  # otherwise, encounters error when pass through forward()
                param = torch.transpose(param, 0, 1)

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

            self.starting_hand = None
            self.action_indices_5, self.inverse_indices_5, \
                self.action_indices_3, self.inverse_indices_3, \
                self.action_indices_2, self.inverse_indices_2 = 0, 0, 0, 0, 0, 0
            self.load_indices_lookup()

    def set_starting_hand(self, hand):
        self.starting_hand = hand  # initialise a starting hand of 13 cards

    def load_indices_lookup(self):
        # idea: given hand (list of ints), get its indices, then get its NN input representation
        self.action_indices_5 = np.zeros((13, 13, 13, 13, 13), dtype=np.int16)
        self.inverse_indices_5 = np.zeros((1287, 5), dtype=np.int16)  # since 13 choose 5 = 1287
        i_5 = 0
        for c_1 in range(0, 9):
            for c_2 in range(c_1 + 1, 10):
                for c_3 in range(c_2 + 1, 11):
                    for c_4 in range(c_3 + 1, 12):
                        for c_5 in range(c_4 + 1, 13):
                            self.action_indices_5[c_1, c_2, c_3, c_4, c_5] = i_5
                            self.inverse_indices_5[i_5] = np.array([c_1, c_2, c_3, c_4, c_5])
                            i_5 += 1

        self.action_indices_3 = np.zeros((13, 13, 13), dtype=np.int16)
        self.inverse_indices_3 = np.zeros((31, 3), dtype=np.int16)
        i_3 = 0
        for c_1 in range(0, 11):
            n_1 = min(c_1 + 2, 11)
            for c_2 in range(c_1 + 1, n_1 + 1):
                n_2 = min(c_1 + 3, 12)
                for c_3 in range(c_2 + 1, n_2 + 1):
                    self.action_indices_3[c_1, c_2, c_3] = i_3
                    self.inverse_indices_3[i_3] = np.array([c_1, c_2, c_3])
                    i_3 += 1

        self.action_indices_2 = np.zeros((13, 13), dtype=np.int16)
        self.inverse_indices_2 = np.zeros((33, 2), dtype=np.int16)
        i_2 = 0
        for c_1 in range(0, 12):
            n_1 = min(c_1 + 3, 12)
            for c_2 in range(c_1 + 1, n_1 + 1):
                self.action_indices_2[c_1, c_2] = i_2
                self.inverse_indices_2[i_2] = np.array([c_1, c_2])
                i_2 += 1

    def act(self, game_env):
        """
        Given an Env object available to this agent, takes the current PPO state and action and computes forward
        pass of model to get the suggested legal action.
        However, if only one action is legal (pass), then take that action.
        """

        infoset = game_env.infoset
        action_sequence = game_env._env.card_play_action_seq
        # num singles, num singles+pairs, num singles+pairs+triples, num all except pass
        num_of_actions = [13, 46, 407, 1694]

        if len(infoset.legal_actions) == 1:  # default case
            return infoset.legal_actions[0]

        # after computing state and action, make forward pass
        state = torch.from_numpy(get_ppo_state(self.starting_hand, infoset, action_sequence))
        available_actions = torch.from_numpy(get_ppo_action(
            self.starting_hand, infoset, num_of_actions,
            self.action_indices_2, self.action_indices_3, self.action_indices_5))

        if torch.cuda.is_available():
            state, available_actions = state.cuda(), available_actions.cuda()
        # print(state.shape)  # should be size 1*self.obs_dim
        # print(available_actions.shape)  # should be size 1*self.act_dim
        with torch.no_grad():
            pred_action = self.model_pytorch.forward(state, available_actions)['out_action'].detach().cpu().numpy()
        assert pred_action.shape[0] == self.act_dim

        # convert the output back to action
        best_action_index = np.argmax(pred_action)
        if best_action_index == num_of_actions[3]:
            return []
        elif best_action_index >= num_of_actions[2]:
            v = self.inverse_indices_5[best_action_index-num_of_actions[2]].tolist()
        elif best_action_index >= 77:
            raise Exception("invalid action index: played 4-card move")
        elif best_action_index >= num_of_actions[1]:
            v = self.inverse_indices_3[best_action_index-num_of_actions[1]].tolist()
        elif best_action_index >= num_of_actions[0]:
            v = self.inverse_indices_2[best_action_index-num_of_actions[0]].tolist()
        elif best_action_index >= 0:
            return [self.starting_hand[best_action_index]]
        else:
            raise Exception("PPO action index < 0")
        move = []
        for i in v:
            move.append(self.starting_hand[i])
        return move
