import torch
from torch import nn
from big2_rl.env.move_generator import MovesGener
import big2_rl.env.move_detector as md
from big2_rl.env.game import Position
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
        self.obs_dim = 412
        self.act_dim = 1695

        # load the Big2PPO model's pretrained parameters
        self.model_pytorch = PPONetworkPytorch(self.obs_dim, self.act_dim)
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
            # print("{}, shape: {}" .format(i, param.shape))

            self.starting_hand = None
            self.action_indices_5, self.inverse_indices_5, \
                self.action_indices_3, self.inverse_indices_3, \
                self.action_indices_2, self.inverse_indices_2 = 0, 0, 0, 0, 0, 0

    def load_indices_lookup(self, hand):
        self.starting_hand = hand  # initialise a starting hand of 13 cards

        # idea: given hand (list of ints), get its indices, then get its NN input representation
        self.action_indices_5 = np.zeros((13, 13, 13, 13, 13))
        self.inverse_indices_5 = np.zeros((1287, 5))  # since 13 choose 5 = 1287
        i_5 = 0
        for c_1 in range(0, 9):
            for c_2 in range(c_1 + 1, 10):
                for c_3 in range(c_2 + 1, 11):
                    for c_4 in range(c_3 + 1, 12):
                        for c_5 in range(c_4 + 1, 13):
                            self.action_indices_5[c_1, c_2, c_3, c_4, c_5] = i_5
                            self.inverse_indices_5[i_5] = np.array([c_1, c_2, c_3, c_4, c_5])
                            i_5 += 1

        self.action_indices_3 = np.zeros((13, 13, 13))
        self.inverse_indices_3 = np.zeros((31, 3))
        i_3 = 0
        for c_1 in range(0, 11):
            n_31 = min(c_1 + 2, 11)
            for c_2 in range(c_1 + 1, n_31 + 1):
                n_32 = min(c_2 + 3, 12)
                for c_3 in range(c_2 + 1, n_32 + 1):
                    self.action_indices_3[c_1, c_2, c_3] = i_3
                    self.inverse_indices_3[i_3] = np.array([c_1, c_2, c_3])
                    i_3 += 1

        self.action_indices_2 = np.zeros((13, 13))
        self.inverse_indices_2 = np.zeros((33, 3))
        i_2 = 0
        for c_1 in range(0, 12):
            n_21 = min(c_1 + 3, 12)
            for c_2 in range(c_1 + 1, n_21 + 1):
                self.action_indices_2[c_1, c_2] = i_2
                self.inverse_indices_2[i_2] = np.array([c_1, c_2])
                i_2 += 1

    def act(self, game_env):
        """
        Given an infoset available to this agent, takes the z_batch and x_batch features (historical actions
        and current state + action) and computes forward pass of model to get the suggested legal action.
        However, if only one action is legal (pass), then take that action.
        """
        infoset = game_env.infoset
        action_sequence = game_env.card_play_action_seq

        if len(infoset.legal_actions) == 1:  # default case
            return infoset.legal_actions[0]

        # num singles, num singles+pairs, num singles+pairs+triples, num all except pass
        number_of_actions = [13, 46, 407, 1694]

        #### get available actions
        available_actions = np.full(shape=(self.act_dim,), fill_value=np.NINF)  # size 1695
        # note that the PPO model also considers 4-card moves to be valid (four of a kind or two pairs). In our ruleset,
        # these will always be invalid moves.
        if infoset.last_move is not []:  # if last move is not pass, make pass an available move
            available_actions[self.act_dim-1] = 0

        # convert each possible legal move to its index in range [0, 1695)
        for possible_move in infoset.legal_actions:
            # convert card values of that possible move to indices in self.starting_hand
            possible_move_as_ind = []
            for i, val in enumerate(self.starting_hand):
                for card in possible_move:
                    if val == card:
                        possible_move_as_ind.append(i)
            # since num of moves with 1 card: 13, 2 cards: 33, 3 cards: 31, 4 cards: 330, 5 cards: 1287,
            # need to add offset. Eg for 3-card hands, need add offset of 1-card and 2-card (13+33=46)
            nn_input_ind = None
            assert type(self.action_indices_3) is not int
            if len(possible_move_as_ind) == 2:
                nn_input_ind = self.action_indices_2[possible_move_as_ind[0], possible_move_as_ind[1]] + \
                               number_of_actions[0]
            elif len(possible_move_as_ind) == 3:
                nn_input_ind = self.action_indices_3[possible_move_as_ind[0], possible_move_as_ind[1],
                                                     possible_move_as_ind[2]] + number_of_actions[1]
            elif len(possible_move_as_ind) == 5:
                nn_input_ind = self.action_indices_5[possible_move_as_ind[0], possible_move_as_ind[1],
                                                     possible_move_as_ind[2], possible_move_as_ind[3],
                                                     possible_move_as_ind[4]] + number_of_actions[2]
            elif len(possible_move_as_ind) == 1:
                nn_input_ind = possible_move_as_ind[0]
            assert nn_input_ind is not None
            available_actions[nn_input_ind] = 0

        #### get current state
        # ignore 4-card hands (2Pr, Quad). Agent is still able to play Quads+kicker but won't have a feature for it
        # 1. for each of 13 cards in hand: rank (13) + suit (4) + inPair + in3 + inFour + inStraight + inFlush (5)
        # Total for (1): (13+4+5)*13
        # 2. for each opponent in {downstream, across, upstream}:
        # cards left (one-hot of 13) + has played Ax or 2x (8) + hasPlayed Pr,Triple,2Pr,Str,Flush,FH (6)
        # Total for (2): (13+8+6)*3
        # 3. global Qx, Kx, Ax, 2x played. Total for (3): 16
        # 4. rank of the highest card in previous non-pass-move (13) + suit of the highest card in previous
        # non-pass-move (4) + previousMoveIsSingle,Pr,Triple,2Pr,Quad,Str,Flush,FH (8) + control,0pass,1pass,2pass (4)
        # Total for (4): 13+4+8+4
        # size of state: (13+4+5)*13 + (13+8+6)*3 + 16 + (13+4+8+4) = 412
        state = np.zeros((self.obs_dim,))
        feat_1_size = 13+4+5
        mg = MovesGener(infoset.player_hand_cards)
        for index, card in enumerate(self.starting_hand):
            if card not in infoset.player_hand_cards:  # if card only in starting hand, it has already been played
                continue
            suit, rank = card % 4, card // 4
            state[feat_1_size*index+13+suit] = 1
            state[feat_1_size*index+rank] = 1

            in_hand = [mg.gen_type_2_pair(), mg.gen_type_3_triple(), [], mg.gen_type_4_straight() +
                       mg.gen_type_8_straightflush(), mg.gen_type_5_flush() + mg.gen_type_8_straightflush(),
                       mg.gen_type_6_fullhouse()]
            for in_hand_index, hand_type in enumerate(in_hand):
                found = 0
                if in_hand_index != 2:  # Two pairs never valid
                    for hand in hand_type:
                        if card in hand:
                            found = 1
                            break
                state[feat_1_size*index+17+in_hand_index] = found

        # get current position. Iterate through each opponent position, compute feature 2
        feat_2_offset = (13+4+5)*13
        feat_2_size = 13+8+6
        this_position = Position[infoset.player_position].value
        total_played_cards = infoset.played_cards[this_position]  # for feature 3
        # get list of opponent positions in order
        position_range = [_ % 4 for _ in range(this_position + 1, this_position + 4)]
        for opp_ind, pos in enumerate(position_range):
            posname = Position(pos).name
            num_cards_left = infoset.num_cards_left_dict[posname]
            state[feat_2_offset+feat_2_size*opp_ind + num_cards_left-1] = 1

            played_cards = infoset.played_cards[posname]
            total_played_cards += played_cards
            high_cards = [_ for _ in range(44, 52)]
            for card_ind, high_card in enumerate(high_cards):  # iterate over {Ad, Ac, ..., 2h, 2s}
                if high_card in played_cards:
                    state[feat_2_offset+feat_2_size*opp_ind + 13+card_ind] = 1

            found = [False for _ in range(5)]  # found_i=True if we already found a hand of that type by current player
            for hand in action_sequence:
                if not found[0] and md.get_move_type(hand)['type'] == md.TYPE_2_PAIR:
                    state[feat_2_offset+feat_2_size*opp_ind + 21] = 1
                    found[0] = True
                elif not found[1] and md.get_move_type(hand)['type'] == md.TYPE_3_TRIPLE:
                    state[feat_2_offset+feat_2_size*opp_ind + 22] = 1
                    found[1] = True
                elif not found[2] and md.get_move_type(hand)['type'] == md.TYPE_4_STRAIGHT:
                    state[feat_2_offset+feat_2_size*opp_ind + 24] = 1
                    found[2] = True
                elif not found[3] and md.get_move_type(hand)['type'] == md.TYPE_5_FLUSH:
                    state[feat_2_offset+feat_2_size*opp_ind + 25] = 1
                    found[3] = True
                elif not found[4] and md.get_move_type(hand)['type'] == md.TYPE_6_FULLHOUSE:
                    state[feat_2_offset+feat_2_size*opp_ind + 26] = 1
                    found[4] = True

        feat_3_offset = feat_2_offset + (13+8+6)*3
        global_high_cards = [_ for _ in range(36, 52)]  # {Qd, Qc, ..., 2d, 2c, 2h, 2s}
        for card_ind, high_card in enumerate(global_high_cards):
            if high_card in global_high_cards:
                state[feat_3_offset + card_ind] = 1

        feat_4_offset = 16 + feat_3_offset
        rival_move = []  # get most recent non-pass move
        passes = -99
        if len(action_sequence) != 0:
            if len(action_sequence[-1]) == 0:
                if len(action_sequence[-2]) == 0:
                    if len(action_sequence[-3]) == 0:
                        rival_move = []
                        passes = -1  # for easy indexing
                    else:
                        rival_move = action_sequence[-3]
                        passes = 2
                else:
                    rival_move = action_sequence[-2]
                    passes = 1
            else:
                rival_move = action_sequence[-1]
                passes = 0
        assert passes != -99

        if rival_move is not []:
            max_card = max(rival_move)
            max_card_rank, max_card_suit = max_card // 4, max_card % 4
            state[feat_4_offset + max_card_rank] = 1
            state[feat_4_offset + 13+max_card_suit] = 1

        # get the type of the previous move played
        rival_type = md.get_move_type(rival_move)['type']
        if rival_type == md.TYPE_1_SINGLE:
            state[feat_4_offset+17] = 1
        elif rival_type == md.TYPE_2_PAIR:
            state[feat_4_offset+18] = 1
        elif rival_type == md.TYPE_3_TRIPLE:
            state[feat_4_offset+19] = 1
        elif rival_type == md.TYPE_4_STRAIGHT or rival_type == md.TYPE_8_STRAIGHTFLUSH:
            state[feat_4_offset+22] = 1
        elif rival_type == md.TYPE_5_FLUSH or rival_type == md.TYPE_8_STRAIGHTFLUSH:
            state[feat_4_offset+23] = 1
        elif rival_type == md.TYPE_6_FULLHOUSE:
            state[feat_4_offset+24] = 1
        # 25 = control, 26 = 0 pass, 27 = 1 pass, 28 = 2 passes
        state[feat_4_offset + 26 + passes] = 1

        # after computing state and action, make forward pass
        state = torch.from_numpy(state)
        available_actions = torch.from_numpy(available_actions)
        if torch.cuda.is_available():
            state, available_actions = state.cuda(), available_actions.cuda()
        pred_action = self.model_pytorch.forward(state, available_actions)['out_action'].detach().cpu().numpy()
        assert pred_action.shape[0] == self.act_dim

        # TODO convert the output back to action
        best_action_index = np.argmax(pred_action, axis=0)[0]
        if best_action_index == number_of_actions[3]:
            return []
        elif best_action_index >= number_of_actions[2]:
            v = self.inverse_indices_5[best_action_index-number_of_actions[2]].tolist()
        elif best_action_index >= 77:
            raise Exception("invalid action index: played 4-card move")
        elif best_action_index >= number_of_actions[1]:
            v = self.inverse_indices_3[best_action_index-number_of_actions[1]].tolist()
        elif best_action_index >= number_of_actions[0]:
            v = self.inverse_indices_2[best_action_index-number_of_actions[0]].tolist()
        elif best_action_index >= 0:
            return [self.starting_hand[best_action_index]]
        else:
            raise Exception("PPO action index < 0")
        move = []
        for i in v:
            move.append(self.starting_hand[i])
        return move
