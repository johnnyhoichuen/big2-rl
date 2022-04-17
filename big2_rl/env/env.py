import numpy as np

from .game import GameEnv, Position
from .move_generator import MovesGener
from big2_rl.env import move_detector as md

deck = [i for i in range(0, 52)]


class Env:
    """
    Doudizhu multi-agent wrapper
    """

    def __init__(self):
        """
        The objective for each agent is number of units won (which may vary depending on
        the reward policy (whether unplayed 2's, quads, SFs etc. double the penalty)).
        Here, we use dummy agents to isolate
        players and environments to have a more gym style
        interface. To achieve this, we use dummy players
        to play. For each move, we tell the corresponding
        dummy player which action to play, then the player
        will perform the actual action in the game engine.
        """

        # Initialize players
        # We use 4 dummy players for each target position
        self.players = {}
        for pos in Position:
            self.players[pos.name] = DummyAgent(pos.name)

        # Initialize the internal environment
        self._env = GameEnv(self.players)

        self.infoset = None

    def reset(self):
        """
        Every time reset is called, the environment
        will be re-initialized with a new deck of cards.
        This function is usually called when a game is over.
        """
        self._env.reset()  # reset attributes for the next deal

        # Randomly shuffle the deck
        _deck = deck.copy()
        np.random.shuffle(_deck)

        card_play_data = {}
        for pos in Position:
            card_play_data[pos.name] = _deck[pos.value*13:(pos.value+1)*13]

        for key in card_play_data:
            card_play_data[key].sort()  # sort each hand in ascending order

        # Initialize the cards in each player's hand and get starting player
        self._env.card_play_init(card_play_data)
        self.infoset = self._game_infoset

        return get_obs(self.infoset)

    def step(self, action):
        """
        step() takes as input the action, which
        is a list of integers, and outputs the next observation (a dict),
        reward (dict: integer of the corresponding position and their rewards),
        and a Boolean variable 'done' indicating whether the current game is finished.
        It also returns an empty dictionary that is reserved to pass useful information.
        """
        assert action in self.infoset.legal_actions
        self.players[self._acting_player_position].set_action(action)
        self._env.step()
        self.infoset = self._game_infoset
        done = False
        reward = 0.0
        if self._game_over:
            done = True
            reward = self._get_reward()
            obs = None
        else:  # get set of features
            obs = get_obs(self.infoset)
        return obs, reward, done, {}

    def _get_reward(self):
        """
        This function is called in the end of each deal.
        It returns a (dict String : integer) corresponding to amount won/lost by each position.
        """
        self._env.compute_player_reward()  # compute player reward then find the amount won by the winner
        return self._env.player_reward_dict

    # these properties are used to access attributes of the GameEnv object wrapped inside the Env instance
    @property
    def _game_infoset(self):
        """
        returns:
        InfoSet(String pos.name) containing all information of the current situation
        (hands of all players, all historical moves, etc.)
        """
        return self._env.game_infoset

    @property
    def _game_winner(self):
        """
        returns:
        (String) containing position name of the winner
        """
        return self._env.get_winner()

    @property
    def _acting_player_position(self):
        """
        (String) one of the pos.names in Position
        """
        return self._env.acting_player_position

    @property
    def _game_over(self):
        """
        (boolean)
        """
        return self._env.game_over


class DummyAgent(object):
    """
    Dummy agent is designed to easily interact with the
    game engine. The agent will first be told what action
    to perform. Then the environment will call this agent
    to perform the actual action. This can help us to
    isolate environment and agents towards a gym like
    interface.
    """
    def __init__(self, position):
        self.position = position
        self.action = None

    def act(self, infoset):
        """
        Simply return the action that is set previously.
        """
        assert self.action in infoset.legal_actions
        return self.action

    def set_action(self, action):
        """
        The environment uses this function to tell
        the dummy agent what to do.
        """
        self.action = action


def get_obs(infoset):
    """
    This function obtains observations with imperfect information
    from the infoset and returns a dictionary named `obs`, which contains
    several fields that are used to train the model.

    `x_batch` is a batch of features (excluding the historical moves).
    It also encodes the action feature

    `z_batch` is a batch of features with historical moves only.

    `legal_actions` is the set of legal moves

    `x_no_action`: the features (excluding the historical moves and
    the action features). It does not have the batch dim.

    `z`: same as z_batch but not a batch.
    """

    # reminder: infoset is specific to each player
    num_legal_actions = len(infoset.legal_actions)  # get number of available legal moves = NL

    # convert:
    # 1. current player's hand to one hot vector
    # 2. union of current player's opponent's hands to one hot vector
    # 3. convert last non-pass move (list of int) to one hot vector
    # then get their batch representations (increase dim by 1) for training purposes
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :], num_legal_actions, axis=0)
    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :], num_legal_actions, axis=0)
    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :], num_legal_actions, axis=0)

    # create a NL*52 matrix. Populate each 52-card vector (there are NL of them) with the one-hot encoded version of
    # each possible legal move
    my_action_batch = np.zeros(my_handcards_batch.shape)
    for i, action in enumerate(infoset.legal_actions):
        my_action_batch[i, :] = _cards2array(action)

    # list of 3 nparray-s storing other 3 players' previous action, num cards left, set of cards played
    # in both regular and batch form
    other_players_action = []
    other_players_num_cards_left = []
    other_players_played_cards = []

    other_players_action_batch = []
    other_players_num_cards_left_batch = []
    other_players_played_cards_batch = []

    # iterate through every position not the current position (i.e. not equal to infoset.player_position)
    # get value of position. store opponent features as [next, across, before] for all 4 players
    # e.g. north stores opponent features as [west, south, east]
    this_position = Position[infoset.player_position].value
    # get list of opponent positions in order
    position_range = [_ % 4 for _ in range(this_position + 1, this_position + 4)]
    for pos in position_range:
        posname = Position(pos).name

        # convert:
        # 1. opponent's most recent move (including pass) to one hot vector
        # 2. number of cards left in opponent hand to one hot vector
        # 3. set of cards played by this opponent to one hot vector
        opponent_action = _cards2array(infoset.last_move_dict[posname])
        opponent_num_cards_left = _get_one_hot_array(infoset.num_cards_left_dict[posname], 13)
        opponent_played_cards = _cards2array(infoset.played_cards[posname])

        # then get their batch representations (increase dim by 1) for training purposes
        opponent_action_batch = np.repeat(opponent_action[np.newaxis, :], num_legal_actions, axis=0)
        opponent_num_cards_left_batch = np.repeat(opponent_num_cards_left[np.newaxis, :], num_legal_actions, axis=0)
        opponent_played_cards_batch = np.repeat(opponent_played_cards[np.newaxis, :], num_legal_actions, axis=0)

        other_players_action.append(opponent_action)
        other_players_num_cards_left.append(opponent_num_cards_left)
        other_players_played_cards.append(opponent_played_cards)

        other_players_action_batch.append(opponent_action_batch)
        other_players_num_cards_left_batch.append(opponent_num_cards_left_batch)
        other_players_played_cards_batch.append(opponent_played_cards_batch)

    # print(np.array(other_players_action_batch).shape) = (3, NL, 52). Need to reshape to shape (NL, 3, 52)

    # construct the feature groupings
    x_batch = np.hstack((my_handcards_batch,
                         other_handcards_batch,
                         last_action_batch,
                         np.array(other_players_action_batch).reshape((num_legal_actions, 3, 52)).reshape(
                             num_legal_actions, 52 * 3),
                         np.array(other_players_num_cards_left_batch).reshape((num_legal_actions, 3, 13)).reshape(
                             num_legal_actions, 13 * 3),
                         np.array(other_players_played_cards_batch).reshape((num_legal_actions, 3, 52)).reshape(
                             num_legal_actions, 52 * 3),
                         my_action_batch
                         ))
    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             last_action,
                             np.array(other_players_action).reshape(3, 52).reshape(52 * 3),
                             np.array(other_players_num_cards_left).reshape(3, 13).reshape(13 * 3),
                             np.array(other_players_played_cards).reshape(3, 52).reshape(52 * 3)
                             ))
    # z should be a 4*208 matrix encoding the previous 16 moves
    z = _action_seq_list2array(_process_action_seq(infoset.card_play_action_seq))
    z_batch = np.repeat(z[np.newaxis, :, :], num_legal_actions, axis=0)
    obs = {
            'position': infoset.player_position,
            'x_batch': x_batch.astype(np.float32),
            'z_batch': z_batch.astype(np.float32),
            'legal_actions': infoset.legal_actions,
            'x_no_action': x_no_action.astype(np.int8),
            'z': z.astype(np.int8),
          }

    assert obs['x_batch'].shape[1] == 559
    assert obs['z_batch'].shape[2] == 208 and obs['z_batch'].shape[1] == 4
    assert obs['x_no_action'].shape[0] == 507
    assert obs['z'].shape[1] == 208 and obs['z'].shape[0] == 4

    # Feature list:
    # opp1 = opponent acting after player, opp2 = opponent across player, opp3 = opponent acting before player
    #        | Feature                                     | Size
    # Action | one-hot vector of a prospective action | 52
    # State  | one-hot vector of hand cards held by current player | 52
    # State  | one-hot vector of union of opponents' cards left | 52
    # State  | one-hot vector of most recent non-pass move | 52
    # State  | one-hot vector of opp1's most recent action (can include pass) | 52
    # State  | one-hot vector of opp2's most recent action (can include pass) | 52
    # State  | one-hot vector of opp3's most recent action (can include pass) | 52
    # State  | one-hot vector of number of cards left in opp1's hand | 13
    # State  | one-hot vector of number of cards left in opp2's hand | 13
    # State  | one-hot vector of number of cards left in opp3's hand | 13
    # State  | one-hot vector of set of cards played by opp1 | 52
    # State  | one-hot vector of set of cards played by opp2 | 52
    # State  | one-hot vector of set of cards played by opp3 | 52
    # State  | concatenated one-hot matrices describing most recent 16 moves | 4 * 208

    # Expected sizes:
    # x_batch size: (NL, 559)
    # z_batch size: (NL, 4, 208)
    # legal actions size: not nparray, is list of size NL
    # x_no_action size: (507,)
    # z size: (4, 208)
    return obs


def get_ppo_action(starting_hand, infoset, number_of_actions, act_lookup_2, act_lookup_3, act_lookup_5, act_dim=1695):
    """
    Get available actions for PPO as a np-array of size 1695
    """
    # requires: act_dim, infoset, starting_hand, action lookup tables (2,3,5), number_of_actions

    available_actions = np.full(shape=(act_dim,), fill_value=np.NINF, dtype=np.float32)  # size 1695
    # note that the PPO model also considers 4-card moves to be valid (four of a kind or two pairs). In our ruleset,
    # these will always be invalid moves.

    # convert each possible legal move to its index in range [0, 1695)
    for possible_move in infoset.legal_actions:
        # convert card values of that possible move to indices in self.starting_hand
        possible_move_as_ind = []
        for i, val in enumerate(starting_hand):
            for card in possible_move:
                if val == card:
                    possible_move_as_ind.append(i)

        # since num of moves with 1 card: 13, 2 cards: 33, 3 cards: 31, 4 cards: 330, 5 cards: 1287,
        # need to add offset. Eg for 3-card hands, need add offset of 1-card and 2-card (13+33=46)
        nn_input_ind = None
        assert type(act_lookup_3) is not int
        if len(possible_move_as_ind) == 2:
            nn_input_ind = act_lookup_2[possible_move_as_ind[0], possible_move_as_ind[1]] + \
                           number_of_actions[0]
        elif len(possible_move_as_ind) == 3:
            nn_input_ind = act_lookup_3[possible_move_as_ind[0], possible_move_as_ind[1],
                                        possible_move_as_ind[2]] + number_of_actions[1]
        elif len(possible_move_as_ind) == 5:
            nn_input_ind = act_lookup_5[possible_move_as_ind[0], possible_move_as_ind[1],
                                        possible_move_as_ind[2], possible_move_as_ind[3],
                                        possible_move_as_ind[4]] + number_of_actions[2]
        elif len(possible_move_as_ind) == 1:
            nn_input_ind = possible_move_as_ind[0]
        elif len(possible_move_as_ind) == 0:  # pass
            nn_input_ind = number_of_actions[3]
        assert nn_input_ind is not None
        available_actions[nn_input_ind] = 0
        return available_actions


def get_ppo_state(starting_hand, infoset, action_sequence, obs_dim=412):
    """
    Returns current state for PPO (a 412-dimensional feature) as nparray

    Note: We ignore 4-card hands (2Pr, Quad). Agent is still able to play Quads+kicker but won't have a feature for it
    F1. for each of 13 cards in hand: rank (13) + suit (4) + inPair + in3 + inFour + inStraight + inFlush (5)
    # Total for (1): (13+4+5)*13
    F2. for each opponent in {downstream, across, upstream}:
    cards left (one-hot of 13) + has played Ax or 2x (8) + hasPlayed Pr,Triple,2Pr,Str,Flush,FH (6)
    # Total for (2): (13+8+6)*3
    F3. global Qx, Kx, Ax, 2x played.
    # Total for (3): 16
    F4. rank of the highest card in previous non-pass-move (13) + suit of the highest card in previous
    non-pass-move (4) + previousMoveIsSingle,Pr,Triple,2Pr,Quad,Str,Flush,FH (8) + control,0pass,1pass,2pass (4)
    # Total for (4): 13+4+8+4
    # Total size of state: (13+4+5)*13 + (13+8+6)*3 + 16 + (13+4+8+4) = 412
    """
    # requires: starting_hand, obs_dim, infoset, action_sequence, MovesGener, MoveDetector as md
    state = np.zeros((obs_dim,), dtype=np.float32)
    feat_1_size = 13 + 4 + 5
    mg = MovesGener(infoset.player_hand_cards)
    for index, card in enumerate(starting_hand):
        if card not in infoset.player_hand_cards:  # if card only in starting hand, it has already been played
            continue
        suit, rank = card % 4, card // 4
        state[feat_1_size * index + 13 + suit] = 1
        state[feat_1_size * index + rank] = 1

        in_hand = [mg.gen_type_2_pair(), mg.gen_type_3_triple(), [], mg.gen_type_4_straight() +
                   mg.gen_type_8_straightflush(), mg.gen_type_5_flush() + mg.gen_type_8_straightflush()]
        for in_hand_index, hand_type in enumerate(in_hand):
            found = 0
            for hand in hand_type:
                if card in hand:
                    found = 1
                    break
            state[feat_1_size * index + 17 + in_hand_index] = found

    # get current position. Iterate through each opponent position, compute feature 2
    feat_2_offset = (13 + 4 + 5) * 13
    feat_2_size = 13 + 8 + 6
    this_position = Position[infoset.player_position].value
    total_played_cards = infoset.played_cards[infoset.player_position]  # for feature 3
    # get list of opponent positions in order
    position_range = [_ % 4 for _ in range(this_position + 1, this_position + 4)]
    for opp_ind, pos in enumerate(position_range):
        posname = Position(pos).name
        num_cards_left = infoset.num_cards_left_dict[posname]
        state[feat_2_offset + feat_2_size * opp_ind + num_cards_left - 1] = 1

        played_cards = infoset.played_cards[posname]
        total_played_cards += played_cards
        high_cards = [_ for _ in range(44, 52)]
        for card_ind, high_card in enumerate(high_cards):  # iterate over {Ad, Ac, ..., 2h, 2s}
            if high_card in played_cards:
                state[feat_2_offset + feat_2_size * opp_ind + 13 + card_ind] = 1

        found = [False for _ in range(5)]  # found_i=True if we already found a hand of that type by current player
        for hand in action_sequence:
            if not found[0] and md.get_move_type(hand)['type'] == md.TYPE_2_PAIR:
                state[feat_2_offset + feat_2_size * opp_ind + 21] = 1
                found[0] = True
            elif not found[1] and md.get_move_type(hand)['type'] == md.TYPE_3_TRIPLE:
                state[feat_2_offset + feat_2_size * opp_ind + 22] = 1
                found[1] = True
            elif not found[2] and md.get_move_type(hand)['type'] == md.TYPE_4_STRAIGHT:
                state[feat_2_offset + feat_2_size * opp_ind + 24] = 1
                found[2] = True
            elif not found[3] and md.get_move_type(hand)['type'] == md.TYPE_5_FLUSH:
                state[feat_2_offset + feat_2_size * opp_ind + 25] = 1
                found[3] = True
            elif not found[4] and md.get_move_type(hand)['type'] == md.TYPE_6_FULLHOUSE:
                state[feat_2_offset + feat_2_size * opp_ind + 26] = 1
                found[4] = True

    feat_3_offset = feat_2_offset + (13 + 8 + 6) * 3
    global_high_cards = [_ for _ in range(36, 52)]  # {Qd, Qc, ..., 2d, 2c, 2h, 2s}
    for card_ind, high_card in enumerate(global_high_cards):
        if high_card in total_played_cards:
            state[feat_3_offset + card_ind] = 1

    feat_4_offset = 16 + feat_3_offset
    # get most recent non-pass move
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
    else:
        rival_move = []
        passes = -1

    if rival_move != []:
        max_card = max(rival_move)
        max_card_rank, max_card_suit = max_card // 4, max_card % 4
        state[feat_4_offset + max_card_rank] = 1
        state[feat_4_offset + 13 + max_card_suit] = 1

    # get the type of the previous move played
    rival_type = md.get_move_type(rival_move)['type']
    if rival_type == md.TYPE_1_SINGLE:
        state[feat_4_offset + 17] = 1
    elif rival_type == md.TYPE_2_PAIR:
        state[feat_4_offset + 18] = 1
    elif rival_type == md.TYPE_3_TRIPLE:
        state[feat_4_offset + 19] = 1
    elif rival_type == md.TYPE_4_STRAIGHT or rival_type == md.TYPE_8_STRAIGHTFLUSH:
        state[feat_4_offset + 22] = 1
    elif rival_type == md.TYPE_5_FLUSH or rival_type == md.TYPE_8_STRAIGHTFLUSH:
        state[feat_4_offset + 23] = 1
    elif rival_type == md.TYPE_6_FULLHOUSE:
        state[feat_4_offset + 24] = 1
    # 25 = control, 26 = 0 pass, 27 = 1 pass, 28 = 2 passes
    state[feat_4_offset + 26 + passes] = 1
    return state


def _get_one_hot_array(num_left_cards, max_num_cards):
    """
    A utility function to obtain one-hot encoding of the number of cards left in a player's hand
    returns a 13-dimensional vector with element i=1 and all other elements=0 iff that player has i+1 cards remaining
    """
    one_hot = np.zeros(max_num_cards)
    one_hot[num_left_cards - 1] = 1
    return one_hot


def _cards2array(list_cards):
    """
    A utility function that transforms the actions, i.e.,
    A list of integers into a 52-element one-hot vector of cards.
    """
    one_hot = np.zeros(52, dtype=np.int8)
    if len(list_cards) == 0:
        return one_hot
    # since each card already in [0, 51] range
    for card in list_cards:
        one_hot[card] = 1
    return one_hot


def _action_seq_list2array(action_seq_list):
    """
    A utility function to encode the historical moves.
    We encode the historical 16 actions (4 by each player).
    If there are insufficient actions, we pad the features with 0.
    Since 4 moves is a round in Big Two, we concatenate
    the representations for each consecutive 4 moves.
    Finally, we obtain a 4x208 matrix, which will be fed
    into LSTM for encoding.
    """
    action_seq_array = np.zeros((len(action_seq_list), 52))
    for row, list_cards in enumerate(action_seq_list):
        action_seq_array[row, :] = _cards2array(list_cards)
    action_seq_array = action_seq_array.reshape(4, 208)
    return action_seq_array


def _process_action_seq(sequence, length=16):
    """
    A utility function encoding historical moves. We
    encode 16 moves. If there is no 16 moves, we pad
    with zeros.
    """
    sequence = sequence[-length:].copy()
    if len(sequence) < length:
        empty_sequence = [[] for _ in range(length - len(sequence))]
        empty_sequence.extend(sequence)
        sequence = empty_sequence
    return sequence
