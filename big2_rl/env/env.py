import numpy as np

from .game import GameEnv, Position

deck = [range(0, 52)]


class Env:
    """
    Doudizhu multi-agent wrapper
    """

    def __init__(self):
        """
        The objective for each agent is number of units won (which may vary depending on
        the reward policy (whether or not unplayed 2's, quads, SFs etc double the penalty).
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
        is a list of integers, and outputs the next observation,
        reward, and a Boolean variable indicating whether the
        current game is finished. It also returns an empty
        dictionary that is reserved to pass useful information.
        TODO
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
        This function is called in the end of each
        game. It returns either 1/-1 for win/loss,
        or ADP, i.e., every bomb will double the score.
        TODO
        """
        self._env.compute_player_reward()  # compute player reward then find the amount won by the winner
        return self._env.player_reward_dict[self._game_winner]

    # these properties are used to access attributes of the GameEnv object wrapped inside the Env instance
    @property
    def _game_infoset(self):
        """
        returns:
        InfoSet(String pos.name) containing all information of the current situation
        (hands of all players, all historical moves, etc)
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

    # (dict String : some nparray) storing other 3 players' previous action, num cards left, set of cards played
    # in both regular and batch form
    other_players_action = {}
    other_players_num_cards_left = {}
    other_players_played_cards = {}

    other_players_action_batch = {}
    other_players_num_cards_left_batch = {}
    other_players_played_cards_batch = {}

    # iterate through every position not the current position (i.e. not equal to infoset.player_position)
    # TODO should this be symmetric for all 4 players?
    for pos in Position:
        if pos.name == infoset.player_position:
            continue  # break is incorrect

        # convert:
        # 1. opponent's most recent move (including pass) to one hot vector
        # 2. number of cards left in opponent hand to one hot vector
        # 3. set of cards played by this opponent to one hot vector
        opponent_action = _cards2array(infoset.last_move_dict[pos.name])
        opponent_num_cards_left = _get_one_hot_array(infoset.num_cards_left_dict[pos.name], 13)
        opponent_played_cards = _cards2array(infoset.played_cards[pos.name])

        other_players_action[pos.name] = opponent_action
        other_players_num_cards_left[pos.name] = opponent_num_cards_left
        other_players_played_cards[pos.name] = opponent_played_cards

        # then get their batch representations (increase dim by 1) for training purposes
        opponent_action_batch = np.repeat(opponent_action[np.newaxis, :], num_legal_actions, axis=0)
        opponent_num_cards_left_batch = np.repeat(opponent_num_cards_left[np.newaxis, :], num_legal_actions, axis=0)
        opponent_played_cards_batch = np.repeat(opponent_played_cards[np.newaxis, :], num_legal_actions, axis=0)

        other_players_action_batch[pos.name] = opponent_action_batch
        other_players_num_cards_left_batch[pos.name] = opponent_num_cards_left_batch
        other_players_played_cards_batch[pos.name] = opponent_played_cards_batch

    # construct the feature groupings
    x_batch = np.hstack((my_handcards_batch,
                         other_handcards_batch,
                         last_action_batch,
                         [value for key, value in other_players_action_batch],
                         [value for key, value in other_players_num_cards_left_batch],
                         [value for key, value in other_players_played_cards_batch],
                         my_action_batch
                         ))
    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             last_action,
                             [value for key, value in other_players_action],
                             [value for key, value in other_players_num_cards_left],
                             [value for key, value in other_players_played_cards]
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

    # Feature list:
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
