from collections import Counter
from turtle import position
import numpy as np

from env.game import GameEnv, Position

deck = [range(0,52)]

class Env:
    """
    Doudizhu multi-agent wrapper
    """

    # POSITIONS = ["south", "east", "north", "west"]

    def __init__(self, objective):
        """
        Objective is wp/adp/logadp. It indicates whether considers
        bomb in reward calculation. Here, we use dummy agents.
        This is because, in the orignial game, the players
        are `in` the game. Here, we want to isolate
        players and environments to have a more gym style
        interface. To achieve this, we use dummy players
        to play. For each move, we tell the corresponding
        dummy player which action to play, then the player
        will perform the actual action in the game engine.
        """
        self.objective = objective

        # Initialize players
        # We use three dummy player for the target position
        self.players = {}
        for pos in Position:
            self.players[pos] = DummyAgent(pos)

        # Initialize the internal environment
        self._env = GameEnv(self.players)

        self.infoset = None

    def reset(self):
        """
        Every time reset is called, the environment
        will be re-initialized with a new deck of cards.
        This function is usually called when a game is over.
        """
        self._env.reset()

        # Randomly shuffle the deck
        _deck = deck.copy()
        np.random.shuffle(_deck)

        card_play_data = {}
        # for i in range(len(Position)):
        #   card_play_data[Position(i)] = _deck[i*13:(i+1)*13]
        for i, position in enumerate(Position):
            card_play_data[position] = _deck[i*13:(i+1)*13]

        for key in card_play_data:
            card_play_data[key].sort() # sort each hand in ascending order

        # Initialize the cards
        self._env.card_play_init(card_play_data)
        self.infoset = self._game_infoset

        return get_obs(self.infoset)

    def step(self, action):
        """
        Step function takes as input the action, which
        is a list of integers, and output the next obervation,
        reward, and a Boolean variable indicating whether the
        current game is finished. It also returns an empty
        dictionary that is reserved to pass useful information.
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
        else:
            obs = get_obs(self.infoset)
        return obs, reward, done, {}

    def _get_reward(self):
        """
        This function is called in the end of each
        game. It returns either 1/-1 for win/loss,
        or ADP, i.e., every bomb will double the score.
        """
        winner = self._game_winner

        return self._env.compute_player_reward()

        # bomb_num = self._game_bomb_num
        # if winner == 'landlord':
        #     if self.objective == 'adp':
        #         return 2.0 ** bomb_num
        #     elif self.objective == 'logadp':
        #         return bomb_num + 1.0
        #     else:
        #         return 1.0
        # else:
        #     if self.objective == 'adp':
        #         return -2.0 ** bomb_num
        #     elif self.objective == 'logadp':
        #         return -bomb_num - 1.0
        #     else:
        #         return -1.0

    @property
    def _game_infoset(self):
        """
        Here, inforset is defined as all the information
        in the current situation, incuding the hand cards
        of all the players, all the historical moves, etc.
        That is, it contains perferfect infomation. Later,
        we will use functions to extract the observable
        information from the views of the three players.
        """
        return self._env.game_infoset

    @property
    def _game_winner(self):
        """ A string of landlord/peasants
        """
        return self._env.get_winner()

    @property
    def _acting_player_position(self):
        """
        The player that is active. It can be landlord,
        landlod_down, or landlord_up.
        """
        return self._env.acting_player_position

    @property
    def _game_over(self):
        """ Returns a Boolean
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
    from the infoset. It has three branches since we encode
    different features for different positions.

    This function will return dictionary named `obs`. It contains
    several fields. These fields will be used to train the model.
    One can play with those features to improve the performance.

    `position` is a string that can be landlord/landlord_down/landlord_up

    `x_batch` is a batch of features (excluding the hisorical moves).
    It also encodes the action feature

    `z_batch` is a batch of features with hisorical moves only.

    `legal_actions` is the legal moves

    `x_no_action`: the features (exluding the hitorical moves and
    the action features). It does not have the batch dim.

    `z`: same as z_batch but not a batch.
    """

    if infoset.player_position not in Position:
        raise ValueError('')

    # return _get_obs_landlord_down(infoset)
    # return _get_obs(infoset)

    # reminder: infoset is specific to each player

    num_legal_actions = len(infoset.legal_actions)
    my_handcards = infoset.player_hand_cards
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :], num_legal_actions, axis=0)
    other_handcards = infoset.other_hand_cards
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :], num_legal_actions, axis=0)
    last_action = infoset.last_move
    last_action_batch = np.repeat(last_action[np.newaxis, :], num_legal_actions, axis=0)

    # WTF is this?? (original code from douzero)
    # my_handcards = _cards2array(infoset.player_hand_cards)
    '''
    _cards2array beginning:  [3, 4, 4, 5, 7, 7, 8, 8, 9, 10, 10, 11, 11, 11, 12, 14, 14, 14, 17, 20]
    card, num_times: 3, 1
    card, num_times: 4, 2
    card, num_times: 5, 1
    card, num_times: 7, 2
    card, num_times: 8, 2
    card, num_times: 9, 1
    card, num_times: 10, 2
    card, num_times: 11, 3
    card, num_times: 12, 1
    card, num_times: 14, 3
    card, num_times: 17, 1
    card, num_times: 20, 1
    _cards2array result:  [1 0 0 0 1 1 0 0 1 0 0 0 0 0 0 0 1 1 0 0 1 1 0 0 1 0 0 0 1 1 0 0 1 1 1 0 1
    0 0 0 0 0 0 0 1 1 1 0 1 0 0 0 1 0]
    '''
    # my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
    #                                num_legal_actions, axis=0)
    '''
    example
    array = np.array([1,0,0,0,0,1,1,1,1,0])
    num_legal_actions = 2
    print(array)
    print(array[np.newaxis, :])
    list = np.repeat(array[np.newaxis, :], num_legal_actions, axis=0)
    print(list)

    result:
    [1 0 0 0 0 1 1 1 1 0]

    [[1 0 0 0 0 1 1 1 1 0]]

    [[1 0 0 0 0 1 1 1 1 0]
    [1 0 0 0 0 1 1 1 1 0]]
    '''


    my_action_batch = np.zeros(my_handcards_batch.shape)
    for i, action in enumerate(infoset.legal_actions):
        # my_action_batch[i, :] = _cards2array(action)

        # TODO: not sure about this one
        my_action_batch[i, :] = action

    # 2d list storing other 3 players' card left, one hot array
    other_players_num_cards_left_batch = {}
    other_players_played_cards_batch = {}

    # return a list of other players' position enum
    # other_players = [pos for pos in Position if pos != infoset.player_position]
    for position, enum in Position.__members__.items():
        if position is infoset.player_position:
            break

        # get one-hot array from 1. no. of cards left and 2. played cards
        other_players_num_cards_left = _get_one_hot_array(infoset.num_cards_left_dict[position])
        other_players_played_cards = _get_one_hot_array(infoset.played_cards[position])

        other_players_num_cards_left_batch[position] = \
            np.repeat(other_players_num_cards_left[np.newaxis, :], num_legal_actions, axis=0)
        other_players_played_cards_batch[position] = \
            np.repeat(other_players_played_cards[np.newaxis, :], num_legal_actions, axis=0)

    x_batch = np.hstack((my_handcards_batch,
                         other_handcards_batch,
                         last_action_batch,
                         [batch for batch in other_players_num_cards_left_batch],
                         [batch for batch in other_players_played_cards_batch],
                         my_action_batch))
    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             last_action,
                             [batch for batch in other_players_num_cards_left_batch],
                             [batch for batch in other_players_played_cards_batch]))

    z = _action_seq_list2array(_process_action_seq(
        infoset.card_play_action_seq))
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    obs = {
            'position': 'landlord',
            'x_batch': x_batch.astype(np.float32),
            'z_batch': z_batch.astype(np.float32),
            'legal_actions': infoset.legal_actions,
            'x_no_action': x_no_action.astype(np.int8),
            'z': z.astype(np.int8),
          }

    return obs


def _get_one_hot_array(num_left_cards, max_num_cards):
    """
    A utility function to obtain one-hot endoding
    """
    one_hot = np.zeros(max_num_cards)
    one_hot[num_left_cards - 1] = 1

    return one_hot

# def _cards2array(list_cards):
#     """
#     A utility function that transforms the actions, i.e.,
#     A list of integers into card matrix. Here we remove
#     the six entries that are always zero and flatten the
#     the representations.
#     """
#     if len(list_cards) == 0:
#         return np.zeros(54, dtype=np.int8)

#     matrix = np.zeros([4, 13], dtype=np.int8)
#     jokers = np.zeros(2, dtype=np.int8)
#     counter = Counter(list_cards)
#     for card, num_times in counter.items():
#         if card < 20:
#             matrix[:, Card2Column[card]] = NumOnes2Array[num_times]
#         elif card == 20:
#             jokers[0] = 1
#         elif card == 30:
#             jokers[1] = 1
#     return np.concatenate((matrix.flatten('F'), jokers))

def _action_seq_list2array(action_seq_list):
    """
    A utility function to encode the historical moves.
    We encode the historical 15 actions. If there is
    no 15 actions, we pad the features with 0. Since
    three moves is a round in DouDizhu, we concatenate
    the representations for each consecutive three moves.
    Finally, we obtain a 5x162 matrix, which will be fed
    into LSTM for encoding.
    """
    action_seq_array = np.zeros((len(action_seq_list), 54))
    for row, list_cards in enumerate(action_seq_list):
        action_seq_array[row, :] = _cards2array(list_cards)
    action_seq_array = action_seq_array.reshape(5, 162)
    return action_seq_array

def _process_action_seq(sequence, length=15):
    """
    A utility function encoding historical moves. We
    encode 15 moves. If there is no 15 moves, we pad
    with zeros.
    """
    sequence = sequence[-length:].copy()
    if len(sequence) < length:
        empty_sequence = [[] for _ in range(length - len(sequence))]
        empty_sequence.extend(sequence)
        sequence = empty_sequence
    return sequence