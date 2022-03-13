from copy import deepcopy
from . import move_detector as md, move_selector as ms
from .move_generator import MovesGener


class GameEnv(object):
    """
    GameEnv object describes the game environment for one deal of Big 2
    directly used by `./env.py` and `../evaluation/simulation.py`
    """
    def __init__(self, players):

        # (list of int) stores list of the cards played so far
        self.card_play_action_seq = []

        # (bool) whether the current deal is over
        self.game_over = False

        # (String) storing the acting player (one of 'south', 'east', 'north', 'west') aka SENW
        self.acting_player_position = None

        # (dict String : int) that stores the rewards for each player in SENW positions
        self.player_reward_dict = None

        # (dict String : Agent) stores the agent for each of the SENW positions
        self.players = players

        # (dict String : list of int) stores the integer ids of the cards played in the last move made by each position
        self.last_move_dict = {'south': [],
                               'east': [],
                               'north': [],
                               'west': []}

        # (dict String : list of int) stores the integer ids of ALL cards played by each position
        self.played_cards = {'south': [],
                             'east': [],
                             'north': [],
                             'west': []}

        # (list of int) stores the integer ids of the cards played in the last move made
        self.last_move = []

        # (list of int) stores the integer ids of the cards played in the last 2 moves made
        self.last_two_moves = []

        # (dict String : int) stores number of games won by each player
        self.num_wins = {'south': [], 'east': [], 'north': [], 'west': []}

        # (dict String : int) stores total amount won by each player
        self.num_scores = {'south': [], 'east': [], 'north': [], 'west': []}

        # (dict String : InfoSet) stores all information about the current game state
        self.info_sets = {'south': InfoSet('south'),
                          'east': InfoSet('east'),
                          'north': InfoSet('north'),
                          'west': InfoSet('west')}

        # (String) stores information about who was the previous player
        self.last_pid = None

        # (InfoSet) stores information about the information available to the current player
        self.game_infoset = None

        # (String) stores information about winner of the current deal
        self.winner = None

    def card_play_init(self, card_play_data):
        """
        initialises each player's hand cards

        Arguments:
        card_play_data:
            dict String : list of int
            something like 'north':deck[:13] where deck is list of int in a shuffled deck
        """
        self.info_sets['north'].player_hand_cards = \
            card_play_data['north']
        self.info_sets['east'].player_hand_cards = \
            card_play_data['east']
        self.info_sets['west'].player_hand_cards = \
            card_play_data['west']
        self.info_sets['south'].player_hand_cards = \
            card_play_data['south']
        # TODO need to declare the player who has the 3d is the one to start
        self.get_acting_player_position()
        self.game_infoset = self.get_infoset()

    def game_done(self):
        """
        checks to see if a given game is over (i.e. one player has emptied their hand)
        """
        if len(self.info_sets['north'].player_hand_cards) == 0 or \
                len(self.info_sets['east'].player_hand_cards) == 0 or \
                len(self.info_sets['west'].player_hand_cards) == 0 or \
                len(self.info_sets['south'].player_hand_cards) == 0:
            self.compute_player_reward()
            self.game_over = True

    def compute_player_reward(self):
        """
        computes reward given the remaining players and updates `self.num_wins` and `self.num_scores`
        """
        count = 0
        self.player_reward_dict = {}
        positions = ["north", "east", "west", "south"]
        for pos in positions:
            hand_size = len(self.info_sets[pos].player_hand_cards)
            if hand_size > 0:
                penalty_multiplier = 1
                for i in range(len(GameSettings.PENALTY_THRESHOLD)):
                    if hand_size >= GameSettings.PENALTY_THRESHOLD[i]:
                        penalty_multiplier += 1
                # TODO more penalty multiplier logic needed here
                self.player_reward_dict[pos] = -hand_size * penalty_multiplier
                self.num_scores[pos] -= hand_size * penalty_multiplier
                count += hand_size * penalty_multiplier
            elif hand_size == 0:
                self.winner = pos
                self.num_wins[pos] += 1

        self.player_reward_dict[self.winner] = count
        self.num_scores[self.winner] += count

    def get_winner(self):
        return self.winner

    def step(self):
        """
        have the current player act
        TODO here onwards
        """
        action = self.players[self.acting_player_position].act(
            self.game_infoset)
        assert action in self.game_infoset.legal_actions

        if len(action) > 0:
            self.last_pid = self.acting_player_position

        self.last_move_dict[
            self.acting_player_position] = action.copy()

        self.card_play_action_seq.append(action)
        self.update_acting_player_hand_cards(action)

        self.played_cards[self.acting_player_position] += action

        self.game_done()
        if not self.game_over:
            self.get_acting_player_position()
            self.game_infoset = self.get_infoset()

    def get_last_move(self):
        last_move = []
        if len(self.card_play_action_seq) != 0:
            if len(self.card_play_action_seq[-1]) == 0:
                last_move = self.card_play_action_seq[-2]
            else:
                last_move = self.card_play_action_seq[-1]

        return last_move

    def get_last_two_moves(self):
        last_two_moves = [[], []]
        for card in self.card_play_action_seq[-2:]:
            last_two_moves.insert(0, card)
            last_two_moves = last_two_moves[:2]
        return last_two_moves

    def get_acting_player_position(self):
        if self.acting_player_position is None:
            self.acting_player_position = 'landlord'

        else:
            if self.acting_player_position == 'landlord':
                self.acting_player_position = 'landlord_down'

            elif self.acting_player_position == 'landlord_down':
                self.acting_player_position = 'landlord_up'

            else:
                self.acting_player_position = 'landlord'

        return self.acting_player_position

    def update_acting_player_hand_cards(self, action):
        if action != []:
            for card in action:
                self.info_sets[
                    self.acting_player_position].player_hand_cards.remove(card)
            self.info_sets[self.acting_player_position].player_hand_cards.sort()

    def get_legal_card_play_actions(self):
        mg = MovesGener(
            self.info_sets[self.acting_player_position].player_hand_cards)

        action_sequence = self.card_play_action_seq

        rival_move = []
        if len(action_sequence) != 0:
            if len(action_sequence[-1]) == 0:
                rival_move = action_sequence[-2]
            else:
                rival_move = action_sequence[-1]

        rival_type = md.get_move_type(rival_move)
        rival_move_type = rival_type['type']
        rival_move_len = rival_type.get('len', 1)
        moves = list()

        if rival_move_type == md.TYPE_0_PASS:
            moves = mg.gen_moves()

        elif rival_move_type == md.TYPE_1_SINGLE:
            all_moves = mg.gen_type_1_single()
            moves = ms.filter_type_1_single(all_moves, rival_move)

        elif rival_move_type == md.TYPE_2_PAIR:
            all_moves = mg.gen_type_2_pair()
            moves = ms.filter_type_2_pair(all_moves, rival_move)

        elif rival_move_type == md.TYPE_3_TRIPLE:
            all_moves = mg.gen_type_3_triple()
            moves = ms.filter_type_3_triple(all_moves, rival_move)

        elif rival_move_type == md.TYPE_4_BOMB:
            all_moves = mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb()
            moves = ms.filter_type_4_bomb(all_moves, rival_move)

        elif rival_move_type == md.TYPE_5_KING_BOMB:
            moves = []

        elif rival_move_type == md.TYPE_6_3_1:
            all_moves = mg.gen_type_6_3_1()
            moves = ms.filter_type_6_3_1(all_moves, rival_move)

        elif rival_move_type == md.TYPE_7_3_2:
            all_moves = mg.gen_type_7_3_2()
            moves = ms.filter_type_7_3_2(all_moves, rival_move)

        elif rival_move_type == md.TYPE_8_SERIAL_SINGLE:
            all_moves = mg.gen_type_8_serial_single(repeat_num=rival_move_len)
            moves = ms.filter_type_8_serial_single(all_moves, rival_move)

        elif rival_move_type == md.TYPE_9_SERIAL_PAIR:
            all_moves = mg.gen_type_9_serial_pair(repeat_num=rival_move_len)
            moves = ms.filter_type_9_serial_pair(all_moves, rival_move)

        elif rival_move_type == md.TYPE_10_SERIAL_TRIPLE:
            all_moves = mg.gen_type_10_serial_triple(repeat_num=rival_move_len)
            moves = ms.filter_type_10_serial_triple(all_moves, rival_move)

        elif rival_move_type == md.TYPE_11_SERIAL_3_1:
            all_moves = mg.gen_type_11_serial_3_1(repeat_num=rival_move_len)
            moves = ms.filter_type_11_serial_3_1(all_moves, rival_move)

        elif rival_move_type == md.TYPE_12_SERIAL_3_2:
            all_moves = mg.gen_type_12_serial_3_2(repeat_num=rival_move_len)
            moves = ms.filter_type_12_serial_3_2(all_moves, rival_move)

        elif rival_move_type == md.TYPE_13_4_2:
            all_moves = mg.gen_type_13_4_2()
            moves = ms.filter_type_13_4_2(all_moves, rival_move)

        elif rival_move_type == md.TYPE_14_4_22:
            all_moves = mg.gen_type_14_4_22()
            moves = ms.filter_type_14_4_22(all_moves, rival_move)

        if rival_move_type not in [md.TYPE_0_PASS,
                                   md.TYPE_4_BOMB, md.TYPE_5_KING_BOMB]:
            moves = moves + mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb()

        if len(rival_move) != 0:  # rival_move is not 'pass'
            moves = moves + [[]]

        for m in moves:
            m.sort()

        return moves

    def reset(self):
        self.card_play_action_seq = []

        self.three_landlord_cards = None
        self.game_over = False

        self.acting_player_position = None
        self.player_reward_dict = None

        self.last_move_dict = {'landlord': [],
                               'landlord_up': [],
                               'landlord_down': []}

        self.played_cards = {'landlord': [],
                             'landlord_up': [],
                             'landlord_down': []}

        self.last_move = []
        self.last_two_moves = []

        self.info_sets = {'landlord': InfoSet('landlord'),
                         'landlord_up': InfoSet('landlord_up'),
                         'landlord_down': InfoSet('landlord_down')}

        self.bomb_num = 0
        self.last_pid = 'landlord'

    def get_infoset(self):
        self.info_sets[
            self.acting_player_position].last_pid = self.last_pid

        self.info_sets[
            self.acting_player_position].legal_actions = \
            self.get_legal_card_play_actions()

        self.info_sets[
            self.acting_player_position].bomb_num = self.bomb_num

        self.info_sets[
            self.acting_player_position].last_move = self.get_last_move()

        self.info_sets[
            self.acting_player_position].last_two_moves = self.get_last_two_moves()

        self.info_sets[
            self.acting_player_position].last_move_dict = self.last_move_dict

        self.info_sets[self.acting_player_position].num_cards_left_dict = \
            {pos: len(self.info_sets[pos].player_hand_cards)
             for pos in ['landlord', 'landlord_up', 'landlord_down']}

        self.info_sets[self.acting_player_position].other_hand_cards = []
        for pos in ['landlord', 'landlord_up', 'landlord_down']:
            if pos != self.acting_player_position:
                self.info_sets[
                    self.acting_player_position].other_hand_cards += \
                    self.info_sets[pos].player_hand_cards

        self.info_sets[self.acting_player_position].played_cards = \
            self.played_cards
        self.info_sets[self.acting_player_position].three_landlord_cards = \
            self.three_landlord_cards
        self.info_sets[self.acting_player_position].card_play_action_seq = \
            self.card_play_action_seq

        self.info_sets[
            self.acting_player_position].all_handcards = \
            {pos: self.info_sets[pos].player_hand_cards
             for pos in ['landlord', 'landlord_up', 'landlord_down']}

        return deepcopy(self.info_sets[self.acting_player_position])

class InfoSet(object):
    """
    The game state is described as infoset, which
    includes all the information in the current situation,
    such as the hand cards of the three players, the
    historical moves, etc.
    """
    def __init__(self, player_position):
        # The player position, i.e., landlord, landlord_down, or landlord_up
        self.player_position = player_position
        # The hand cands of the current player. A list.
        self.player_hand_cards = None
        # The number of cards left for each player. It is a dict with str-->int 
        self.num_cards_left_dict = None
        # The three landload cards. A list.
        self.three_landlord_cards = None
        # The historical moves. It is a list of list
        self.card_play_action_seq = None
        # The union of the hand cards of the other two players for the current player 
        self.other_hand_cards = None
        # The legal actions for the current move. It is a list of list
        self.legal_actions = None
        # The most recent valid move
        self.last_move = None
        # The most recent two moves
        self.last_two_moves = None
        # The last moves for all the postions
        self.last_move_dict = None
        # The played cands so far. It is a list.
        self.played_cards = None
        # The hand cards of all the players. It is a dict. 
        self.all_handcards = None
        # Last player position that plays a valid move, i.e., not `pass`
        self.last_pid = None
        # The number of bombs played so far
        self.bomb_num = None
