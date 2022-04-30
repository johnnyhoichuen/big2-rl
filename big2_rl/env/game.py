from copy import deepcopy
from big2_rl.env import move_detector as md, move_selector as ms
from big2_rl.env.move_generator import MovesGener
import settings
from enum import Enum, unique


@unique
class Position(Enum):
    # DON'T CHANGE THE VALUES or re-indexing will have to be done
    SOUTH = 0
    EAST = 1
    NORTH = 2
    WEST = 3


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
        self.last_move_dict = {}

        # (dict String : list of int) stores the integer ids of ALL cards played by each position
        self.played_cards = {}

        # (list of int) stores the integer ids of the cards played in the last move made
        self.last_move = []

        # (list of int) stores the integer ids of the cards played in the last 2 moves made
        self.last_two_moves = []

        # (dict String : int) stores number of games won by each player
        self.num_wins = {}

        # (dict String : int) stores total amount won by each player
        self.num_scores = {}

        # (dict String : InfoSet) stores all information about the current game state
        self.info_sets = {}

        self.initialise_positional_dicts()  # initialise all dicts with positions (SENW) as the keys

        # (String) stores information about who was the previous player
        self.last_pid = None

        # (InfoSet) stores information about the information available to the current player
        self.game_infoset = None

        # (String) stores information about winner of the current deal
        self.winner = None

    def initialise_positional_dicts(self, reset=True):
        """
        initialise/reset all dictionaries with positions (SENW) as the keys
        """
        for pos in Position:
            self.last_move_dict[pos.name] = []
            self.played_cards[pos.name] = []
            if reset:  # don't reset these between deals
                self.num_wins[pos.name] = 0
                self.num_scores[pos.name] = 0
            self.info_sets[pos.name] = InfoSet(pos.name)

    def card_play_init(self, card_play_data):
        """
        initialises each player's hand cards
        Can be also called externally to preset the cards for a deal (in evaluation/simulation.py)

        Arguments:
        card_play_data:
            dict String : list of int
            something like 'north':deck[:13] where deck is list of int in a shuffled deck
        """
        for pos in Position:
            self.info_sets[pos.name].player_hand_cards = card_play_data[pos.name]

        self.get_acting_player_position()
        self.game_infoset = self.get_infoset()

    def game_done(self):
        """
        checks to see if a given game is over (i.e. one player has emptied their hand)
        """
        for pos in Position:
            if len(self.info_sets[pos.name].player_hand_cards) == 0:
                self.compute_player_reward()
                self.game_over = True
                break

    def compute_player_reward(self):
        """
        computes reward given the remaining players and updates `self.num_wins` and `self.num_scores`
        """
        count = 0
        self.player_reward_dict = {}
        for pos in Position:
            hand_size = len(self.info_sets[pos.name].player_hand_cards)
            if hand_size > 0:
                penalty_multiplier = 1

                # TODO fix bug for penalty_threshold
                for i in range(len(settings.penalty_threshold)):
                    if hand_size >= settings.penalty_threshold[i]:
                        penalty_multiplier += 1

                # TODO more penalty multiplier logic needed here
                self.player_reward_dict[pos.name] = -hand_size * penalty_multiplier
                self.num_scores[pos.name] -= hand_size * penalty_multiplier
                count += hand_size * penalty_multiplier
            elif hand_size == 0:
                self.winner = pos.name
                self.num_wins[pos.name] += 1

        self.player_reward_dict[self.winner] = count
        self.num_scores[self.winner] += count

    def get_winner(self):
        return self.winner

    def step(self):
        """
        have the current player/agent act
        """
        # self.players[pos] for each position pos is an Agent. each Agent defines the Agent.act() function which takes
        # an infoset (the information available to that player) as argument and decides on a legal action to make

        # PPOAgent has a different act() call compared to DMC and Random
        if self.players[self.acting_player_position].__class__.__name__ == "PPOAgent":
            action = self.players[self.acting_player_position].act(self.game_infoset, self.card_play_action_seq)
        else:
            action = self.players[self.acting_player_position].act(self.game_infoset)
        assert action in self.game_infoset.legal_actions

        # if move is not pass, update the acting player to be the current one
        if len(action) > 0:
            self.last_pid = self.acting_player_position

        # update self.last_move_dict
        self.last_move_dict[
            self.acting_player_position] = action.copy()

        # update the sequence of moves played so far, and remove cards from the player hand used in the move
        self.card_play_action_seq.append(action)
        self.update_acting_player_hand_cards(action)

        # update the set of played cards in that position
        self.played_cards[self.acting_player_position] += action

        # check if game is done
        self.game_done()
        if not self.game_over:
            self.get_acting_player_position()  # move to the next player
            self.game_infoset = self.get_infoset()  # update self.game_infoset

    def get_last_move(self):
        """
        get the last move made, i.e. a list of ints corresponding to the cards of the last move made
        TODO: is this never used?
        """
        last_move = []
        if len(self.card_play_action_seq) != 0:
            if len(self.card_play_action_seq[-1]) == 0:
                if len(self.card_play_action_seq[-2]) == 0:
                    if len(self.card_play_action_seq[-3]) == 0:
                        last_move = []
                    else:
                        last_move = self.card_play_action_seq[-3]
                else:
                    last_move = self.card_play_action_seq[-2]
            else:
                last_move = self.card_play_action_seq[-1]

        return last_move

    def get_last_two_moves(self):
        """
        get the last 2 moves made, i.e. a list of ints corresponding to the cards of the last move made
        TODO: is this never used?
        """
        last_two_moves = [[], []]
        for card in self.card_play_action_seq[-2:]:
            last_two_moves.insert(0, card)
            last_two_moves = last_two_moves[:2]
        return last_two_moves

    def get_acting_player_position(self):
        """
        updates self.acting_player_position to the next player to play
        """
        if self.acting_player_position is None:
            for pos in Position:
                if 0 in self.info_sets[pos.name].player_hand_cards:
                    self.acting_player_position = pos.name
                    break
        else:
            ind = Position[self.acting_player_position].value  # e.g.: given 'SOUTH' get 0
            self.acting_player_position = Position((ind + 1) % 4).name

        return self.acting_player_position

    def update_acting_player_hand_cards(self, action):
        """
        if acting player did not pass, remove the cards they played from their hand
        """
        if action != []:
            for card in action:
                self.info_sets[self.acting_player_position].player_hand_cards.remove(card)
            self.info_sets[self.acting_player_position].player_hand_cards.sort()

    def get_legal_card_play_actions(self):
        """
        get list of possible moves that a given player can make
        """
        mg = MovesGener(
            self.info_sets[self.acting_player_position].player_hand_cards)

        # `action_sequence` is a list of the previous moves played in chronological order
        action_sequence = self.card_play_action_seq

        # get the most recent non-pass move and store this in `rival_move`
        # if action sequence is empty or 3 most recent moves are pass (0 length), treat rival move as []
        rival_move = []
        if len(action_sequence) != 0:
            if len(action_sequence[-1]) == 0:
                if len(action_sequence[-2]) == 0:
                    if len(action_sequence[-3]) == 0:
                        rival_move = []
                    else:
                        rival_move = action_sequence[-3]
                else:
                    rival_move = action_sequence[-2]
            else:
                rival_move = action_sequence[-1]

        # get the type of the previous move played
        rival_type = md.get_move_type(rival_move)
        rival_move_type = rival_type['type']
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

        elif rival_move_type == md.TYPE_4_STRAIGHT:
            all_moves = mg.gen_type_4_straight()
            moves = ms.filter_type_4_straight(all_moves, rival_move)
            moves += mg.gen_type_5_flush() + mg.gen_type_6_fullhouse() + mg.gen_type_7_quads() + mg.\
                gen_type_8_straightflush()

        elif rival_move_type == md.TYPE_5_FLUSH:
            all_moves = mg.gen_type_5_flush()
            moves = ms.filter_type_5_flush(all_moves, rival_move)
            moves += mg.gen_type_6_fullhouse() + mg.gen_type_7_quads() + mg.gen_type_8_straightflush()

        elif rival_move_type == md.TYPE_6_FULLHOUSE:
            all_moves = mg.gen_type_6_fullhouse()
            moves = ms.filter_type_6_fullhouse(all_moves, rival_move)
            moves += mg.gen_type_7_quads() + mg.gen_type_8_straightflush()

        elif rival_move_type == md.TYPE_7_QUADS:
            all_moves = mg.gen_type_7_quads()
            moves = ms.filter_type_7_quads(all_moves, rival_move)
            moves += mg.gen_type_8_straightflush()

        elif rival_move_type == md.TYPE_8_STRAIGHTFLUSH:
            all_moves = mg.gen_type_8_straightflush()
            moves = ms.filter_type_8_straightflush(all_moves, rival_move)

        if len(rival_move) != 0:  # if rival move is not pass
            moves = moves + [[]]

        for m in moves:  # for each move, sort integer ids of each card to be in ascending order
            m.sort()

        return moves

    def reset(self):
        """
        reset attributes for the next deal
        """
        self.card_play_action_seq = []

        self.game_over = False

        self.acting_player_position = None
        self.player_reward_dict = None

        self.last_move = []
        self.last_two_moves = []

        self.initialise_positional_dicts(reset=False)

        self.last_pid = None

    def get_infoset(self):
        """
        the infoset contains all the information currently available to a given player
        however get_infoset() returns the infoset for each player to describe the total state of the game
        """

        # only attribute that isn't returned is self.player_hand_cards

        self.info_sets[self.acting_player_position].num_cards_left_dict = \
            {pos.name: len(self.info_sets[pos.name].player_hand_cards)
             for pos in Position}

        self.info_sets[self.acting_player_position].card_play_action_seq = \
            self.card_play_action_seq

        self.info_sets[self.acting_player_position].other_hand_cards = []

        for pos in Position:
            if pos.name != self.acting_player_position:
                self.info_sets[
                    self.acting_player_position].other_hand_cards += \
                    self.info_sets[pos.name].player_hand_cards

        self.info_sets[
            self.acting_player_position].legal_actions = \
            self.get_legal_card_play_actions()

        self.info_sets[
            self.acting_player_position].last_move = self.get_last_move()

        self.info_sets[
            self.acting_player_position].last_two_moves = self.get_last_two_moves()

        self.info_sets[
            self.acting_player_position].last_move_dict = self.last_move_dict

        self.info_sets[self.acting_player_position].played_cards = \
            self.played_cards

        self.info_sets[
            self.acting_player_position].last_pid = self.last_pid

        return deepcopy(self.info_sets[self.acting_player_position])


class InfoSet(object):
    """
    The game state is described as infoset, which
    includes all the information in the current situation,
    such as the hand cards of all players, the
    historical moves, etc.
    """
    def __init__(self, player_position):
        # (String) the player position (must be element of Position enum)
        self.player_position = player_position
        # (list of int) The hand cards of the current player
        self.player_hand_cards = None
        # (dict String : int) The number of cards in each player's hand
        self.num_cards_left_dict = None
        # (list of list of int) A list of all moves played. Each move is a list of ints (or [] if pass)
        self.card_play_action_seq = None
        # (list of int) for the current player, this is the union of the hand cards of all other players
        self.other_hand_cards = None
        # (list of list of int) Given the most recent non-pass move, this is a list of moves that this player can make
        self.legal_actions = None
        # (list of int) The most recent valid move
        self.last_move = None
        # (list of list of int) The most recent two moves
        self.last_two_moves = None
        # (dict String : int) The last moves for all the positions
        self.last_move_dict = None
        # (dict String : (list of int)) The played cards so far for each position.
        self.played_cards = None
        # removed (redundant with self.played_cards)
        # self.all_handcards = None
        # (String) Last player position that plays a valid move, i.e., not `pass`
        self.last_pid = None
