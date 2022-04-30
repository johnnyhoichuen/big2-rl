from big2_rl.env.utils import *
import collections
from big2_rl.env.move_generator import *


# return the type of the move as a dict {'type':  ENUM}
def get_move_type(move):
    # 'move' is an array of integers corresponding to the cards played in that move.
    # Each element is integer from 0-51 inclusive eg [51, 49] represents the pair (2s, 2c)

    # in general, for a card of rank S (S in [0, 12]) and suit H (H in [0, 3]),
    # S*4 + H is that card's integer representation
    
    move_size = len(move)

    if move_size == 0:
        return {'type': TYPE_0_PASS}

    if move_size == 1:
        return {'type': TYPE_1_SINGLE}

    if move_size == 2:
        if (move[0] // 4) == (move[1] // 4):  # if same rank
            return {'type': TYPE_2_PAIR}
        else:  # two cards can only be played as a pair
            return {'type': TYPE_9_WRONG}

    if move_size == 3:
        if ((move[0] // 4) == (move[1] // 4)) and ((move[0] // 4) == (move[2] // 4)):
            return {'type': TYPE_3_TRIPLE}
        else:  # three cards can only be played as triple
            return {'type': TYPE_9_WRONG}

    if move_size == 5:
        # gets the ranks eg [2s 2h 2c 3h 3d] = [51, 50, 49, 6, 4] = [12 12 12 1 1]
        move_ranks = list(map(lambda x: x//4, move))
        move_ranks_dict = collections.Counter(move_ranks)  # gets number of each rank eg {12: 3, 1: 2}
        
        if len(move_ranks_dict) == 2:  # full house or quads
            if max(move_ranks_dict.values()) == 3:
                return {'type': TYPE_6_FULLHOUSE}
            elif max(move_ranks_dict.values()) == 4:
                return {'type': TYPE_7_QUADS}
            else:
                return {'type': TYPE_9_WRONG}
            
        elif len(move_ranks_dict) == 5:  # straight, flush or SF
            if is_valid_straight_flush(move):
                return {'type': TYPE_8_STRAIGHTFLUSH}
            if is_valid_straight(move):
                return {'type': TYPE_4_STRAIGHT}
            if is_valid_flush(move):
                return {'type': TYPE_5_FLUSH}
            return {'type': TYPE_9_WRONG}
        
        else:
            return {'type': TYPE_9_WRONG}

    return {'type': TYPE_9_WRONG}
