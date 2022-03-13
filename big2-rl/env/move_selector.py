# return a list of all moves that can beat an opponent
from .utils import *

# `moves` is a list of a move of a particular type (eg pair) where each element is a move represented by a list of integer ids
# `opponent_move` is a list of integer ids corresponding to the cards in the move an opponent has made
def compare_max(moves, opponent_move):
    new_moves = list()
    for move in moves:
        # since each move is given in ascending order, only compare the highest card for singles, pairs, triples, full houses and quads
        if move[len(move)-1] > opponent_move[len(move)-1]:
            new_moves.append(move)
    return new_moves

def filter_type_1_single(moves, opponent_move):
    return compare_max(moves, opponent_move)

def filter_type_2_pair(moves, opponent_move):
    return compare_max(moves, opponent_move)

def filter_type_3_triple(moves, opponent_move):
    return compare_max(moves, opponent_move)

def filter_type_4_straight(moves, opponent_move):
    new_moves = []
    for move in moves:
        move_ranks = set(map(lambda x: x//4, move))
        opp_move_ranks = set(map(lambda x: x//4, opponent_move))
        
        move_straight_index = -1
        opp_straight_index = -1
        
        for s_i in range(len(STRAIGHT_ORDERS)):
            if len(move_ranks.intersection(STRAIGHT_ORDERS[s_i]))==5:
                move_straight_index = s_i
            if len(opp_move_ranks.intersection(STRAIGHT_ORDERS[s_i]))==5:
                opp_straight_index = s_i
                
        if move_straight_index==-1 or opp_straight_index==-1:
            raise Exception("Someone doesn't have a straight")
        
        if move_straight_index > opp_straight_index: # new move has higher straight rank
            new_moves.append(move)
        elif move_straight_index == opp_straight_index: # same rank but move has higher suit
            if move[len(move)-1] > opponent_move[len(move)-1]:
                new_moves.append(move)
        
    return new_moves
        
def filter_type_5_flush(moves, opponent_move):
    new_moves = []
    for move in moves:
        if FLUSH_ORDERBY_SUIT and FLUSH_ORDERBY_RANK:
            raise Exception("inconsistent settings for flush ranking")
            
        if FLUSH_ORDERBY_SUIT: # first compare by suit, then by rank
            if list(map(lambda x: x%4, move))[0] > list(map(lambda x: x%4, opponent_move))[0]:
                new_moves.append(move)
            elif list(map(lambda x: x%4, move))[0] == list(map(lambda x: x%4, opponent_move))[0]:
                if move[len(move)-1] > opponent_move[len(move)-1]: # if both flushes of the same suit the winner is the one with higher rank
                    new_moves.append(move)
                    
        elif FLUSH_ORDERBY_RANK: # first compare by rank, then by suit
            move_ranks = list(map(lambda x: x//4, move))[::-1]
            opp_move_ranks = list(map(lambda x: x//4, opponent_move))[::-1]
            if move_ranks > opp_move_ranks: # does elementwise comparison of the ranks in each flush in descending order
                new_moves.append(move)
            elif move_ranks == opp_move_ranks:
                if move[len(move)-1] > opponent_move[len(move)-1]:
                    new_moves.append(move)
                    
    return new_moves

def filter_type_6_fullhouse(moves, opponent_move):
    return compare_max(moves, opponent_move)

def filter_type_7_quads(moves, opponent_move):
    return compare_max(moves, opponent_move)

def filter_type_8_straightflush(moves, opponent_move):
    return filter_type_4_straight(moves, opponent_move)
