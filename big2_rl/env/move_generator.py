import itertools
from big2_rl.env.settings import Settings


# determine if a list of length 5 (each element corresponding to a card value) is a valid straight but not SF
def is_valid_straight(input_combination):
    # convert each card value to its rank,
    # and check if each rank in the input combo has the same values as a potential straight
    # here [0,1,2,11,12] should return True since it contains the same values as [11,12,0,1,2]
    for possible_straight in Settings.get_attrs()['straight_orders']:
        # if intersection has length 5, then all elements in one set present in another
        if len(set(map(lambda x: x//4, input_combination)).intersection(set(possible_straight))) == 5:
            # check if all have same suits, if yes return false
            if len(set(map(lambda x: x % 4, input_combination))) == 1:
                return False
            else:
                return True
    return False  # we have to check every straight before we can say it is not a straight


# determine if a list of length 5 (each element corresponding to a card value) is a valid flush but not SF
def is_valid_flush(input_combination):
    if len(set(map(lambda x: x % 4, input_combination))) == 1:  # check if all have same suits, if no return false
        # if yes check if it is SF. Need to check all straights to ensure flush is not SF.
        for possible_straight in Settings.get_attrs()['straight_orders']:
            if len(set(map(lambda x: x//4, input_combination)).intersection(set(possible_straight))) == 5:
                return False
        return True
    else:
        return False


# determine if a list of length 5 (each element corresponding to a card value) is a valid SF
def is_valid_straight_flush(input_combination):
    for possible_straight in Settings.get_attrs()['straight_orders']:
        if len(set(map(lambda x: x//4, input_combination)).intersection(possible_straight)) == 5:
            # technically don't need the flush condition since self.flush_moves guarantees it
            return True
        else:
            return False


class MovesGener(object):
    """
    Generates the possible moves a given hand can make.
    """
    def __init__(self, cards_list):
        # cards_list is a sorted list of integers with each element from 0-51 inclusive
        # (in ascending order eg [3, 9, 24, 28, 30, 31])
        self.cards_list = cards_list

        self.single_card_moves = []
        self.gen_type_1_single()
        self.pair_moves = []
        self.gen_type_2_pair()
        self.triple_cards_moves = []
        self.gen_type_3_triple()
        self.flush_moves = []
        self.gen_type_5_flush()
    
    # returns a list of lists. Each list consists of a single element corresponding to integer id of that card
    def gen_type_1_single(self):
        self.single_card_moves = []
        for i in self.cards_list:
            self.single_card_moves.append([i])
        return self.single_card_moves
    
    # returns a list of lists.
    # Each list consists of 2 elements in ascending order, each corresponding to the int ids of the cards in a pair
    def gen_type_2_pair(self):
        self.pair_moves = []
        # for each card, check to see if 1 of the following 3 cards in the sorted list have the same rank. 
        for i in range(len(self.cards_list)):
            for j in range(1, 4):
                if i+j > len(self.cards_list) - 1:
                    break
                if (self.cards_list[i] // 4) == (self.cards_list[i+j] // 4):
                    self.pair_moves.append([self.cards_list[i], self.cards_list[i+j]])
                else:
                    # break out of loop early since (for instance if j=1 not same rank),
                    # then j=2 guaranteed not to be same rank
                    break
        return self.pair_moves

    # returns a list of lists.
    # Each list consists of 3 elements in ascending order, each corresponding to integer id of cards in a triple
    def gen_type_3_triple(self):
        self.triple_cards_moves = []
        # for each card, check to see if 2 of the following 3 cards in the sorted list have the same rank. 
        # suppose in the hand there exist cards w, x, y, z with the same rank.
        # Then wxy, wyz, wxz will get added when i is the index for card w,
        # and xyz will get added when i is the index for card x.
        for i in range(len(self.cards_list)):
            if i+2 > len(self.cards_list) - 1:  # if currently on 2nd last index, no more triples possible
                break
            if ((self.cards_list[i] // 4) == (self.cards_list[i+1] // 4)) and \
               ((self.cards_list[i] // 4) == (self.cards_list[i+2] // 4)):
                self.triple_cards_moves.append([self.cards_list[i],
                                                self.cards_list[i+1],
                                                self.cards_list[i+2]])
            # if currently on 3rd last index, no more triples possible besides the last 3 cards (accounted for above)
            if i+3 > len(self.cards_list) - 1:
                break
            if ((self.cards_list[i] // 4) == (self.cards_list[i+2] // 4)) and \
               ((self.cards_list[i] // 4) == (self.cards_list[i+3] // 4)):
                self.triple_cards_moves.append([self.cards_list[i],
                                                self.cards_list[i+2],
                                                self.cards_list[i+3]])
            if ((self.cards_list[i] // 4) == (self.cards_list[i+1] // 4)) and \
               ((self.cards_list[i] // 4) == (self.cards_list[i+3] // 4)):
                self.triple_cards_moves.append([self.cards_list[i],
                                                self.cards_list[i+1],
                                                self.cards_list[i+3]])
        return self.triple_cards_moves
    
    # returns a list of lists.
    # Each list consists of 5 elements corresponding to the 5 cards that make up a potential straight (but NOT SF)
    # if input hands are sorted in ascending order, the move's cards will be sorted in ascending order 3 < ... < 2
    def gen_type_4_straight(self):
        # TODO: need to stress test this function's performance for a hand like 3334445566777 or 3344455667788
        # worst comes to worst manually generate a massive lookup dictionary and use that lmao
        result = list()
        # https://stackoverflow.com/questions/27150990/python-itertools-combinations-how-to-obtain-the-indices-of-the-combined-numbers
        five_card_moves = list(itertools.combinations(self.cards_list, 5))  # generate all 5 card combos
        five_card_moves = [list(_) for _ in five_card_moves]  # fix tuple m.sort() error
        # filter() takes an iterable and a function and returns all items in iterable that return true for the function.
        # here it generates all combinations of 5 cards that are valid straights
        result.extend(filter(is_valid_straight, five_card_moves))
        return result
    
    # returns a list of lists. Each list consists of 5 elements corresponding to the 5 cards that make up a
    # potential flush (but NOT SF)
    # if input hands are sorted in ascending order, the move's cards will be sorted in ascending order 3 < ... < 2
    def gen_type_5_flush(self):
        # TODO: need to stress test this function's performance for a hand like A23456789TJQK all hearts
        self.flush_moves = []
        five_card_moves = list(itertools.combinations(self.cards_list, 5))  # generate all 5 card combos
        five_card_moves = [list(_) for _ in five_card_moves]
        # filter() takes an iterable and a function and returns all items in iterable that return true for the function.
        # here it generates all combinations of 5 cards that are valid flushes
        self.flush_moves.extend(filter(is_valid_flush, five_card_moves))
        return self.flush_moves

    # returns a list of lists.
    # Each list consists of 5 elements in ascending order, first 2 to the pair, last 3 to the triple.
    def gen_type_6_fullhouse(self):
        result = list()
        for t in self.triple_cards_moves:  # triples rarer than pairs so search those first (slight optimization)
            for p in self.pair_moves:
                if (t[0] // 4) != (p[0] // 4):  # need them to not be the same rank
                    result.append(p+t)
        return result

    # returns a list of lists.
    # Each list consists of 5 elements, with the 1st element being the integer id of the kicker,
    # and the other 4 being the integer id of the cards in a possible quad
    def gen_type_7_quads(self):
        result = list()
        # for each card, check to see if each of the following 3 cards in the sorted list have the same rank. 
        for i in range(len(self.cards_list)):
            if self.cards_list[i] % 4 != 0:  # a quad of a rank must include the lowest card in that rank
                continue
            
            if i+3 > len(self.cards_list) - 1:  # if currently on 3rd last index or after, no more quads possible
                break
            if ((self.cards_list[i] // 4) == (self.cards_list[i+1] // 4)) and \
               ((self.cards_list[i] // 4) == (self.cards_list[i+2] // 4)) and \
               ((self.cards_list[i] // 4) == (self.cards_list[i+3] // 4)):
                # if we found potential quads, find every possible kicker (has to be of different rank)
                for j in range(len(self.cards_list)):
                    if (self.cards_list[j] // 4) == (self.cards_list[i] // 4):
                        continue
                    kicker = self.cards_list[j]
                    result.append([kicker, self.cards_list[i], self.cards_list[i+1], self.cards_list[i+2],
                                   self.cards_list[i+3]])
        return result

    # returns a list of lists.
    # Each list consists of 5 elements, each corresponding to the integer id of the cards in a possible SF
    def gen_type_8_straightflush(self):
        result = list()
        result.extend(filter(is_valid_straight_flush, self.flush_moves))
        return result

    # generate all possible moves from given cards
    def gen_moves(self):
        moves = []
        moves.extend(self.gen_type_1_single())
        moves.extend(self.gen_type_2_pair())
        moves.extend(self.gen_type_3_triple())
        moves.extend(self.gen_type_4_straight())
        moves.extend(self.gen_type_5_flush())
        moves.extend(self.gen_type_6_fullhouse())
        moves.extend(self.gen_type_7_quads())
        moves.extend(self.gen_type_8_straightflush())
        return moves
