# TODO: having a class for settings might not be the best idea
class GameSettings:
    """
    defines the settings to use for Big Two (including penalties, flush/straight rankings
    """
    # if FLUSH_ORDERBY_RANK is used 2h Qh Jh 9h 4h > 2s Qs Ts 7s 6s
    # if FLUSH_ORDERBY_SUIT is used 2h Qh Jh 9h 4h < 2s Qs Ts 7s 6s since spade > heart > club > diamond
    # note that only one of these settings can be true at any given time!
    CONST_FLUSH_ORDERBY_RANK = 0
    CONST_FLUSH_ORDERBY_SUIT = 1

    PENALISE_QUADS = -1
    PENALISE_STRAIGHT_FLUSH = -1
    PENALISE_DEUCES = -1

    REWARD_QUADS = -1
    REWARD_STRAIGHT_FLUSH = -1
    REWARD_DEUCES = -1

    # 8-9 cards = double, 10-12 cards = triple, 13 cards = quadruple
    PENALTY_THRESHOLD = [8, 10, 13]

    # flush and straight ordering settings
    FLUSH_ORDERS = CONST_FLUSH_ORDERBY_RANK
    # a list of the rankings of the valid straights
    # default: 34567 < 45678 < ... < TJQKA < 23456 < A2345
    # JQKA2 is not a valid straight
    STRAIGHT_ORDERS = [[range(0, 5)],  # ranks 0-4 corresponds to 34567
                        [range(1, 6)], [range(2, 7)], [range(3, 8)], [range(4, 9)], [range(5, 10)],
                        [range(6, 11)], [range(7, 12)],  # ranks 7-11 corresponds to TJQKA
                        [12, 0, 1, 2, 3],  # 23456
                        [11, 12, 0, 1, 2]]  # A2345

    def __init__(self, penalise_quads=1, penalise_sf=1, penalise_deuces=2,
                 reward_quads=1, reward_sf=1, reward_deuces=1,
                 penalty_threshold=None):
        # in some variations if losing players still have deuces, 1 or more SF, or 1 or more quads the winner's reward is doubled
        self.PENALISE_QUADS = penalise_quads
        self.PENALISE_STRAIGHT_FLUSH = penalise_sf
        self.PENALISE_DEUCES = penalise_deuces

        # in some variations if winning player ends with a deuce, SF or quads their reward is doubled
        self.REWARD_QUADS = reward_quads
        self.REWARD_STRAIGHT_FLUSH = reward_sf
        self.REWARD_DEUCES = reward_deuces

        if penalty_threshold:
            self.PENALTY_THRESHOLD = penalty_threshold

    def set_straight_order(self, straight_order=None):
        """
        takes a 3 letter code corresponding to the lowest card of the highest 3 straights in the sequence.
        T2A assumes 34567 < ... < TJQKA < 23456 < A2345
        TA2 assumes 34567 < ... < TJQKA < A2345 < 23456
        9TA assumes 23456 < 34567 < ... < 9TJQK < TJQKA < A2345
        9AT assumes 23456 < 34567 < ... < 9TJQK < A2345 < TJQKA
        89T assumes A2345 < 23456 < 34567 < ... < 9TJQK < TJQKA
        """
        if straight_order == "T2A":
            pass  # default
        elif straight_order == "TA2":
            self.STRAIGHT_ORDERS = [[range(0, 5)],
                        [range(1, 6)], [range(2, 7)], [range(3, 8)], [range(4, 9)], [range(5, 10)],
                        [range(6, 11)], [range(7, 12)], [11, 12, 0, 1, 2], [12, 0, 1, 2, 3]]
        elif straight_order == "9TA":
            self.STRAIGHT_ORDERS = [[12, 0, 1, 2, 3], [range(0, 5)],
                                    [range(1, 6)], [range(2, 7)], [range(3, 8)], [range(4, 9)], [range(5, 10)],
                                    [range(6, 11)], [range(7, 12)], [11, 12, 0, 1, 2]]
        elif straight_order == "9AT":
            self.STRAIGHT_ORDERS = [[12, 0, 1, 2, 3], [range(0, 5)],
                                    [range(1, 6)], [range(2, 7)], [range(3, 8)], [range(4, 9)], [range(5, 10)],
                                    [range(6, 11)], [11, 12, 0, 1, 2], [range(7, 12)]]
        elif straight_order == "89T":
            self.STRAIGHT_ORDERS = [[11, 12, 0, 1, 2], [12, 0, 1, 2, 3], [range(0, 5)],
                                    [range(1, 6)], [range(2, 7)], [range(3, 8)], [range(4, 9)], [range(5, 10)],
                                    [range(6, 11)], [range(7, 12)]]
        else:
            raise Exception("invalid parameter for set_straight_order")

    def set_flush_order(self, flush_order=None):
        if flush_order == "suit":
            self.FLUSH_ORDERS = self.CONST_FLUSH_ORDERBY_SUIT
        elif flush_order == "rank":
            pass  # default
        else:
            raise Exception("invalid parameter for set_flush_order")
