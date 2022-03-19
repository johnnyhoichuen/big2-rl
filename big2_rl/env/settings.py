# should only create one instance of this class
# singleton: https://www.tutorialspoint.com/python_design_patterns/python_design_patterns_singleton.htm

class GameSettings:
    """
    defines the settings to use for Big Two (including penalties, flush/straight rankings)
    """
    __instance = None

    # if FLUSH_ORDERBY_RANK is used 2h Qh Jh 9h 4h > 2s Qs Ts 7s 6s
    # if FLUSH_ORDERBY_SUIT is used 2h Qh Jh 9h 4h < 2s Qs Ts 7s 6s since spade > heart > club > diamond
    # note that only one of these settings can be true at any given time!
    CONST_FLUSH_ORDERBY_RANK = 0
    CONST_FLUSH_ORDERBY_SUIT = 1

    @staticmethod
    def getInstance():
        """Static access method."""
        if not GameSettings.__instance:
            GameSettings.__instance = GameSettings()
        return GameSettings.__instance

    def __init__(self, penalise_quads=1, penalise_sf=1, penalise_deuces=2,
                 reward_quads=1, reward_sf=1, reward_deuces=1,
                 penalty_threshold=None):

        # in some variations if losing players still have deuces, 1 or more SF, or 1 or more quads, then
        # the winner's reward is doubled
        self._penalise_quads = penalise_quads
        self._penalise_sf = penalise_sf
        self._penalise_deuces = penalise_deuces

        # in some variations if winning player ends with a deuce, SF or quads their reward is doubled
        self._reward_quads = reward_quads
        self._reward_sf = reward_sf
        self._reward_deuces = reward_deuces

        self._flush_orders = GameSettings.CONST_FLUSH_ORDERBY_RANK

        # a list of the rankings of the valid straights
        # default: 34567 < 45678 < ... < TJQKA < 23456 < A2345
        # JQKA2 is not a valid straight
        self._straight_orders = [[range(0, 5)],  # ranks 0-4 corresponds to 34567
                        [range(1, 6)], [range(2, 7)], [range(3, 8)], [range(4, 9)], [range(5, 10)],
                        [range(6, 11)], [range(7, 12)],  # ranks 7-11 corresponds to TJQKA
                        [12, 0, 1, 2, 3],  # 23456
                        [11, 12, 0, 1, 2]]  # A2345

        # 8-9 cards = double, 10-12 cards = triple, 13 cards = quadruple
        if penalty_threshold:
            self._penalty_threshold = penalty_threshold
        else:
            self._penalty_threshold = [8, 10, 13]

    def set_straight_order(self, straight_order=None):
        """
        takes a 3 letter code corresponding to the lowest card of the highest 3 straights in the sequence.
        T2A assumes 34567 < ... < TJQKA < 23456 < A2345
        TA2 assumes 34567 < ... < TJQKA < A2345 < 23456
        9TA assumes 23456 < 34567 < ... < 9TJQK < TJQKA < A2345
        9AT assumes 23456 < 34567 < ... < 9TJQK < A2345 < TJQKA
        89T assumes A2345 < 23456 < 34567 < ... < 9TJQK < TJQKA
        """
        if straight_order == "T2A":  # default
            self._straight_orders = [[range(0, 5)],  # ranks 0-4 corresponds to 34567
                                     [range(1, 6)], [range(2, 7)], [range(3, 8)], [range(4, 9)], [range(5, 10)],
                                     [range(6, 11)], [range(7, 12)],  # ranks 7-11 corresponds to TJQKA
                                     [12, 0, 1, 2, 3],  # 23456
                                     [11, 12, 0, 1, 2]]
        elif straight_order == "TA2":
            self._straight_orders = [[range(0, 5)],
                        [range(1, 6)], [range(2, 7)], [range(3, 8)], [range(4, 9)], [range(5, 10)],
                        [range(6, 11)], [range(7, 12)], [11, 12, 0, 1, 2], [12, 0, 1, 2, 3]]
        elif straight_order == "9TA":
            self._straight_orders = [[12, 0, 1, 2, 3], [range(0, 5)],
                                    [range(1, 6)], [range(2, 7)], [range(3, 8)], [range(4, 9)], [range(5, 10)],
                                    [range(6, 11)], [range(7, 12)], [11, 12, 0, 1, 2]]
        elif straight_order == "9AT":
            self._straight_orders = [[12, 0, 1, 2, 3], [range(0, 5)],
                                    [range(1, 6)], [range(2, 7)], [range(3, 8)], [range(4, 9)], [range(5, 10)],
                                    [range(6, 11)], [11, 12, 0, 1, 2], [range(7, 12)]]
        elif straight_order == "89T":
            self._straight_orders = [[11, 12, 0, 1, 2], [12, 0, 1, 2, 3], [range(0, 5)],
                                    [range(1, 6)], [range(2, 7)], [range(3, 8)], [range(4, 9)], [range(5, 10)],
                                    [range(6, 11)], [range(7, 12)]]
        else:
            raise Exception("invalid parameter for set_straight_order")

    def set_flush_order(self, flush_order=None):
        if flush_order == "suit":
            self._flush_orders = self.CONST_FLUSH_ORDERBY_SUIT
        elif flush_order == "rank":
            self._flush_orders = self.CONST_FLUSH_ORDERBY_RANK  # default
        else:
            raise Exception("invalid parameter for set_flush_order")

    def get_attrs(self):
        return {'penalise_quads': self._penalise_quads,
                'penalise_sf': self._penalise_sf,
                'penalise_deuces': self._penalise_deuces,
                'reward_quads': self._reward_quads,
                'reward_sf': self._reward_sf,
                'reward_deuces': self._reward_deuces,
                'flush_orders': self._flush_orders,
                'straight_orders': self._straight_orders,
                'penalty_threshold': self._penalty_threshold
                }
