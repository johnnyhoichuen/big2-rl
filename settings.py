# should be shared across all modules

# default initialisation
penalise_quads = 1
penalise_sf = 1
penalise_deuces = 1

# in some variations if winning player ends with a deuce, SF or quads their reward is doubled
reward_quads = 1
reward_sf = 1
reward_deuces = 1

flush_orders = 0

# a list of the rankings of the valid straights
# default: 34567 < 45678 < ... < TJQKA < 23456 < A2345
# JQKA2 is not a valid straight
# default is T2A
straight_orders = [[_ for _ in range(0, 5)], [_ for _ in range(1, 6)], [_ for _ in range(2, 7)],
                         [_ for _ in range(3, 8)], [_ for _ in range(4, 9)],
                         [_ for _ in range(5, 10)], [_ for _ in range(6, 11)],
                         [_ for _ in range(7, 12)], [12, 0, 1, 2, 3], [11, 12, 0, 1, 2]]

# 8-9 cards = double, 10-12 cards = triple, 13 cards = quadruple
penalty_threshold = [8, 10, 13]
