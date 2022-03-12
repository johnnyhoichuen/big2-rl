# Action types. More can be defined here if necessary.
TYPE_0_PASS = 0
TYPE_1_SINGLE = 1
TYPE_2_PAIR = 2
TYPE_3_TRIPLE = 3
TYPE_4_STRAIGHT = 4 # doesn't include straight flushes
TYPE_5_FLUSH = 5 # doesn't include straight flushes
TYPE_6_FULLHOUSE = 6
TYPE_7_QUADS = 7
TYPE_8_STRAIGHTFLUSH = 8
TYPE_9_WRONG = 9

# a list of the rankings of the valid straights
# default: 34567 < 45678 < ... < TJQKA < 23456 < A2345
# JQKA2 is not a valid straight
STRAIGHT_ORDERS = [
    [range(0,5)], # ranks 0-4 corresponds to 34567
    [range(1,6)],
    [range(2,7)],
    [range(3,8)],
    [range(4,9)],
    [range(5,10)],
    [range(6,11)],
    [range(7,12)], # ranks 7-11 corresponds to TJQKA
    [12,0,1,2,3], # 23456
    [11,12,0,1,2] # A2345
]

# if FLUSH_ORDERBY_RANK is used 2h Qh Jh 9h 4h > 2s Qs Ts 7s 6s
# if FLUSH_ORDERBY_SUIT is used 2h Qh Jh 9h 4h < 2s Qs Ts 7s 6s since spade > heart > club > diamond
# note that only one of these settings can be true at any given time!
FLUSH_ORDERBY_RANK = True
FLUSH_ORDERBY_SUIT = False
