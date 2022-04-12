import argparse

# Creates parser for game-specific settings. Required since training and evaluation have different sets of
# cmdline arguments, but they share this subset of arguments.
parser = argparse.ArgumentParser(description='Big2-RL')

parser.add_argument('--penalise_quads', default=1, type=int,
                    help='Multiplicative penalty for a player if they have 1 or more unplayed quads')
parser.add_argument('--penalise_sf', default=1, type=int,
                    help='Multiplicative penalty for a player if they have 1 or more unplayed straight flushes')
parser.add_argument('--penalise_deuces', default=2, type=int,
                    help='Multiplicative penalty for a player if they have 1 or more unplayed deuces')
parser.add_argument('--reward_quads', default=1, type=int,
                    help='Multiplicative reward for a player if they win and their last move was quads')
parser.add_argument('--reward_sf', default=1, type=int,
                    help='Multiplicative reward for a player if they win and their last move was straight flush')
parser.add_argument('--reward_deuces', default=1, type=int,
                    help='Multiplicative reward for a player if they win and their last move was a deuce')

parser.add_argument('--flush_orders', default='rank', type=str,
                    help='Whether flushes ranked by highest card (default) or by suit. Possible values: rank, suit')
parser.add_argument('--straight_orders', default='T2A', type=str,
                    help='How straights are ranked. Value refers to the lowest card in the three highest-ranked \
                    straights. Example: TA2 -> [T]JQKA < [A]2345 < [2]3456. Possible values: T2A, TA2, 9TA, 9AT, 89T')

# there can be 0 or more arguments of this
# Use like:
# python settings_parser_arguments.py -pt 8 11 13
parser.add_argument('-pt', '--penalty_threshold', nargs='*',
                    help="""Thresholds at which multiplicative penalties apply for players which equal 
                    to or more than the specified number of cards""", type=int, required=False)
