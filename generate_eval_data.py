import argparse
import numpy as np
import pickle
from big2_rl.env.game import Position

deck = [i for i in range(0, 52)]


def get_parser():
    """
    Create parser for generating evaluation data
    """
    parser = argparse.ArgumentParser(description='Big2RL - generate random games for evaluation')
    parser.add_argument('--output', default='eval_data', type=str)
    parser.add_argument('--num_games', default=20, type=int)  # TODO increase this
    return parser


def generate_game():
    """
    Generates a random deal (each player's hand is sorted)
    """
    _deck = deck.copy()
    np.random.shuffle(_deck)
    card_play_data = {p.name: 0 for p in Position}
    for p in Position:
        card_play_data[p.name] = _deck[p.value*13: (p.value+1)*13]
    for pos in card_play_data:
        card_play_data[pos].sort()
    return card_play_data


if __name__ == '__main__':
    flags = get_parser().parse_args()
    output_pickle = flags.output + '.pkl'
    print("output_pickle: {}", output_pickle)
    print("Generating games...")

    data = []
    for _ in range(flags.num_games):
        data.append(generate_game())
    print("Saving pickled output.")
    with open(output_pickle, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
