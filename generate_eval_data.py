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
    parser.add_argument('--num_games', default=500, type=int)  # TODO increase this
    return parser


def generate_game():
    """
    Generates a random deal (each player's hand is sorted).
    Then generates 4 games (so each agent has a chance to play each position)
    """
    _deck = deck.copy()
    np.random.shuffle(_deck)
    card_play_datas = []
    for p_offset in range(0, 4):
        card_play_data = {p.name: 0 for p in Position}
        for p in Position:
            pval = (p.value + p_offset) % 4
            card_play_data[p.name] = _deck[pval*13: (pval+1)*13]
        for pos in card_play_data:
            card_play_data[pos].sort()
        card_play_datas.append(card_play_data)
    return card_play_datas


if __name__ == '__main__':
    flags = get_parser().parse_args()
    output_pickle = flags.output + '.pkl'
    print("output_pickle: {}" .format(output_pickle))
    print("Generating games...")

    data = []  # list of card_play_data aka list of dict{String: list of int}
    for _ in range(flags.num_games):
        data.extend(generate_game())
    print("Saving pickled output.")
    with open(output_pickle, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
