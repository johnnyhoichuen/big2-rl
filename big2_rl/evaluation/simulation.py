import multiprocessing as mp
import pickle

from big2_rl.env.game import GameEnv, Position

from .random_agent import RandomAgent
from .dmc_agent import DMCAgent


def load_models(model_path):
    """
    Loads a specified agent type (random or deep MC)
    """
    players = {}
    for p in Position:
        if model_path[p.name] == 'random':
            players[p.name] = RandomAgent()
        else:
            players[p.name] = DMCAgent(model_path)
    return players


def mp_simulate(card_play_data_list, card_play_model_path_dict, queue):
    """
    Handles GameEnv definition, initialisation and reset for processing evaluation data for a given worker
    """
    players = load_models(card_play_model_path_dict)
    env = GameEnv(players)
    # play each game by creating a GameEnv
    for idx, card_play_data in enumerate(card_play_data_list):
        env.card_play_init(card_play_data)
        while not env.game_over:
            env.step()
        env.reset()
    queue.put((env.num_wins, env.num_scores))


def evaluate(south, east, north, west, eval_data, num_workers):
    """
    Uses multiprocessing on 'num_workers' many workers to evaluate a given set of evaluation data
    (pickled deals) by metrics like EV, win %, etc.
    """
    # load card play evaluation data
    with open(eval_data, 'rb') as f:
        card_play_data_list = pickle.load(f)

    # separate evaluation data into multiple workers
    card_play_data_list_per_worker = [[] for _ in range(num_workers)]
    for idx, data in enumerate(card_play_data_list):
        card_play_data_list_per_worker[idx % num_workers].append(data)
    del card_play_data_list

    card_play_model_path_dict = {'SOUTH': south, 'EAST': east, 'NORTH': north, 'WEST': west}

    # define metrics we want to compute over evaluation data
    num_wins_by_position = {p.name: 0 for p in Position}  # number of wins over evaluation data
    ev_by_position = {p.name: 0 for p in Position}  # EV (money won) over evaluation data

    # use multiprocessing for evaluation
    ctx = mp.get_context('spawn')
    q = ctx.SimpleQueue()
    eval_processes = []
    for card_play_data in card_play_data_list_per_worker:
        process = ctx.Process(target=mp_simulate, args=(card_play_data, card_play_model_path_dict, q))
        process.start()
        eval_processes.append(process)
    for process in eval_processes:
        process.join()

    total_wins = 0
    for i in range(num_workers):
        result = q.get()
        for p in Position:
            num_wins_by_position[p.name] += result[p.name][0]
            ev_by_position[p.name] += result[p.name][1]
            total_wins += num_wins_by_position[p.name]

    for p in Position:
        print("Games won percentage: {}" .format(num_wins_by_position[p.name]/total_wins))
        print("Aggregate EV: {}".format(ev_by_position[p.name]))
