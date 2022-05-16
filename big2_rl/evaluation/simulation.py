import multiprocessing as mp
import pickle

from big2_rl.env.game import GameEnv, Position

from big2_rl.evaluation.random_agent import RandomAgent
from big2_rl.evaluation.dmc_agent import DMCAgent
from big2_rl.evaluation.ppo_agent import PPOAgent


def load_models(model_path, model_type):
    """
    Loads a specified agent type (random, PPO or deep MC)
    """
    # model_path = {'SOUTH': south, 'EAST': east, 'NORTH': north, 'WEST': west}

    players = {}
    for p in Position:
        if model_path[p.name] == 'random':
            players[p.name] = RandomAgent()
        elif model_path[p.name] == 'ppo':
            players[p.name] = PPOAgent()
        else:
            # 'prior' will be a path instead of name
            players[p.name] = DMCAgent(model_path[p.name], model_type)
    return players


def mp_simulate(card_play_data_list, card_play_model_path_dict, queue, model_type):
    """
    Handles GameEnv definition, initialisation and reset for processing evaluation data for a given worker
    """
    players = load_models(card_play_model_path_dict, model_type)
    env = GameEnv(players)
    # play each game by creating a GameEnv
    for idx, card_play_data in enumerate(card_play_data_list):
        env.card_play_init(card_play_data)
        # added by HCJ to set starting_hands attribute of each PPOAgent
        for p in Position:
            hand = env.info_sets[p.name].player_hand_cards
            if players[p.name].__class__.__name__ == "PPOAgent":
                players[p.name].set_starting_hand(hand)
        # end
        while not env.game_over:
            env.step()
        env.reset()
    queue.put((env.num_wins, env.num_scores))


def evaluate(south, east, north, west, eval_data, num_workers, model_type):
    """
    Uses multiprocessing on 'num_workers' many workers to evaluate a given set of evaluation data
    (pickled deals) by metrics like EV, win %, etc.
    """
    # load card play evaluation data
    with open(eval_data, 'rb') as f:
        card_play_data_list = pickle.load(f)

    # separate evaluation data into multiple workers
    card_play_data_list_per_worker = [[] for _ in range(num_workers)]
    total_games = len(card_play_data_list)
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
        process = ctx.Process(target=mp_simulate, args=(card_play_data, card_play_model_path_dict, q, model_type))
        process.start()
        eval_processes.append(process)
    for process in eval_processes:
        process.join()

    for i in range(num_workers):
        result = q.get()
        for p in Position:
            num_wins_by_position[p.name] += result[0][p.name]
            ev_by_position[p.name] += result[1][p.name]

    print('total games (play at all 4 positions in the same deck): ', total_games)

    wp = {}
    agg_ev = {}
    avg_ev = {}
    for p in Position:
        # print("{}".format(p.name))
        wp[p.name] = num_wins_by_position[p.name] / total_games
        agg_ev = ev_by_position[p.name]
        avg_ev[p.name] = ev_by_position[p.name] / total_games
        # print("Games won percentage: {}".format(wp[p.name]))
        # print("Aggregate EV: {}, average EV per game: {}".format(ev_by_position[p.name],
        #                                                          avg_ev))

    print(f'WP: {wp}')
    print(f'Aggregate EV: {agg_ev}')
    print(f'Average EV: {avg_ev}')
