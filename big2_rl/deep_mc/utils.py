import typing
import logging
import traceback
from big2_rl.env.game import Position
from big2_rl.evaluation.random_agent import RandomAgent
from big2_rl.evaluation.ppo_agent import PPOAgent
from big2_rl.deep_mc.model import Big2Model
from copy import deepcopy

import torch
import os

from big2_rl.deep_mc.env_utils import Environment
from big2_rl.env.env import Env
from big2_rl.env.env import _cards2array

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('big2-rl')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

# Buffers are used to transfer data between actor processes
# and learner threads on the same actor device. They are shared tensors on that device
Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def hand_to_string(x):
    hand = []
    ranks = ['3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', '2']
    suits = ['d', 'c', 'h', 's']
    for card in x:
        hand.append(ranks[card // 4] + suits[card % 4])
    hand_str = ','.join(hand)
    return hand_str


def string_to_hand(x):
    # example: '3d,3h,8h,9h,Th,Jh,Qh,Kc,Ks,As,Ad,2c'
    ranks = ['3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', '2']
    suits = ['d', 'c', 'h', 's']
    x_list = x.split(",")
    hand = []
    for i in x_list:
        rank_val = ranks.index(i[0])
        suit_val = suits.index(i[1])
        hand.append(rank_val*4 + suit_val)
    return sorted(hand)


def get_batch(free_queue,
              full_queue,
              buffers,
              flags,
              lock):
    """
    This function will sample a batch from the buffers of a given actor device based
    on the indices received from the full queue. It will also free the indices by sending it to free_queue.
    """
    with lock:
        indices = [full_queue.get() for _ in range(flags.batch_size)]
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    for m in indices:
        free_queue.put(m)
    return batch


def create_buffers(flags, device_iterator):
    """
    We create buffers for each actor device. Each actor device shall have 'flags.num_buffers' many buffers
    and these buffers will be shared across each of the four positions.
    """
    T = flags.unroll_length
    # according to TorchBeast IMPALA architecture: https://arxiv.org/pdf/1910.03552.pdf
    # each actor produces rollouts in an indefinite loop.
    # A rollout consists of unroll_length many environment-agent interactions
    # learner consumes batches of these rollouts. So the learner input takes the form:
    buffers = {}
    for device in device_iterator:
        # dim = size of state features not including action or historical moves (see env/env.py)
        x_dim = 507  # regardless of position
        specs = dict(
            done=dict(size=(T,), dtype=torch.bool),
            episode_return=dict(size=(T,), dtype=torch.float32),
            target=dict(size=(T,), dtype=torch.float32),
            obs_x_no_action=dict(size=(T, x_dim), dtype=torch.int8),
            obs_action=dict(size=(T, 52), dtype=torch.int8),
            obs_z=dict(size=(T, 4, 208), dtype=torch.int8),
        )
        _buffers: Buffers = {key: [] for key in specs}
        for _ in range(flags.num_buffers):
            for key in _buffers:
                # https://stackoverflow.com/questions/21809112/what-does-tuple-and-dict-mean-in-python
                # **(dict) allows us to define empty torch tensors of their corresponding sizes.
                # Example: _buffers['obs_action'] has value: size (T,52) empty torch tensor on the correct device
                if not device == "cpu":
                    _buffer = torch.empty(**specs[key]).to(torch.device('cuda:'+str(device))).share_memory_()
                else:
                    _buffer = torch.empty(**specs[key]).to(torch.device('cpu')).share_memory_()
                _buffers[key].append(_buffer)
        buffers[device] = _buffers
    # each buffers[device] should be {'done':[tensor0, tensor1, ..., tensor_{num_buffers-1}],
    # 'episode_return':[tensor0, tensor1, ..., tensor_{num_buffers-1}], ...}
    return buffers


def act(i, device, free_queue, full_queue, model, buffers, flags):
    """
    This function is run by each single (of potentially multiple) actors on a given actor device.
    It will run forever until we stop it, and generates
    data from the environment and sends the data to the shared buffer of that actor device. It uses
    a free queue and full queue to sync with the main process.
    """
    try:
        T = flags.unroll_length
        log.info('Device %s Actor %i started.', str(device), i)

        env = Environment(Env(), device)

        # initialise buffers to store values for each position
        done_buf = {p.name: [] for p in Position}
        episode_return_buf = {p.name: [] for p in Position}
        target_buf = {p.name: [] for p in Position}
        obs_x_no_action_buf = {p.name: [] for p in Position}
        obs_action_buf = {p.name: [] for p in Position}
        obs_z_buf = {p.name: [] for p in Position}
        size = {p.name: 0 for p in Position}

        # this will reset the Env class, since Environment.initial() is called
        # 'position' is a string (enum name) corresponding to current player position
        # 'obs' is a dict of information available to that position
        # 'env_output' is a dict of info such as episodic reward, whether current deal is over,
        # historical actions (obs_z), and game state (obs_x_no_action)
        position, obs, env_output = env.initial()

        # since positions are symmetric, we only consider this position for learner
        observed_player = Position.SOUTH.name

        # stays constant at all other positions. Can import other agents for evaluating our DMC agent against
        random_agent = RandomAgent()
        ppo_agent = PPOAgent()

        prior_model = Big2Model(device)
        psd = prior_model.state_dict()
        msd = deepcopy(model.state_dict())
        psd.update(msd)
        prior_model.load_state_dict(psd)
        prior_model.eval()

        games = 0
        model_path = os.path.expandvars(os.path.expanduser('%s/%s/%s' % (
            flags.savedir, flags.xpid, 'model.tar')))

        # outer loop plays infinite deals (is never broken), inner while loop corresponds to one game
        while True:
            # load starting hands for each player
            starting_hands = {}
            start_pos = Position[position].value
            for idx in range(start_pos, start_pos+4):
                pos_name = Position(idx % 4).name
                hand = env.env._env.info_sets[pos_name].player_hand_cards
                starting_hands[pos_name] = hand
            while True:
                # save current turn's possible state (obs_x_no_action) and historical moves (obs_z)
                obs_x_no_action_buf[position].append(env_output['obs_x_no_action'])
                obs_z_buf[position].append(env_output['obs_z'])
                if position == observed_player:
                    # use the model to predict next action to make
                    with torch.no_grad():
                        agent_output = model.forward(obs['z_batch'], obs['x_batch'], flags=flags)
                    # given the action (a torch tensor), get the corresponding card values eg [51, 50] for playing 2s2h
                    _action_idx = int(agent_output['action'].cpu().detach().numpy())
                    action = obs['legal_actions'][_action_idx]
                else:
                    if flags.opponent_agent == 'ppo':
                        ppo_agent.set_starting_hand(starting_hands[position])
                        action = ppo_agent.act(env.env.infoset, env.env._env.card_play_action_seq)
                    elif flags.opponent_agent == 'prior':
                        with torch.no_grad():
                            agent_output = prior_model.forward(obs['z_batch'], obs['x_batch'], flags=flags)
                        _action_idx = int(agent_output['action'].cpu().detach().numpy())
                        action = obs['legal_actions'][_action_idx]
                        # update weights of actor models to most recent parameters only intermittently
                        if games % 1000 == 0 and os.path.exists(model_path):
                            if torch.cuda.is_available():
                                pretrained = torch.load(model_path, map_location='cuda:0')
                            else:
                                pretrained = torch.load(model_path, map_location='cpu')
                            prior_model.load_state_dict(pretrained["model_state_dict"])
                            prior_model.eval()
                    else:  # random agent
                        action = random_agent.act(env.env.infoset)
                # save current turn's action (as 1 hot torch tensor) to the corresponding buffer
                obs_action_buf[position].append(torch.from_numpy(_cards2array(action)))
                # number of moves made by that position
                size[position] += 1
                # advance Env object by one step:
                # asserts the action is legal, and updates position of next player and
                # obs of current position, and gets env_output (x_no_action and z) of the next player's turn
                position, obs, env_output = env.step(action)
                if env_output['done']:  # repeat unless deal isn't done. Else compute episode return for each position
                    for p in Position:  # extend 'episode_return', 'done' and 'target' buffers
                        diff = size[p.name] - len(target_buf[p.name])
                        if diff > 0:
                            done_buf[p.name].extend([False for _ in range(diff-1)])
                            done_buf[p.name].append(True)

                            # this should only be defined when env_output['done']==True (else it is torch tensor of 0's)
                            episode_return = env_output['episode_return'][p.name]
                            episode_return_buf[p.name].extend([0.0 for _ in range(diff-1)])
                            episode_return_buf[p.name].append(episode_return)
                            target_buf[p.name].extend([episode_return for _ in range(diff)])
                    break

            games += 1  # for prior

            # Upon completion of an episode/deal, data D is generated (above).
            # for every T (unroll length) moves made by 'observed_player', an actor A on device V pops an index I
            # from the free queue of V. It saves D to the shared buffer (on that actor device) at index I in
            # increments of T (unroll length), and pushes the index I to the full queue on that actor device.
            # Then, the first T elements of each buffer attribute are removed.

            # Basically, after episode reward calculated, update buffer values of that actor device
            # ACCORDING TO observed_player (this is different from DouZero, where every position is important
            # since the game is asymmetric!)
            while size[observed_player] > T:
                index = free_queue.get()
                if index is None:
                    break
                for t in range(T):
                    buffers['done'][index][t, ...] = done_buf[observed_player][t]
                    buffers['episode_return'][index][t, ...] = episode_return_buf[observed_player][t]
                    buffers['target'][index][t, ...] = target_buf[observed_player][t]
                    buffers['obs_x_no_action'][index][t, ...] = obs_x_no_action_buf[observed_player][t]
                    buffers['obs_action'][index][t, ...] = obs_action_buf[observed_player][t]
                    buffers['obs_z'][index][t, ...] = obs_z_buf[observed_player][t]
                full_queue.put(index)

                for p in Position:
                    done_buf[p.name] = done_buf[p.name][T:]
                    episode_return_buf[p.name] = episode_return_buf[p.name][T:]
                    target_buf[p.name] = target_buf[p.name][T:]
                    obs_x_no_action_buf[p.name] = obs_x_no_action_buf[p.name][T:]
                    obs_action_buf[p.name] = obs_action_buf[p.name][T:]
                    obs_z_buf[p.name] = obs_z_buf[p.name][T:]
                    size[p.name] -= T

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e
