import argparse

parser = argparse.ArgumentParser(description='Big2-RL')

# General Settings
parser.add_argument('--xpid', default='big2rl',
                    help='Experiment id (default: big2rl)')
parser.add_argument('--save_interval', default=10, type=int,
                    help='Time interval (in minutes) at which to save the model')

# Game-specific settings
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

parser.add_argument('--flush_orders', default=1, type=int,
                    help='Whether flushes ranked by highest card (default) or by suit')
parser.add_argument('--straight_orders', default=1, type=int,
                    help='How straights are ranked')
parser.add_argument('--penalty_threshold', default=1, type=int,
                    help="""Thresholds at which multiplicative penalties apply for players which equal 
                    to or more than a specified number of cards""")
# TODO idk if this works
parser.add_argument('-l', '--list', nargs='*', help="""Thresholds at which multiplicative penalties apply for players which equal 
                    to or more than a specified number of cards""", type=int, required=False)
# TODO move this to wherever it is parsed:
#args = parser.parse_args()
#valid input = exactly 3 arguments and all 3 are positive integers
#if integers not in order put them in ascending order
#if invalid input or not specified: use [8, 10, 13]
#allow something like [10, 12, 18] that means 10-11 2x, 12-13 3x, no circumstance 4x
#my_list = [int(item) for item in args.list.split(',')]


# Training settings
parser.add_argument('--actor_device_cpu', action='store_true',
                    help='Use CPU as actor device')
parser.add_argument('--gpu_devices', default='0', type=str,
                    help='Which GPUs to be used for training')
parser.add_argument('--num_actor_devices', default=1, type=int,
                    help='The number of devices used for simulation')
parser.add_argument('--num_actors', default=5, type=int,
                    help='The number of actors for each simulation device')
parser.add_argument('--training_device', default='0', type=str,
                    help='The index of the GPU used for training models. `cpu` means using cpu')
parser.add_argument('--load_model', action='store_true',
                    help='Load an existing model')
parser.add_argument('--disable_checkpoint', action='store_true',
                    help='Disable saving checkpoint')
parser.add_argument('--savedir', default='big2rl_checkpoints',
                    help='Root dir where experiment data will be saved')

# Hyperparameters
parser.add_argument('--total_frames', default=100000000000, type=int,
                    help='Total environment frames to train for')
parser.add_argument('--exp_epsilon', default=0.01, type=float,
                    help='The probability for exploration')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Learner batch size')
parser.add_argument('--unroll_length', default=100, type=int,
                    help='The unroll length (time dimension)')
parser.add_argument('--num_buffers', default=50, type=int,
                    help='Number of shared-memory buffers')
parser.add_argument('--num_threads', default=4, type=int,
                    help='Number learner threads')
parser.add_argument('--max_grad_norm', default=40., type=float,
                    help='Max norm of gradients')

# Optimizer settings
parser.add_argument('--learning_rate', default=0.0001, type=float,
                    help='Learning rate')
parser.add_argument('--alpha', default=0.99, type=float,
                    help='RMSProp smoothing constant')
parser.add_argument('--momentum', default=0, type=float,
                    help='RMSProp momentum')
parser.add_argument('--epsilon', default=1e-5, type=float,
                    help='RMSProp epsilon')
