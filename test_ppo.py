from big2_rl.evaluation.ppo_agent import *

if __name__ == '__main__':
    from big2_rl.deep_mc.env_utils import Environment
    from big2_rl.deep_mc.utils import *
    from big2_rl.deep_mc.model import Big2Model
    from big2_rl.env.env import Env
    import multiprocessing as mp
    from big2_rl.deep_mc import parser

    new_env = Environment(Env(), "cpu")
    ctx = mp.get_context('spawn')
    free = ctx.SimpleQueue()
    full = ctx.SimpleQueue()
    model = Big2Model("cpu")
    parser.add_argument('--xpid', default='big2rl',
                        help='Experiment id (default: big2rl)')
    parser.add_argument('--save_interval', default=10, type=int,
                        help='Time interval (in minutes) at which to save the model')
    parser.add_argument('--opponent_agent', default='ppo', type=str,
                        help='Type of opponent agent to be placed in other 3 positions \
                            which model will be tested again. Values = {prior, ppo, random}')

    # Training settings
    parser.add_argument('--savedir', default='ppo_test',
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
                        help='Number of shared-memory buffers for a given actor device')
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
    flags = parser.parse_args()
    flags.num_actors = 1
    buffers = create_buffers(flags, ["cpu"])
    act(0, "cpu", free, full, model, buffers, flags)
    # ppo_agent = PPOAgent()
    print("success")
