from big2_rl.deep_mc.settings_parser_arguments import parser
from big2_rl.evaluation.play import play_against

if __name__ == '__main__':

    # General Settings
    # by default you play as south
    parser.add_argument('--east', type=str, default='random')
    parser.add_argument('--north', type=str, default='random')
    parser.add_argument('--west', type=str, default='random')

    args = parser.parse_args()

    # TODO remove
    # baselines folder stores models for external testing/play,
    # big2rl_checkpoints stores ckpt and tar for internal testing
    args.east = 'baselines/prior-model.tar'
    args.north = 'baselines/prior-model.tar'
    args.west = 'baselines/prior-model.tar'
    #args.east = 'ppo'
    #args.north = 'ppo'
    #args.west = 'ppo'

    play_against(args)
