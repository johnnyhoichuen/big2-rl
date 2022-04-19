from big2_rl.deep_mc.settings_parser_arguments import parser
import big2_rl.env.parse_game_settings
from big2_rl.evaluation.play import play_against

if __name__ == '__main__':

    # General Settings
    # by default you play as south
    parser.add_argument('--east', type=str, default='random')
    parser.add_argument('--north', type=str, default='random')
    parser.add_argument('--west', type=str, default='random')

    args = parser.parse_args()

    # (re-)initialise game settings
    big2_rl.env.parse_game_settings.parse_settings(args)

    # TODO remove
    args.east = 'big2rl_checkpoints/prior-test/prior-model.tar'
    args.north = 'big2rl_checkpoints/prior-test/prior-model.tar'
    args.west = 'big2rl_checkpoints/prior-test/prior-model.tar'

    play_against(args)
