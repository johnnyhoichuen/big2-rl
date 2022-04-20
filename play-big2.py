from big2_rl.deep_mc.settings_parser_arguments import parser
from big2_rl.evaluation.play import play_against

if __name__ == '__main__':

    # General Settings
    # by default you play as south
    parser.add_argument('--east', type=str, default='random')
    parser.add_argument('--north', type=str, default='random')
    parser.add_argument('--west', type=str, default='random')

    args = parser.parse_args()

    """
    from big2_rl.env import move_detector as md, move_selector as ms
    from big2_rl.env.move_generator import MovesGener
    from big2_rl.deep_mc.utils import hand_to_string, string_to_hand
    hand = '3d,Qd,7d,7s,7d,7h,Qd,Qc,Qh,Ad,Ac,Ah,As'
    mg = MovesGener(string_to_hand(hand))

    # get the type of the previous move played
    rival_move = string_to_hand('2h,4c,4s,4d,4c')
    rival_type = md.get_move_type(rival_move)
    rival_move_type = rival_type['type']
    moves = list()

    if rival_move_type == md.TYPE_0_PASS:
        moves = mg.gen_moves()

    elif rival_move_type == md.TYPE_1_SINGLE:
        all_moves = mg.gen_type_1_single()
        moves = ms.filter_type_1_single(all_moves, rival_move)

    elif rival_move_type == md.TYPE_2_PAIR:
        all_moves = mg.gen_type_2_pair()
        moves = ms.filter_type_2_pair(all_moves, rival_move)

    elif rival_move_type == md.TYPE_3_TRIPLE:
        all_moves = mg.gen_type_3_triple()
        moves = ms.filter_type_3_triple(all_moves, rival_move)

    elif rival_move_type == md.TYPE_4_STRAIGHT:
        all_moves = mg.gen_type_4_straight()
        moves = ms.filter_type_4_straight(all_moves, rival_move)
        moves += mg.gen_type_5_flush() + mg.gen_type_6_fullhouse() + mg.gen_type_7_quads() + mg. \
            gen_type_8_straightflush()

    elif rival_move_type == md.TYPE_5_FLUSH:
        all_moves = mg.gen_type_5_flush()
        moves = ms.filter_type_5_flush(all_moves, rival_move)
        moves += mg.gen_type_6_fullhouse() + mg.gen_type_7_quads() + mg.gen_type_8_straightflush()

    elif rival_move_type == md.TYPE_6_FULLHOUSE:
        all_moves = mg.gen_type_6_fullhouse()
        moves = ms.filter_type_6_fullhouse(all_moves, rival_move)
        moves += mg.gen_type_7_quads() + mg.gen_type_8_straightflush()

    elif rival_move_type == md.TYPE_7_QUADS:
        all_moves = mg.gen_type_7_quads()
        moves = ms.filter_type_7_quads(all_moves, rival_move)
        moves += mg.gen_type_8_straightflush()

    elif rival_move_type == md.TYPE_8_STRAIGHTFLUSH:
        all_moves = mg.gen_type_8_straightflush()
        moves = ms.filter_type_8_straightflush(all_moves, rival_move)

    for m in moves:  # for each move, sort integer ids of each card to be in ascending order
        m.sort()

    for i in moves:
        print(hand_to_string(i))
    """

    # TODO remove
    args.east = 'big2rl_checkpoints/prior-test/prior-model.tar'
    args.north = 'big2rl_checkpoints/prior-test/prior-model.tar'
    args.west = 'big2rl_checkpoints/prior-test/prior-model.tar'
    #args.east = 'ppo'
    #args.north = 'ppo'
    #args.west = 'ppo'

    play_against(args)
