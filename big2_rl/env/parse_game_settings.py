from big2_rl.env.settings import Settings


def parse_game_settings(args):
    """
    Create GameSettings instance (if not defined) and updates singleton instance's attribute values
    (penalties, rewards, flush & straight orders)
    """
    gs = Settings
    gs.set_penalties_and_rewards(
        args.penalise_quads, args.penalise_sf, args.penalise_deuces,
        args.reward_quads, args.reward_sf, args.reward_deuces)
    gs.set_flush_order(args.flush_orders)
    gs.set_straight_order(args.straight_orders)

    pt_list = args.penalty_threshold
    # in order for this argument to be valid, must:
    # 1. have exactly 3 arguments, where all 3 are positive integers
    # 2. have no repeated values
    # 3. be ordered (in ascending order)
    # We allow something like [10, 12, 18] that means 10-11 2x, 12-13 3x, no circumstance 4x
    if len(pt_list) == 3 and min(pt_list) > 0 and len(set(pt_list)) == 3:
        pt_list = sorted(pt_list)  # in ascending order
    else:
        pt_list = [8, 10, 13]  # else use default value
    gs.set_penalty_threshold(pt_list)
    return gs
