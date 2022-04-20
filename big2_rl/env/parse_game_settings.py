import settings


def parse_settings(args):
    # Create GameSettings instance (if not defined) and updates singleton instance's attribute values
    # (penalties, rewards, flush & straight orders)

    straight_order = args.straight_orders
    if straight_order == "T2A":  # default
        straight_orders = [[_ for _ in range(0, 5)], [_ for _ in range(1, 6)], [_ for _ in range(2, 7)],
                           [_ for _ in range(3, 8)], [_ for _ in range(4, 9)],
                           [_ for _ in range(5, 10)], [_ for _ in range(6, 11)],
                           [_ for _ in range(7, 12)], [12, 0, 1, 2, 3], [11, 12, 0, 1, 2]]
    elif straight_order == "TA2":
        straight_orders = [[_ for _ in range(0, 5)], [_ for _ in range(1, 6)], [_ for _ in range(2, 7)],
                           [_ for _ in range(3, 8)], [_ for _ in range(4, 9)],
                           [_ for _ in range(5, 10)], [_ for _ in range(6, 11)],
                           [_ for _ in range(7, 12)], [11, 12, 0, 1, 2], [12, 0, 1, 2, 3]]
    elif straight_order == "9TA":
        straight_orders = [[12, 0, 1, 2, 3], [_ for _ in range(0, 5)], [_ for _ in range(1, 6)],
                           [_ for _ in range(2, 7)], [_ for _ in range(3, 8)], [_ for _ in range(4, 9)],
                           [_ for _ in range(5, 10)], [_ for _ in range(6, 11)], [_ for _ in range(7, 12)],
                           [11, 12, 0, 1, 2]]
    elif straight_order == "9AT":
        straight_orders = [[12, 0, 1, 2, 3], [_ for _ in range(0, 5)], [_ for _ in range(1, 6)],
                           [_ for _ in range(2, 7)], [_ for _ in range(3, 8)], [_ for _ in range(4, 9)],
                           [_ for _ in range(5, 10)], [_ for _ in range(6, 11)], [11, 12, 0, 1, 2],
                           [_ for _ in range(7, 12)]]
    elif straight_order == "89T":
        straight_orders = [[11, 12, 0, 1, 2], [12, 0, 1, 2, 3], [_ for _ in range(0, 5)],
                           [_ for _ in range(1, 6)], [_ for _ in range(2, 7)], [_ for _ in range(3, 8)],
                           [_ for _ in range(4, 9)], [_ for _ in range(5, 10)], [_ for _ in range(6, 11)],
                           [_ for _ in range(7, 12)]]
    else:
        raise Exception("invalid parameter for set_straight_order")

    if args.flush_orders == "suit":
        f_o = 1
    elif args.flush_orders == "rank":
        f_o = 0
    else:
        raise Exception("invalid flush order")

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

    settings.flush_orders = f_o
    settings.straight_orders = straight_orders
    settings.penalty_threshold = pt_list

    # in some variations if losing players still have deuces, 1 or more SF, or 1 or more quads, then
    # the winner's reward is doubled
    settings.penalise_quads = args.penalise_quads
    settings.penalise_sf = args.penalise_sf
    settings.penalise_deuces = args.penalise_deuces

    # in some variations if winning player ends with a deuce, SF or quads their reward is doubled
    settings.reward_quads = args.reward_quads
    settings.reward_sf = args.reward_sf
    settings.reward_deuces = args.reward_deuces
