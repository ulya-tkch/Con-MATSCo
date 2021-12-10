"""
Creation Date : 30th Nov 2021
Last Updated : 30th Nov 2021
Author/s : Mayank Sharan

Contains functions to compute the constraint metrics given a list of states for all intersections in the
"""

# Base python imports

import os
import pickle
import numpy as np

from datetime import datetime

# Define global variables

constraint_const_dict = {
    "min_green_time": 15,
}


def compute_min_switching_metric(base_dir, num_inter):

    """
    Compute cost for minimum signal switching time constraint violation.

    Parameters:
        base_dir            (str)               : Path to the directory containing pickle file for each intersection
        num_inter           (int)               : Number of intersections in the network being evaluated

    Returns:
        None
    """

    t_start = datetime.now()

    print("Initiating min switching violation metric computation for", base_dir)

    int_sig_switch_dict = {}

    # Loop over every intersection

    for inter_idx in range(num_inter):

        # Load the pickle file for the intersection

        f = open(os.path.join(base_dir, "inter_{0}.pkl".format(inter_idx)), "rb")
        try:
            states_arr = pickle.load(f)
        except Exception:
            return

        num_ts = len(states_arr)

        # Initialize variables to compute signal switch array

        past_phase = None
        time_this_phase = None

        curr_sig_switch_time = []
        curr_from_phase = []
        curr_to_phase = []

        for idx in range(num_ts):

            curr_state = states_arr[idx]['state']
            curr_phase = curr_state['cur_phase'][0]

            # If there is a change in phase then store the information in arrays

            if past_phase is not None and not(past_phase == curr_phase):
                curr_from_phase.append(past_phase)
                curr_to_phase.append(curr_phase)
                curr_sig_switch_time.append(time_this_phase)

            past_phase = curr_phase
            time_this_phase = curr_state['time_this_phase'][0]

        # Populate the signal switch information array in the dictionary

        sig_switch_arr = np.c_[curr_from_phase, curr_to_phase, curr_sig_switch_time]
        int_sig_switch_dict[inter_idx] = sig_switch_arr

        f.close()

    # Compute the metric using the data we have computed

    violation_comp_arr = np.zeros(shape=(num_inter, 3))

    for inter_idx in range(num_inter):

        # TODO : Add back intersection 0 in the computation once the issue with pickle is figured out

        if inter_idx == 0:
            continue

        # Identify violating switches

        sig_switch_arr = int_sig_switch_dict.get(inter_idx)
        num_switches = sig_switch_arr.shape[0]

        violating_switch_bool = sig_switch_arr[:, 2] < constraint_const_dict.get('min_green_time')
        exempt_switch_bool = sig_switch_arr[:, 0] == -1

        violating_switch_bool = np.logical_and(violating_switch_bool, np.logical_not(exempt_switch_bool))

        violation_comp_arr[inter_idx, 2] = num_switches - np.sum(exempt_switch_bool)
        num_violations = np.sum(violating_switch_bool)

        # If there are violations add to the cost

        if num_violations > 0:
            avg_squared_violation = np.mean(np.square(constraint_const_dict.get('min_green_time') -
                                                      sig_switch_arr[violating_switch_bool, 2]))

            violation_comp_arr[inter_idx, 0] = avg_squared_violation
            violation_comp_arr[inter_idx, 1] = num_violations

    # Compute the final cost

    total_num_violations = np.sum(violation_comp_arr[:, 1])
    min_switch_cost = 0

    if total_num_violations > 0:
        min_switch_cost_1 = np.sum(np.multiply(violation_comp_arr[:, 0],
                                               violation_comp_arr[:, 1])) / total_num_violations
        min_switch_cost_2 = (constraint_const_dict.get('min_green_time') ** 2) * total_num_violations / \
            np.sum(violation_comp_arr[:, 2])

        min_switch_cost = (min_switch_cost_1 + min_switch_cost_2) / \
                          (2 * (constraint_const_dict.get('min_green_time') ** 2))

    print("Min switching violation metric computation completed in time", datetime.now() - t_start)
    print("Min switching violation cost is", np.round(min_switch_cost, 3))


if __name__ == '__main__':

    # Code for testing

    # compute_min_switching_metric('records/1130_20_Fixedtime_6_6_bi_test_rewardshaping/' +
    #                              'anon_6_6_300_0.3_bi.json_11_30_02_45_19_30/', 36)

    compute_min_switching_metric('records/1130_22_CoLight_6_6_bi_test_rewardshaping/' +
                                 'anon_6_6_300_0.3_bi.json_11_30_02_01_18/test_round/round_1', 36)


