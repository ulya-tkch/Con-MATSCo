"""
Creation Date : 7th Dec 2021
Last Updated : 7th Dec 2021
Author/s : Mayank Sharan

File to construct samples for training
"""

# Base python imports
import os
import copy
import pickle
import traceback

import numpy as np
import pandas as pd


class ConstructSample:

    def __init__(self, path_to_samples, cnt_round, dic_traffic_env_conf):
        self.parent_dir = path_to_samples
        self.path_to_samples = path_to_samples + "/round_" + str(cnt_round)
        self.cnt_round = cnt_round
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.logging_data_list_per_gen = None
        self.hidden_states_list = None
        self.samples = []
        self.samples_all_intersection = [None]*self.dic_traffic_env_conf['NUM_INTERSECTIONS']

    def load_data(self, folder, i):

        try:
            f_logging_data = open(os.path.join(self.path_to_samples, folder, "inter_{0}.pkl".format(i)), "rb")
            logging_data = pickle.load(f_logging_data)
            f_logging_data.close()
            return 1, logging_data

        except Exception as _:
            print("Error occurs when making samples for inter {0}".format(i))
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            return 0, None

    def load_data_for_system(self, folder):
        """
        Load data for all intersections in one folder
        :param folder:
        :return: a list of logging data of one intersection for one folder
        """

        self.logging_data_list_per_gen = []

        # load settings

        print("Construct Sample: Load data for system in ", folder)

        self.measure_time = self.dic_traffic_env_conf["MEASURE_TIME"]
        self.interval = self.dic_traffic_env_conf["MIN_ACTION_TIME"]

        for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
            pass_code, logging_data = self.load_data(folder, i)

            if pass_code == 0:
                return 0

            self.logging_data_list_per_gen.append(logging_data)

        return 1

    def load_hidden_state_for_system(self, folder):

        print("loading hidden states: {0}".format(os.path.join(self.path_to_samples, folder, "hidden_states.pkl")))

        # load settings

        if self.hidden_states_list is None:
            self.hidden_states_list = []

        try:
            f_hidden_state_data = open(os.path.join(self.path_to_samples, folder, "hidden_states.pkl"), "rb")
            hidden_state_data = pickle.load(f_hidden_state_data) # hidden state_data is a list of numpy array

            # print(hidden_state_data)

            print(len(hidden_state_data))
            hidden_state_data_h_c = np.stack(hidden_state_data, axis=2)
            hidden_state_data_h_c = pd.Series(list(hidden_state_data_h_c))
            next_hidden_state_data_h_c = hidden_state_data_h_c.shift(-1)
            hidden_state_data_h_c_with_next = pd.concat([hidden_state_data_h_c, next_hidden_state_data_h_c], axis=1)
            hidden_state_data_h_c_with_next.columns = ['cur_hidden', 'next_hidden']
            self.hidden_states_list.append(hidden_state_data_h_c_with_next[:-1].values)
            return 1

        except Exception as e:
            print("Error occurs when loading hidden states in ", folder)
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            return 0

    def construct_state(self, features, time, i):
        """
        :param features:
        :param time:
        :param i:  intersection id
        :return:
        """

        state = self.logging_data_list_per_gen[i][time]
        assert time == state["time"]

        if self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
            state_after_selection = {}
            for key, value in state["state"].items():
                if key in features:
                    if "cur_phase" in key:
                        if value[0] == -1:
                            state_after_selection[key] = [0, 0, 0, 0, 0, 0, 0, 0]
                        else:
                            state_after_selection[key] = \
                                self.dic_traffic_env_conf['PHASE'][
                                    self.dic_traffic_env_conf['SIMULATOR_TYPE']][value[0]]
                    else:
                        state_after_selection[key] = value
        else:
            state_after_selection = {key: value for key, value in state["state"].items() if key in features}

        return state_after_selection

    def get_reward_from_features(self, rs):

        reward = dict()
        reward["sum_lane_queue_length"] = np.sum(rs["lane_queue_length"])
        reward["sum_lane_wait_time"] = np.sum(rs["lane_sum_waiting_time"])
        reward["sum_lane_num_vehicle_left"] = np.sum(rs["lane_num_vehicle_left"])
        reward["sum_duration_vehicle_left"] = np.sum(rs["lane_sum_duration_vehicle_left"])
        reward["sum_num_vehicle_been_stopped_thres01"] = np.sum(rs["lane_num_vehicle_been_stopped_thres01"])
        reward["sum_num_vehicle_been_stopped_thres1"] = np.sum(rs["lane_num_vehicle_been_stopped_thres1"])
        reward['pressure'] = np.sum(rs["pressure"])

        return reward

    def cal_reward(self, rs, rs_og, rewards_components):
        r = 0
        for component, weight in rewards_components.items():
            if weight == 0:
                continue
            if component not in rs.keys():
                continue
            if rs[component] is None:
                continue
            r += rs[component] * weight
        
        # Ulya code light fairness time
        # print("ulya state\n", rs_og["state"]["lane_wait"])
        # lane_wait = rs_og["state"]["lane_wait"]
        # averages = [np.sum(phase) / np.mean(phase) for phase in lane_wait]
        # distances = []
        # for a1 in averages:
        #     for a2 in averages:
        #         if a1 == a2:
        #             continue
        #         div1 = 0 if a2 == 0 else a1 / a2
        #         div2 = 0 if a1 == 0 else a2 / a1
        #         distances.append(0.5 * (div1 + div2) - 1)
        
        # # get MAX distance
        # max_distance = max(distances) / 50
        # # print('ulya lane waits: ', lane_wait)
        # # print("ulya distances: ", distances)
        # # print('ulya max distance: ', max_distance)

        # cur_phase = rs_og["state"]["cur_phase"][0]
        # action = rs_og["action"]
        # if cur_phase !=-1:
        #     if cur_phase == action:
        #         r+= -max_distance   ##  penalty of max distance
        return r        

        # ## rohin code for MIN_SWITCH_TIME
        # cur_phase = rs_og["state"]["cur_phase"][0]
        # time_this_phase = rs_og["state"]["time_this_phase"][0]
        # action = rs_og["action"]
        # # num_vehicles_stopped = rs_og["state"]["lane_num_vehicle_been_stopped_thres1"] # [0, 30, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
        # MIN_SWTICH_TIME = 15
        # if cur_phase !=-1:
        #     if cur_phase == action:
        #         if time_this_phase <= MIN_SWTICH_TIME:
        #             r+= -3.0   ##  penalty
        # return r

    def construct_reward(self,rewards_components,time, i):

        rs = self.logging_data_list_per_gen[i][time + self.measure_time - 1]
        assert time + self.measure_time - 1 == rs["time"]
        rs_new = self.get_reward_from_features(rs['state'])
        # print('ULYA intersection: ', i, ' timestep: ', time)
        r_instant = self.cal_reward(rs_new, rs, rewards_components)

        # average
        list_r = []
        for t in range(time, time + self.measure_time):

            rs = self.logging_data_list_per_gen[i][t]
            assert t == rs["time"]

            rs_new = self.get_reward_from_features(rs['state'])
            r = self.cal_reward(rs_new, rs, rewards_components)

            list_r.append(r)

        r_average = np.average(list_r)

        return r_instant, r_average

    def judge_action(self, time, i):

        if self.logging_data_list_per_gen[i][time]['action'] == -1:
            raise ValueError
        else:
            return self.logging_data_list_per_gen[i][time]['action']

    def make_reward(self, folder, i):
        """
        make reward for one folder and one intersection,
        add the samples of one intersection into the list.samples_all_intersection[i]
        """

        if self.samples_all_intersection[i] is None:
            self.samples_all_intersection[i] = []

        if i % 100 == 0:
            print("Construct Sample: make reward for inter {0} in folder {1}".format(i, folder))

        list_samples = []

        try:
            total_time = int(self.logging_data_list_per_gen[i][-1]['time'] + 1)

            # construct samples

            for time in range(0, total_time - self.measure_time + 1, self.interval):

                state = self.construct_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"], time, i)
                reward_instant, reward_average = self.construct_reward(self.dic_traffic_env_conf["DIC_REWARD_INFO"],
                                                                       time, i)
                action = self.judge_action(time, i)

                if time + self.interval == total_time:
                    next_state = self.construct_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"],
                                                      time + self.interval - 1, i)

                else:
                    next_state = self.construct_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"],
                                                      time + self.interval, i)

                sample = [state, action, next_state, reward_average, reward_instant, time,
                          folder + "-" + "round_{0}".format(self.cnt_round)]

                list_samples.append(sample)

            # list_samples = self.evaluate_sample(list_samples)
            self.samples_all_intersection[i].extend(list_samples)

            return 1

        except Exception as e:
            print("Error occurs when making rewards in generator {0} for intersection {1}".format(folder, i))
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            return 0

    def make_reward_for_system(self):

        """
        Iterate all the generator folders, and load all the logging data for all intersections for that folder
        At last, save all the logging data for that intersection [all the generators]
        :return:
        """

        for folder in os.listdir(self.path_to_samples):
            print("Construct Sample:", folder)
            if "generator" not in folder:
                continue

            if not self.evaluate_sample(folder) or not self.load_data_for_system(folder):
                continue

            for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
                pass_code = self.make_reward(folder, i)
                if pass_code == 0:
                    continue

        for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
            self.dump_sample(self.samples_all_intersection[i], "inter_{0}".format(i))

    def dump_hidden_states(self, folder):

        total_hidden_states = np.vstack(self.hidden_states_list)
        print("Construct Sample: dump_hidden_states shape:", total_hidden_states.shape)

        if folder == "":
            with open(os.path.join(self.parent_dir, "total_hidden_states.pkl"), "ab+") as f:
                pickle.dump(total_hidden_states, f, -1)
        elif "inter" in folder:
            with open(os.path.join(self.parent_dir, "total_hidden_states_{0}.pkl".format(folder)), "ab+") as f:
                pickle.dump(total_hidden_states, f, -1)
        else:
            with open(os.path.join(self.path_to_samples, folder, "hidden_states_{0}.pkl".format(folder)), 'wb') as f:
                pickle.dump(total_hidden_states, f, -1)

    def evaluate_sample(self, generator_folder):

        return True

    def dump_sample(self, samples, folder):

        if folder == "":
            with open(os.path.join(self.parent_dir, "total_samples.pkl"), "ab+") as f:
                pickle.dump(samples, f, -1)
        elif "inter" in folder:
            with open(os.path.join(self.parent_dir, "total_samples_{0}.pkl".format(folder)), "ab+") as f:
                pickle.dump(samples, f, -1)
        else:
            with open(os.path.join(self.path_to_samples, folder, "samples_{0}.pkl".format(folder)), 'wb') as f:
                pickle.dump(samples, f, -1)
