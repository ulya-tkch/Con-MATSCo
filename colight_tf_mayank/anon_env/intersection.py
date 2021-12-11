"""
Creation Date : 5th Dec 2021
Last Updated : 5th Dec 2021
Author/s : Mayank Sharan

Class to manage interactions with Cityflow simulator for Intersections
"""

# Base python imports

import os
import sys
import numpy as np
import pandas as pd

from copy import deepcopy


class Intersection:

    DIC_PHASE_MAP = {
        0: 1,
        1: 2,
        2: 3,
        3: 4,
        -1: 0
    }

    def __init__(self, inter_id, dic_traffic_env_conf, eng, light_id_dict, path_to_log):
        self.inter_id = inter_id

        self.inter_name = "intersection_{0}_{1}".format(inter_id[0], inter_id[1])

        self.eng = eng

        self.fast_compute = dic_traffic_env_conf['FAST_COMPUTE']

        self.controlled_model = dic_traffic_env_conf['MODEL_NAME']
        self.path_to_log = path_to_log

        # =====  intersection settings =====
        self.list_approachs = ["W", "E", "N", "S"]
        self.dic_approach_to_node = {"W": 2, "E": 0, "S": 3, "N": 1}

        self.dic_entering_approach_to_edge = {"W": "road_{0}_{1}_0".format(inter_id[0] - 1, inter_id[1])}
        self.dic_entering_approach_to_edge.update({"E": "road_{0}_{1}_2".format(inter_id[0] + 1, inter_id[1])})
        self.dic_entering_approach_to_edge.update({"S": "road_{0}_{1}_1".format(inter_id[0], inter_id[1] - 1)})
        self.dic_entering_approach_to_edge.update({"N": "road_{0}_{1}_3".format(inter_id[0], inter_id[1] + 1)})

        self.dic_exiting_approach_to_edge = {
            approach: "road_{0}_{1}_{2}".format(inter_id[0], inter_id[1], self.dic_approach_to_node[approach]) for
            approach in self.list_approachs}
        self.dic_entering_approach_lanes = {"W": [0], "E": [0], "S": [0], "N": [0]}
        self.dic_exiting_approach_lanes = {"W": [0], "E": [0], "S": [0], "N": [0]}

        # grid settings
        self.length_lane = 300
        self.length_terminal = 50
        self.length_grid = 5
        self.num_grid = int(self.length_lane // self.length_grid)

        self.list_phases = dic_traffic_env_conf["PHASE"][dic_traffic_env_conf['SIMULATOR_TYPE']]

        # generate all lanes

        self.list_entering_lanes = []
        for approach in self.list_approachs:
            self.list_entering_lanes += [self.dic_entering_approach_to_edge[approach] + '_' + str(i) for i in
                                         range(sum(list(dic_traffic_env_conf["LANE_NUM"].values())))]
        self.list_exiting_lanes = []
        for approach in self.list_approachs:
            self.list_exiting_lanes += [self.dic_exiting_approach_to_edge[approach] + '_' + str(i) for i in
                                        range(sum(list(dic_traffic_env_conf["LANE_NUM"].values())))]

        self.list_lanes = self.list_entering_lanes + self.list_exiting_lanes

        self.adjacency_row = light_id_dict['adjacency_row']
        self.neighbor_ENWS = light_id_dict['neighbor_ENWS']
        self.neighbor_lanes_ENWS = light_id_dict['entering_lane_ENWS']

        def _get_top_k_lane(lane_id_list, top_k_input):
            top_k_lane_indexes = []
            for i in range(top_k_input):
                lane_id_temp = lane_id_list[i] if i < len(lane_id_list) else None
                top_k_lane_indexes.append(lane_id_temp)
            return top_k_lane_indexes

        self._adjacency_row_lanes = {}

        # _adjacency_row_lanes is the lane id, not index

        for lane_id in self.list_entering_lanes:
            if lane_id in light_id_dict['adjacency_matrix_lane']:
                self._adjacency_row_lanes[lane_id] = light_id_dict['adjacency_matrix_lane'][lane_id]
            else:
                self._adjacency_row_lanes[lane_id] = [
                    _get_top_k_lane([], self.dic_traffic_env_conf["TOP_K_ADJACENCY_LANE"]),
                    _get_top_k_lane([], self.dic_traffic_env_conf["TOP_K_ADJACENCY_LANE"])]

        # order is the entering lane order, each element is list of two lists

        self.adjacency_row_lane_id_local = {}
        for index, lane_id in enumerate(self.list_entering_lanes):
            self.adjacency_row_lane_id_local[lane_id] = index

        # previous & current
        self.dic_lane_vehicle_previous_step = {}
        self.dic_lane_waiting_vehicle_count_previous_step = {}

        self.dic_vehicle_speed_previous_step = {}
        self.dic_vehicle_distance_previous_step = {}

        self.dic_lane_vehicle_current_step = {}
        self.dic_lane_waiting_vehicle_count_current_step = {}
        self.dic_vehicle_speed_current_step = {}
        self.dic_vehicle_distance_current_step = {}

        self.list_lane_vehicle_previous_step = []
        self.list_lane_vehicle_current_step = []

        # -1: all yellow, -2: all red, -3: none
        self.all_yellow_phase_index = -1
        self.all_red_phase_index = -2

        self.current_phase_index = 1
        self.previous_phase_index = 1
        self.eng.set_tl_phase(self.inter_name, self.current_phase_index)
        path_to_log_file = os.path.join(self.path_to_log, "signal_inter_{0}.txt".format(self.inter_name))

        df = [self.get_current_time(), self.current_phase_index]
        df = pd.DataFrame(df)
        df = df.transpose()
        df.to_csv(path_to_log_file, mode='a', header=False, index=False)

        self.next_phase_to_set_index = None
        self.current_phase_duration = -1
        self.all_red_flag = False
        self.all_yellow_flag = False
        self.flicker = 0

        self.dic_vehicle_min_speed = {}  # this second
        self.dic_vehicle_arrive_leave_time = dict()  # cumulative

        self.dic_feature = {}  # this second
        self.dic_feature_previous_step = {}  # this second

        self.lane_wait = ([0],[0],[0],[0]) #ulya what length

    def build_adjacency_row_lane(self, lane_id_to_global_index_dict):

        self.adjacency_row_lanes = []  # order is the entering lane order, each element is list of two lists

        for entering_lane_id in self.list_entering_lanes:

            _top_k_entering_lane, _top_k_leaving_lane = self._adjacency_row_lanes[entering_lane_id]
            top_k_entering_lane = []
            top_k_leaving_lane = []

            for lane_id in _top_k_entering_lane:
                top_k_entering_lane.append(lane_id_to_global_index_dict[lane_id] if lane_id is not None else -1)

            for lane_id in _top_k_leaving_lane:
                # TODO leaving lanes of system will also have -1
                top_k_leaving_lane.append(lane_id_to_global_index_dict[lane_id]
                                          if (lane_id is not None) and
                                             (lane_id in lane_id_to_global_index_dict.keys())
                                          else -1)

            self.adjacency_row_lanes.append([top_k_entering_lane, top_k_leaving_lane])

    # set
    def set_signal(self, action, action_pattern, yellow_time, all_red_time):

        if self.all_yellow_flag:

            # in yellow phase
            self.flicker = 0

            if self.current_phase_duration >= yellow_time:  # yellow time reached
                self.current_phase_index = self.next_phase_to_set_index
                self.eng.set_tl_phase(self.inter_name, self.current_phase_index)

                # if multi_phase, need more adjustment
                # print("Changing to phase", self.current_phase_index,
                # "After time", self.current_phase_duration, "for inter", self.inter_name)

                path_to_log_file = os.path.join(self.path_to_log, "signal_inter_{0}.txt".format(self.inter_name))
                df = [self.get_current_time(), self.current_phase_index]
                df = pd.DataFrame(df)
                df = df.transpose()
                df.to_csv(path_to_log_file, mode='a', header=False, index=False)
                self.all_yellow_flag = False

            else:
                pass
        else:
            # determine phase
            if action_pattern == "switch":  # switch by order
                if action == 0:  # keep the phase
                    self.next_phase_to_set_index = self.current_phase_index
                elif action == 1:  # change to the next phase
                    self.next_phase_to_set_index = (self.current_phase_index + 1) % len(self.list_phases)
                    # if multi_phase, need more adjustment
                else:
                    sys.exit("action not recognized\n action must be 0 or 1")

            elif action_pattern == "set":  # set to certain phase
                self.next_phase_to_set_index = self.DIC_PHASE_MAP[action]  # if multi_phase, need more adjustment

            # set phase
            if self.current_phase_index == self.next_phase_to_set_index:  # the light phase keeps unchanged
                pass
            else:
                # the light phase needs to change
                # change to yellow first, and activate the counter and flag

                self.eng.set_tl_phase(self.inter_name, 0)  # !!! yellow, tmp
                path_to_log_file = os.path.join(self.path_to_log, "signal_inter_{0}.txt".format(self.inter_name))

                df = [self.get_current_time(), self.current_phase_index]
                df = pd.DataFrame(df)
                df = df.transpose()
                df.to_csv(path_to_log_file, mode='a', header=False, index=False)

                # traci.trafficlights.setRedYellowGreenState(
                #    self.node_light, self.all_yellow_phase_str)

                self.current_phase_index = self.all_yellow_phase_index
                self.all_yellow_flag = True
                self.flicker = 1

    # update inner measurements

    def update_previous_measurements(self):
        self.previous_phase_index = self.current_phase_index

        self.dic_lane_vehicle_previous_step = self.dic_lane_vehicle_current_step
        self.dic_lane_waiting_vehicle_count_previous_step = self.dic_lane_waiting_vehicle_count_current_step
        self.dic_vehicle_speed_previous_step = self.dic_vehicle_speed_current_step
        self.dic_vehicle_distance_previous_step = self.dic_vehicle_distance_current_step

    def update_current_measurements_map(self, simulator_state):

        # need change, debug in seeing format

        def _change_lane_vehicle_dic_to_list(dic_lane_vehicle):
            list_lane_vehicle = []

            for value in dic_lane_vehicle.values():
                list_lane_vehicle.extend(value)

            return list_lane_vehicle

        if self.current_phase_index == self.previous_phase_index:
            self.current_phase_duration += 1
        else:
            self.current_phase_duration = 1

        self.dic_lane_vehicle_current_step = {}
        self.dic_lane_waiting_vehicle_count_current_step = {}

        for lane in self.list_entering_lanes:
            self.dic_lane_vehicle_current_step[lane] = simulator_state["get_lane_vehicles"][lane]
            self.dic_lane_waiting_vehicle_count_current_step[lane] = \
                simulator_state["get_lane_waiting_vehicle_count"][lane]

        for lane in self.list_exiting_lanes:
            self.dic_lane_vehicle_current_step[lane] = simulator_state["get_lane_vehicles"][lane]
            self.dic_lane_waiting_vehicle_count_current_step[lane] = \
                simulator_state["get_lane_waiting_vehicle_count"][lane]

        self.dic_vehicle_speed_current_step = simulator_state['get_vehicle_speed']
        self.dic_vehicle_distance_current_step = simulator_state['get_vehicle_distance']

        # get vehicle list
        self.list_lane_vehicle_current_step = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_current_step)
        self.list_lane_vehicle_previous_step = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_previous_step)

        list_vehicle_new_arrive = list(set(self.list_lane_vehicle_current_step) -
                                       set(self.list_lane_vehicle_previous_step))

        list_vehicle_new_left = list(set(self.list_lane_vehicle_previous_step) -
                                     set(self.list_lane_vehicle_current_step))

        list_vehicle_new_left_entering_lane_by_lane = self._update_leave_entering_approach_vehicle()
        list_vehicle_new_left_entering_lane = []

        for l_val in list_vehicle_new_left_entering_lane_by_lane:
            list_vehicle_new_left_entering_lane += l_val

        # update vehicle arrive and left time
        self._update_arrive_time(list_vehicle_new_arrive)
        self._update_left_time(list_vehicle_new_left_entering_lane)

        # update vehicle minimum speed in history, # to be implemented
        # self._update_vehicle_min_speed()

        # update feature
        self._update_feature_map(simulator_state)

    # def update_current_measurements(self):
    #
    #     # need change, debug in seeing format
    #
    #     def _change_lane_vehicle_dic_to_list(dic_lane_vehicle):
    #         list_lane_vehicle = []
    #
    #         for value in dic_lane_vehicle.values():
    #             list_lane_vehicle.extend(value)
    #
    #         return list_lane_vehicle
    #
    #     if self.current_phase_index == self.previous_phase_index:
    #         self.current_phase_duration += 1
    #     else:
    #         self.current_phase_duration = 1
    #
    #     self.dic_lane_vehicle_current_step = []  # = self.eng.get_lane_vehicles()
    #
    #     # not implement
    #
    #     flow_tmp = self.eng.get_lane_vehicles()
    #     self.dic_lane_vehicle_current_step = {key: None for key in self.list_entering_lanes}
    #
    #     for lane in self.list_entering_lanes:
    #         self.dic_lane_vehicle_current_step[lane] = flow_tmp[lane]
    #
    #     self.dic_lane_waiting_vehicle_count_current_step = self.eng.get_lane_waiting_vehicle_count()
    #     self.dic_vehicle_speed_current_step = self.eng.get_vehicle_speed()
    #     self.dic_vehicle_distance_current_step = self.eng.get_vehicle_distance()
    #
    #     # get vehicle list
    #     self.list_lane_vehicle_current_step = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_current_step)
    #     self.list_lane_vehicle_previous_step = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_previous_step)
    #
    #     list_vehicle_new_arrive = list(set(self.list_lane_vehicle_current_step) -
    #                                    set(self.list_lane_vehicle_previous_step))
    #
    #     list_vehicle_new_left = list(set(self.list_lane_vehicle_previous_step) -
    #                                  set(self.list_lane_vehicle_current_step))
    #
    #     list_vehicle_new_left_entering_lane_by_lane = self._update_leave_entering_approach_vehicle()
    #     list_vehicle_new_left_entering_lane = []
    #
    #     for l in list_vehicle_new_left_entering_lane_by_lane:
    #         list_vehicle_new_left_entering_lane += l
    #
    #     # update vehicle arrive and left time
    #     self._update_arrive_time(list_vehicle_new_arrive)
    #     self._update_left_time(list_vehicle_new_left_entering_lane)
    #
    #     # update vehicle minimum speed in history, # to be implemented
    #     # self._update_vehicle_min_speed()
    #
    #     # update feature
    #
    #     self._update_feature()

    def _update_leave_entering_approach_vehicle(self):

        list_entering_lane_vehicle_left = []

        # update vehicles leaving entering lane
        if not self.dic_lane_vehicle_previous_step:
            for lane in self.list_entering_lanes:
                list_entering_lane_vehicle_left.append([])
        else:
            last_step_vehicle_id_list = []
            current_step_vehilce_id_list = []
            for lane in self.list_entering_lanes:
                last_step_vehicle_id_list.extend(self.dic_lane_vehicle_previous_step[lane])
                current_step_vehilce_id_list.extend(self.dic_lane_vehicle_current_step[lane])

            list_entering_lane_vehicle_left.append(
                list(set(last_step_vehicle_id_list) - set(current_step_vehilce_id_list))
            )

        return list_entering_lane_vehicle_left

    def _update_arrive_time(self, list_vehicle_arrive):

        ts = self.get_current_time()
        # get dic vehicle enter leave time
        for vehicle in list_vehicle_arrive:
            if vehicle not in self.dic_vehicle_arrive_leave_time:
                self.dic_vehicle_arrive_leave_time[vehicle] = \
                    {"enter_time": ts, "leave_time": np.nan}
            else:
                # print("vehicle: %s already exists in entering lane!"%vehicle)
                # sys.exit(-1)
                pass

    def _update_left_time(self, list_vehicle_left):

        ts = self.get_current_time()
        # update the time for vehicle to leave entering lane
        for vehicle in list_vehicle_left:
            try:
                self.dic_vehicle_arrive_leave_time[vehicle]["leave_time"] = ts
            except KeyError:
                print("vehicle not recorded when entering")
                sys.exit(-1)

    def update_neighbor_info(self, neighbors, dic_feature):
        # print(dic_feature)
        none_dic_feature = deepcopy(dic_feature)
        for key in none_dic_feature.keys():
            if none_dic_feature[key] is not None:
                if "cur_phase" in key:
                    none_dic_feature[key] = [1] * len(none_dic_feature[key])
                else:
                    none_dic_feature[key] = [0] * len(none_dic_feature[key])
            else:
                none_dic_feature[key] = None
        for i in range(len(neighbors)):
            neighbor = neighbors[i]
            example_dic_feature = {}
            if neighbor is None:
                example_dic_feature["cur_phase_{0}".format(i)] = none_dic_feature["cur_phase"]
                example_dic_feature["time_this_phase_{0}".format(i)] = none_dic_feature["time_this_phase"]
                example_dic_feature["lane_num_vehicle_{0}".format(i)] = none_dic_feature["lane_num_vehicle"]
            else:
                example_dic_feature["cur_phase_{0}".format(i)] = neighbor.dic_feature["cur_phase"]
                example_dic_feature["time_this_phase_{0}".format(i)] = neighbor.dic_feature["time_this_phase"]
                example_dic_feature["lane_num_vehicle_{0}".format(i)] = neighbor.dic_feature["lane_num_vehicle"]
            dic_feature.update(example_dic_feature)
        return dic_feature

    @staticmethod
    def _add_suffix_to_dict_key(target_dict, suffix):
        keys = list(target_dict.keys())
        for key in keys:
            target_dict[key+"_"+suffix] = target_dict.pop(key)
        return target_dict

    def _update_feature_map(self, simulator_state):

        dic_feature = dict()

        dic_feature["cur_phase"] = [self.current_phase_index]
        dic_feature["time_this_phase"] = [self.current_phase_duration]
        dic_feature["vehicle_position_img"] = None  # self._get_lane_vehicle_position(self.list_entering_lanes)
        dic_feature["vehicle_speed_img"] = None  # self._get_lane_vehicle_speed(self.list_entering_lanes)
        dic_feature["vehicle_acceleration_img"] = None
        dic_feature["vehicle_waiting_time_img"] = None
        # self._get_lane_vehicle_accumulated_waiting_time(self.list_entering_lanes)

        dic_feature["lane_num_vehicle"] = self._get_lane_num_vehicle(self.list_entering_lanes)
        dic_feature["pressure"] = None  # [self._get_pressure()]

        if self.fast_compute:
            dic_feature["coming_vehicle"] = None
            dic_feature["leaving_vehicle"] = None
        else:
            dic_feature["coming_vehicle"] = self._get_coming_vehicles(simulator_state)
            dic_feature["leaving_vehicle"] = self._get_leaving_vehicles(simulator_state)

        dic_feature["lane_num_vehicle_been_stopped_thres01"] = None
        # self._get_lane_num_vehicle_been_stopped(0.1, self.list_entering_lanes)
        dic_feature["lane_num_vehicle_been_stopped_thres1"] = \
            self._get_lane_num_vehicle_been_stopped(1, self.list_entering_lanes)

        dic_feature["lane_queue_length"] = None  # self._get_lane_queue_length(self.list_entering_lanes)
        dic_feature["lane_num_vehicle_left"] = None
        dic_feature["lane_sum_duration_vehicle_left"] = None
        dic_feature["lane_sum_waiting_time"] = None  # self._get_lane_sum_waiting_time(self.list_entering_lanes)
        dic_feature["terminal"] = None

        dic_feature["adjacency_matrix"] = self._get_adjacency_row()
        # TODO this feature should be a dict? or list of lists

        dic_feature["adjacency_matrix_lane"] = self._get_adjacency_row_lane()
        # row: entering_lane # columns: [inputlanes, outputlanes]

        dic_feature['connectivity'] = self._get_connectivity(self.neighbor_lanes_ENWS)

        dic_feature["lane_wait"] = self._get_lane_wait()

        self.dic_feature = dic_feature

    # ================= calculate features from current observations ======================

    def _get_car_per_phase_wait(self):
        # ANON_PHASE_REPRE = {
        #     1: [0, 1, 0, 1, 0, 0, 0, 0],
        #     2: [0, 0, 0, 0, 0, 1, 0, 1],
        #     3: [1, 0, 1, 0, 0, 0, 0, 0],
        #     4: [0, 0, 0, 0, 1, 0, 1, 0],
        # }
        ANON_PHASE_REPE = [[0, 1, 0, 1, 0, 0, 0, 0],[0, 0, 0, 0, 0, 1, 0, 1],[1, 0, 1, 0, 0, 0, 0, 0],[0, 0, 0, 0, 1, 0, 1, 0]]
        num_vehicles_stopped = self._get_lane_num_vehicle_been_stopped(1, self.list_entering_lanes) # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        num_vehicles_stopped = [num_vehicles_stopped[i] for i in range(len(num_vehicles_stopped)) if (i + 1) % 3 != 0 ]
        car_per_phase_wait = [np.sum(np.logical_and(ANON_PHASE_REPE[i], num_vehicles_stopped)) for i in range(4)]
        return [1 if cars > 0 else 0 for cars in car_per_phase_wait]
        
    def _get_lane_wait(self): 
        # print("ulya lane wait\n")
        # print(self.lane_wait)
        all_phases = [0,1,2,3]
        curr_state = self.current_phase_index
        wait_time = self.current_phase_duration

        cars_at_phase = self._get_car_per_phase_wait()
        # print(cars_at_phase)

        if curr_state in [-1,-2]:
            for state in all_phases:
                self.lane_wait[state][-1] += 1.
        else:
            for state in all_phases:
                if state == curr_state and self.lane_wait[state][-1] != 0:
                    self.lane_wait[state].append(0)
                else:
                    self.lane_wait[state][-1] += 1.
        return self.lane_wait

    def _get_adjacency_row(self):
        return self.adjacency_row

    def _get_adjacency_row_lane(self):
        return self.adjacency_row_lanes

    def lane_position_mapper(self, lane_pos, bins):
        lane_pos_np = np.array(lane_pos)
        digitized = np.digitize(lane_pos_np, bins)
        position_counter = [len(lane_pos_np[digitized == i]) for i in range(1, len(bins))]
        return position_counter

    def _get_coming_vehicles(self, simulator_state):

        # TODO f vehicle position   eng.get_vehicle_distance()  ||  eng.get_lane_vehicles()

        coming_distribution = []
        # dimension = num_lane*3*num_list_entering_lanes

        lane_vid_mapping_dict = simulator_state['get_lane_vehicles']
        vid_distance_mapping_dict = simulator_state['get_vehicle_distance']

        # TODO LANE LENGTH = 300
        bins = np.linspace(0, 300, 4).tolist()

        for lane in self.list_entering_lanes:
            coming_vehicle_position = []
            vehicle_position_lane = lane_vid_mapping_dict[lane]
            for vehicle in vehicle_position_lane:
                coming_vehicle_position.append(vid_distance_mapping_dict[vehicle])
            coming_distribution.extend(self.lane_position_mapper(coming_vehicle_position, bins))

        return coming_distribution

    def _get_leaving_vehicles(self, simulator_state):
        leaving_distribution = []
        # dimension = num_lane*3*num_list_entering_lanes

        lane_vid_mapping_dict = simulator_state['get_lane_vehicles']
        vid_distance_mapping_dict = simulator_state['get_vehicle_distance']

        # TODO LANE LENGTH = 300
        bins = np.linspace(0, 300, 4).tolist()

        for lane in self.list_exiting_lanes:
            coming_vehicle_position = []
            vehicle_position_lane = lane_vid_mapping_dict[lane]
            for vehicle in vehicle_position_lane:
                coming_vehicle_position.append(vid_distance_mapping_dict[vehicle])
            leaving_distribution.extend(self.lane_position_mapper(coming_vehicle_position, bins))

        return leaving_distribution

    def _get_pressure(self):

        # TODO eng.get_vehicle_distance(), another way to calculate pressure & queue length

        pressure = 0
        all_enter_car_queue = 0
        for lane in self.list_entering_lanes:
            all_enter_car_queue += self.dic_lane_waiting_vehicle_count_current_step[lane]

        all_leaving_car_queue = 0
        for lane in self.list_exiting_lanes:
            all_leaving_car_queue += self.dic_lane_waiting_vehicle_count_current_step[lane]

        p = all_enter_car_queue - all_leaving_car_queue

        if p < 0:
            p = -p

        return p

    def _get_lane_queue_length(self, list_lanes):
        """
        queue length for each lane
        """
        return [self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in list_lanes]

    def _get_lane_num_vehicle(self, list_lanes):
        """
        vehicle number for each lane
        """
        return [len(self.dic_lane_vehicle_current_step[lane]) for lane in list_lanes]

    def _get_connectivity(self, dic_of_list_lanes):
        """
        vehicle number for each lane
        """
        result = []

        for i in range(len(dic_of_list_lanes['lane_ids'])):
            num_of_vehicles_on_road = sum([len(self.dic_lane_vehicle_current_step[lane])
                                           for lane in dic_of_list_lanes['lane_ids'][i]])

            result.append(num_of_vehicles_on_road)

        lane_length = [0] + dic_of_list_lanes['lane_length']
        if np.sum(result) == 0:
            result = [1] + result
        else:
            result = [np.sum(result)] + result

        connectivity = list(np.array(result * np.exp(-np.array(lane_length)/(self.length_lane*4))))
        # print(connectivity)
        # sys.exit()
        return connectivity

    def _get_lane_sum_waiting_time(self, list_lanes):
        """
        waiting time for each lane
        """
        raise NotImplementedError

    def _get_lane_list_vehicle_left(self, list_lanes):
        """
        get list of vehicles left at each lane
        ####### need to check
        """

        raise NotImplementedError

    # non temporary
    def _get_lane_num_vehicle_left(self, list_lanes):

        list_lane_vehicle_left = self._get_lane_list_vehicle_left(list_lanes)
        list_lane_num_vehicle_left = [len(lane_vehicle_left) for lane_vehicle_left in list_lane_vehicle_left]
        return list_lane_num_vehicle_left

    def _get_lane_sum_duration_vehicle_left(self, list_lanes):

        # not implemented error
        raise NotImplementedError

    def _get_lane_num_vehicle_been_stopped(self, thres, list_lanes):
        return [self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in list_lanes]

    # def _get_position_grid_along_lane(self, vec):
    #     pos = int(self.dic_vehicle_sub_current_step[vec][get_traci_constant_mapping("VAR_LANEPOSITION")])
    #     return min(pos//self.length_grid, self.num_grid)

    def _get_lane_vehicle_position(self, list_lanes):

        list_lane_vector = []
        for lane in list_lanes:
            lane_vector = np.zeros(self.num_grid)
            list_vec_id = self.dic_lane_vehicle_current_step[lane]
            for vec in list_vec_id:
                pos = int(self.dic_vehicle_distance_current_step[vec])
                pos_grid = min(pos//self.length_grid, self.num_grid)
                lane_vector[pos_grid] = 1
            list_lane_vector.append(lane_vector)
        return np.array(list_lane_vector)

    # debug
    def _get_vehicle_info(self, veh_id):
        try:
            pos = self.dic_vehicle_distance_current_step[veh_id]
            speed = self.dic_vehicle_speed_current_step[veh_id]
            return pos, speed
        except:
            return None, None

    def _get_lane_vehicle_speed(self, list_lanes):
        return [self.dic_vehicle_speed_current_step[lane] for lane in list_lanes]

    def _get_lane_vehicle_accumulated_waiting_time(self, list_lanes):

        raise NotImplementedError

    # ================= get functions from outside ======================
    def get_lane_wait(self):
        return self.get_lane_wait

    def get_current_time(self):
        return self.eng.get_current_time()

    def get_dic_vehicle_arrive_leave_time(self):

        return self.dic_vehicle_arrive_leave_time

    def get_feature(self):

        return self.dic_feature

    def get_state(self, list_state_features):

        # customize your own state
        # print(list_state_features)
        # print(self.dic_feature)

        list_state_features = list_state_features + ['time_this_phase', 'lane_wait']
        dic_state = {state_feature_name: self.dic_feature[state_feature_name] for state_feature_name in
                     list_state_features}

        return dic_state

    def get_reward(self, dic_reward_info):

        # customize your own reward

        dic_reward = dict()
        dic_reward["flickering"] = None
        dic_reward["sum_lane_queue_length"] = None
        dic_reward["sum_lane_wait_time"] = None
        dic_reward["sum_lane_num_vehicle_left"] = None
        dic_reward["sum_duration_vehicle_left"] = None
        dic_reward["sum_num_vehicle_been_stopped_thres01"] = None
        dic_reward["sum_num_vehicle_been_stopped_thres1"] = \
            np.sum(self.dic_feature["lane_num_vehicle_been_stopped_thres1"])

        dic_reward['pressure'] = None # np.sum(self.dic_feature["pressure"])

        reward = 0
        for r in dic_reward_info:
            if dic_reward_info[r] != 0:
                reward += dic_reward_info[r] * dic_reward[r]
        return reward

