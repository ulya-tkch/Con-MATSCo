"""
Creation Date : 4th Dec 2021
Last Updated : 4th Dec 2021
Author/s : Mayank Sharan

Class to manage interactions with Cityflow simulator
"""

# Base python imports

import os
import time
import json
import pickle
import numpy as np
import pandas as pd
import cityflow as engine

from copy import deepcopy
from multiprocessing import Process

# Import from modules

from anon_env.roadnet import RoadNet
from anon_env.intersection import Intersection


class AnonEnv:

    list_intersection_id = [
        "intersection_1_1"
    ]

    def __init__(self, path_to_log, path_to_work_directory, dic_traffic_env_conf):
        self.path_to_log = path_to_log
        self.path_to_work_directory = path_to_work_directory
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.simulator_type = self.dic_traffic_env_conf["SIMULATOR_TYPE"]

        self.list_intersection = None
        self.list_inter_log = None
        self.list_lanes = None
        self.system_states = None
        self.feature_name_for_neighbor = self._reduce_duplicates(self.dic_traffic_env_conf["LIST_STATE_FEATURE"])

        # check min action time
        if self.dic_traffic_env_conf["MIN_ACTION_TIME"] <= self.dic_traffic_env_conf["YELLOW_TIME"]:
            print("MIN_ACTION_TIME should include YELLOW_TIME")
            pass
            # raise ValueError

        # touch new inter_{}.pkl (if exists, remove)
        for inter_ind in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            f.close()

    def reset(self):

        print("# self.eng.reset() to be implemented")

        cityflow_config = {
            "interval": self.dic_traffic_env_conf["INTERVAL"],
            "seed": 0,
            "laneChange": False,
            "dir": self.path_to_work_directory+"/",
            "roadnetFile": self.dic_traffic_env_conf["ROADNET_FILE"],
            "flowFile": self.dic_traffic_env_conf["TRAFFIC_FILE"],
            "rlTrafficLight": self.dic_traffic_env_conf["RLTRAFFICLIGHT"],
            "saveReplay": self.dic_traffic_env_conf["SAVEREPLAY"],
            "roadnetLogFile": "replay_roadnet.json",
            "replayLogFile": "replay.txt"
        }
        print("=========================")
        print(cityflow_config)

        with open(os.path.join(self.path_to_work_directory, "cityflow.config"), "w") as json_file:
            json.dump(cityflow_config, json_file)

        self.eng = engine.Engine(os.path.join(self.path_to_work_directory, "cityflow.config"), thread_num=1)

        # self.load_roadnet()
        # self.load_flow()

        # get adjacency
        if self.dic_traffic_env_conf["USE_LANE_ADJACENCY"]:
            self.traffic_light_node_dict = self._adjacency_extraction_lane()
        else:
            self.traffic_light_node_dict = self._adjacency_extraction()

        # initialize intersections (grid)
        self.list_intersection = [Intersection((i+1, j+1), self.dic_traffic_env_conf, self.eng,
                                               self.traffic_light_node_dict["intersection_{0}_{1}".format(i+1, j+1)],
                                               self.path_to_log)
                                  for i in range(self.dic_traffic_env_conf["NUM_ROW"])
                                  for j in range(self.dic_traffic_env_conf["NUM_COL"])]

        self.list_inter_log = [[] for i in range(self.dic_traffic_env_conf["NUM_ROW"] *
                                                 self.dic_traffic_env_conf["NUM_COL"])]

        # set index for intersections and global index for lanes

        self.id_to_index = {}
        count_inter = 0

        for i in range(self.dic_traffic_env_conf["NUM_ROW"]):
            for j in range(self.dic_traffic_env_conf["NUM_COL"]):
                self.id_to_index['intersection_{0}_{1}'.format(i+1, j+1)] = count_inter
                count_inter += 1

        self.lane_id_to_index = {}
        count_lane = 0

        for i in range(len(self.list_intersection)): # TODO
            for j in range(len(self.list_intersection[i].list_entering_lanes)):
                lane_id = self.list_intersection[i].list_entering_lanes[j]
                if lane_id not in self.lane_id_to_index.keys():
                    self.lane_id_to_index[lane_id] = count_lane
                    count_lane += 1

        # build adjacency_matrix_lane in index from _adjacency_matrix_lane
        for inter in self.list_intersection:
            inter.build_adjacency_row_lane(self.lane_id_to_index)

        # get new measurements

        system_state_start_time = time.time()
        if self.dic_traffic_env_conf["FAST_COMPUTE"]:
            self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                                  "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                                  "get_vehicle_speed": None,
                                  "get_vehicle_distance": None
                                  }
        else:
            self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                                  "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                                  "get_vehicle_speed": self.eng.get_vehicle_speed(),
                                  "get_vehicle_distance": self.eng.get_vehicle_distance()
                                  }
        print("Get system state time: ", time.time()-system_state_start_time)

        update_start_time = time.time()
        for inter in self.list_intersection:
            inter.update_current_measurements_map(self.system_states)
        print("Update_current_measurements_map time: ", time.time()-update_start_time)

        # update neighbor's info
        neighbor_start_time = time.time()
        if self.dic_traffic_env_conf["NEIGHBOR"]:
            for inter in self.list_intersection:
                neighbor_inter_ids = inter.neighbor_ENWS
                neighbor_inters = []
                for neighbor_inter_id in neighbor_inter_ids:
                    if neighbor_inter_id is not None:
                        neighbor_inters.append(self.list_intersection[self.id_to_index[neighbor_inter_id]])
                    else:
                        neighbor_inters.append(None)
                inter.dic_feature = inter.update_neighbor_info(neighbor_inters,deepcopy(inter.dic_feature))
        print("Update_neighbor time: ", time.time()-neighbor_start_time)

        state, done = self.get_state()
        # print(state)
        return state

    def step(self, action):

        step_start_time = time.time()
        list_action_in_sec = [action]
        list_action_in_sec_display = [action]

        # print(self.dic_traffic_env_conf["MIN_ACTION_TIME"])

        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"]-1):
            if self.dic_traffic_env_conf["ACTION_PATTERN"] == "switch":
                list_action_in_sec.append(np.zeros_like(action).tolist())
            elif self.dic_traffic_env_conf["ACTION_PATTERN"] == "set":
                list_action_in_sec.append(np.copy(action).tolist())
            list_action_in_sec_display.append(np.full_like(action, fill_value=-1).tolist())

        # print(len(list_action_in_sec), len(list_action_in_sec[0]))

        average_reward_action_list = [0]*len(action)
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"]):

            action_in_sec = list_action_in_sec[i]
            action_in_sec_display = list_action_in_sec_display[i]

            instant_time = self.get_current_time()
            self.current_time = self.get_current_time()

            before_action_feature = self.get_feature()
            # state = self.get_state()

            # if self.dic_traffic_env_conf['DEBUG']:
            #     print("time: {0}".format(instant_time))
            #     print(action_in_sec)
            # else:
            #     if i == 0:
            #         print("time: {0}".format(instant_time))

            self._inner_step(action_in_sec)

            # get reward
            if self.dic_traffic_env_conf['DEBUG']:
                start_time = time.time()

            reward = self.get_reward()

            # # rohin code
            # inter_idx = -1
            # for inter in self.list_intersection:
            #     inter_idx+=1
            #     phase_time = inter.dic_feature["time_this_phase"][0]
            #     state = inter.get_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"])
            #     cur_phase = state["cur_phase"][0]
            #     # print()
            #     # print("testing")
            #     # print(phase_time, curr_phase, action_in_sec[inter_idx])
            #
            #     # check constraint
            #     min_switch_time = 15
            #     if cur_phase != -1:
            #         if phase_time <= min_switch_time and cur_phase != action_in_sec[inter_idx]:
            #             # print("applying constraint")
            #             reward[inter_idx] += -10.0

            if self.dic_traffic_env_conf['DEBUG']:
                print("Reward time: {}".format(time.time()-start_time))

            for j in range(len(reward)):
                average_reward_action_list[j] = (average_reward_action_list[j] * i + reward[j]) / (i + 1)

            # average_reward_action = (average_reward_action*i + reward[0])/(i+1)

            # log
            self.log(cur_time=instant_time, before_action_feature=before_action_feature, action=action_in_sec_display)

            next_state, done = self.get_state()

        # print("Step time: ", time.time() - step_start_time)
        return next_state, reward, done, average_reward_action_list

    def _inner_step(self, action):

        # copy current measurements to previous measurements
        for inter in self.list_intersection:
            inter.update_previous_measurements()

        # set signals
        # multi_intersection decided by action {inter_id: phase}
        for inter_ind, inter in enumerate(self.list_intersection):
            inter.set_signal(
                action=action[inter_ind],
                action_pattern=self.dic_traffic_env_conf["ACTION_PATTERN"],
                yellow_time=self.dic_traffic_env_conf["YELLOW_TIME"],
                all_red_time=self.dic_traffic_env_conf["ALL_RED_TIME"]
            )

        # run one step
        for i in range(int(1/self.dic_traffic_env_conf["INTERVAL"])):
            self.eng.next_step()

        if self.dic_traffic_env_conf['DEBUG']:
            start_time = time.time()

        system_state_start_time = time.time()
        if self.dic_traffic_env_conf["FAST_COMPUTE"]:
            self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                                  "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                                  "get_vehicle_speed": None,
                                  "get_vehicle_distance": None
                                  }
        else:
            self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                                  "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                                  "get_vehicle_speed": self.eng.get_vehicle_speed(),
                                  "get_vehicle_distance": self.eng.get_vehicle_distance()
                                  }

        # print("Get system state time: ", time.time()-system_state_start_time)

        if self.dic_traffic_env_conf['DEBUG']:
            print("Get system state time: {}".format(time.time()-start_time))
        # get new measurements

        if self.dic_traffic_env_conf['DEBUG']:
            start_time = time.time()

        update_start_time = time.time()
        for inter in self.list_intersection:
            inter.update_current_measurements_map(self.system_states)

        # update neighbor's info
        if self.dic_traffic_env_conf["NEIGHBOR"]:
            for inter in self.list_intersection:
                neighbor_inter_ids = inter.neighbor_ENWS
                neighbor_inters = []
                for neighbor_inter_id in neighbor_inter_ids:
                    if neighbor_inter_id is not None:
                        neighbor_inters.append(self.list_intersection[self.id_to_index[neighbor_inter_id]])
                    else:
                        neighbor_inters.append(None)
                inter.dic_feature = inter.update_neighbor_info(neighbor_inters, deepcopy(inter.dic_feature))

        # print("Update_current_measurements_map time: ", time.time()-update_start_time)

        if self.dic_traffic_env_conf['DEBUG']:
            print("Update measurements time: {}".format(time.time()-start_time))

        # self.log_lane_vehicle_position()
        # self.log_first_vehicle()
        # self.log_phase()

    def load_roadnet(self, roadnetFile=None):
        print("Start load roadnet")
        start_time = time.time()
        if not roadnetFile:
            roadnetFile = "roadnet_1_1.json"
        # print("/n/n", os.path.join(self.path_to_work_directory, roadnetFile))
        self.eng.load_roadnet(os.path.join(self.path_to_work_directory, roadnetFile))
        print("successfully load roadnet:{0}, time: {1}".format(roadnetFile,time.time()-start_time))

    def load_flow(self, flowFile=None):
        print("Start load flowFile")
        start_time = time.time()
        if not flowFile:
            flowFile = "flow_1_1.json"
        self.eng.load_flow(os.path.join(self.path_to_work_directory, flowFile))
        print("successfully load flowFile: {0}, time: {1}".format(flowFile, time.time()-start_time))

    def _check_episode_done(self, list_state):

        # ======== to implement ========

        return False

    @staticmethod
    def convert_dic_to_df(dic):
        list_df = []
        for key in dic:
            df = pd.Series(dic[key], name=key)
            list_df.append(df)
        return pd.DataFrame(list_df)

    def get_feature(self):
        list_feature = [inter.get_feature() for inter in self.list_intersection]
        return list_feature

    def get_state(self):
        # consider neighbor info
        list_state = [inter.get_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"]) for inter in self.list_intersection]
        done = self._check_episode_done(list_state)

        # print(list_state)

        return list_state, done

    @staticmethod
    def _reduce_duplicates(feature_name_list):
        new_list = set()
        for feature_name in feature_name_list:
            if feature_name[-1] in ["0","1","2","3"]:
                new_list.add(feature_name[:-2])
        return list(new_list)

    def get_reward(self):

        list_reward = [inter.get_reward(self.dic_traffic_env_conf["DIC_REWARD_INFO"]) for inter in self.list_intersection]

        return list_reward

    def get_current_time(self):
        return self.eng.get_current_time()

    def log(self, cur_time, before_action_feature, action):

        for inter_ind in range(len(self.list_intersection)):
            self.list_inter_log[inter_ind].append({"time": cur_time,
                                                   "state": before_action_feature[inter_ind],
                                                   "action": action[inter_ind]})

    def batch_log(self, start, stop):
        for inter_ind in range(start, stop):
            if int(inter_ind)%100 == 0:
                print("Batch log for inter ",inter_ind)
            path_to_log_file = os.path.join(self.path_to_log, "vehicle_inter_{0}.csv".format(inter_ind))
            dic_vehicle = self.list_intersection[inter_ind].get_dic_vehicle_arrive_leave_time()
            df = pd.DataFrame.from_dict(dic_vehicle, orient='index')
            df.to_csv(path_to_log_file, na_rep="nan")

            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            pickle.dump(self.list_inter_log[inter_ind], f)
            f.close()

    def bulk_log_multi_process(self, batch_size=100):
        assert len(self.list_intersection) == len(self.list_inter_log)

        if batch_size > len(self.list_intersection):
            batch_size_run = len(self.list_intersection)
        else:
            batch_size_run = batch_size

        process_list = []

        for batch in range(0, len(self.list_intersection), batch_size_run):
            start = batch
            stop = min(batch + batch_size, len(self.list_intersection))
            p = Process(target=self.batch_log, args=(start, stop))
            print("before")
            p.start()
            print("end")
            process_list.append(p)

        print("before join")

        for t in process_list:
            t.join()

        print("end join")

    def bulk_log(self):

        for inter_ind in range(len(self.list_intersection)):
            path_to_log_file = os.path.join(self.path_to_log, "vehicle_inter_{0}.csv".format(inter_ind))
            dic_vehicle = self.list_intersection[inter_ind].get_dic_vehicle_arrive_leave_time()
            df = self.convert_dic_to_df(dic_vehicle)
            df.to_csv(path_to_log_file, na_rep="nan")

        for inter_ind in range(len(self.list_inter_log)):
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            pickle.dump(self.list_inter_log[inter_ind], f)
            f.close()

        self.eng.print_log(os.path.join(self.path_to_log, self.dic_traffic_env_conf["ROADNET_FILE"]),
                           os.path.join(self.path_to_log, "replay_1_1.txt"))

        # print("log files:", os.path.join("data", "frontend", "roadnet_1_1_test.json"))

    def log_attention(self, attention_dict):
        path_to_log_file = os.path.join(self.path_to_log, "attention.pkl")
        f = open(path_to_log_file, "wb")
        pickle.dump(attention_dict, f)
        f.close()

    def log_hidden_state(self, hidden_states):
        path_to_log_file = os.path.join(self.path_to_log, "hidden_states.pkl")

        with open(path_to_log_file, "wb") as f:
            pickle.dump(hidden_states, f)

    def log_lane_vehicle_position(self):
        def list_to_str(alist):
            new_str = ""
            for s in alist:
                new_str = new_str + str(s) + " "
            return new_str
        dic_lane_map = {
            "road_0_1_0_0": "w",
            "road_2_1_2_0": "e",
            "road_1_0_1_0": "s",
            "road_1_2_3_0": "n"
        }
        for inter in self.list_intersection:
            for lane in inter.list_entering_lanes:
                print(str(self.get_current_time()) + ", " + lane + ", " +
                      list_to_str(inter._get_lane_vehicle_position([lane])[0]),
                      file=open(os.path.join(self.path_to_log, "lane_vehicle_position_%s.txt"%dic_lane_map[lane]), "a"))

    def log_lane_vehicle_position(self):
        def list_to_str(alist):
            new_str = ""
            for s in alist:
                new_str = new_str + str(s) + " "
            return new_str
        dic_lane_map = {
            "road_0_1_0_0": "w",
            "road_2_1_2_0": "e",
            "road_1_0_1_0": "s",
            "road_1_2_3_0": "n"
        }
        for inter in self.list_intersection:
            for lane in inter.list_entering_lanes:
                print(str(self.get_current_time()) + ", " + lane + ", " +
                      list_to_str(inter._get_lane_vehicle_position([lane])[0]),
                      file=open(os.path.join(self.path_to_log, "lane_vehicle_position_%s.txt"%dic_lane_map[lane]), "a"))

    def log_first_vehicle(self):
        _veh_id = "flow_0_"
        _veh_id_2 = "flow_2_"
        _veh_id_3 = "flow_4_"
        _veh_id_4 = "flow_6_"

        for inter in self.list_intersection:
            for i in range(100):
                veh_id = _veh_id + str(i)
                veh_id_2 = _veh_id_2 + str(i)
                pos, speed = inter._get_vehicle_info(veh_id)
                pos_2, speed_2 = inter._get_vehicle_info(veh_id_2)
                # print(i, veh_id, pos, veh_id_2, speed, pos_2, speed_2)
                if not os.path.exists(os.path.join(self.path_to_log, "first_vehicle_info_a")):
                    os.makedirs(os.path.join(self.path_to_log, "first_vehicle_info_a"))

                if not os.path.exists(os.path.join(self.path_to_log, "first_vehicle_info_b")):
                    os.makedirs(os.path.join(self.path_to_log, "first_vehicle_info_b"))

                if pos and speed:
                    print("%f, %f, %f" % (self.get_current_time(), pos, speed),
                          file=open(os.path.join(self.path_to_log, "first_vehicle_info_a", "first_vehicle_info_a_%d.txt" % i), "a"))
                if pos_2 and speed_2:
                    print("%f, %f, %f" % (self.get_current_time(), pos_2, speed_2),
                          file=open(os.path.join(self.path_to_log, "first_vehicle_info_b", "first_vehicle_info_b_%d.txt" % i), "a"))

                veh_id_3 = _veh_id_3 + str(i)
                veh_id_4 = _veh_id_4 + str(i)
                pos_3, speed_3 = inter._get_vehicle_info(veh_id_3)
                pos_4, speed_4 = inter._get_vehicle_info(veh_id_4)
                # print(i, veh_id, pos, veh_id_2, speed, pos_2, speed_2)
                if not os.path.exists(os.path.join(self.path_to_log, "first_vehicle_info_c")):
                    os.makedirs(os.path.join(self.path_to_log, "first_vehicle_info_c"))

                if not os.path.exists(os.path.join(self.path_to_log, "first_vehicle_info_d")):
                    os.makedirs(os.path.join(self.path_to_log, "first_vehicle_info_d"))

                if pos_3 and speed_3:
                    print("%f, %f, %f" % (self.get_current_time(), pos_3, speed_3),
                          file=open(
                              os.path.join(self.path_to_log, "first_vehicle_info_c", "first_vehicle_info_a_%d.txt" % i),
                              "a"))
                if pos_4 and speed_4:
                    print("%f, %f, %f" % (self.get_current_time(), pos_4, speed_4),
                          file=open(
                              os.path.join(self.path_to_log, "first_vehicle_info_d", "first_vehicle_info_b_%d.txt" % i),
                              "a"))

    def log_phase(self):
        for inter in self.list_intersection:
            print("%f, %f" % (self.get_current_time(), inter.current_phase_index),
                  file=open(os.path.join(self.path_to_log, "log_phase.txt"), "a"))

    def _adjacency_extraction(self):
        traffic_light_node_dict = {}
        file = os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf["ROADNET_FILE"])
        with open('{0}'.format(file)) as json_data:
            net = json.load(json_data)
            # print(net)
            for inter in net['intersections']:
                if not inter['virtual']:
                    traffic_light_node_dict[inter['id']] = {'location': {'x': float(inter['point']['x']),
                                                                         'y': float(inter['point']['y'])},
                                                            "total_inter_num": None, 'adjacency_row': None,
                                                            "inter_id_to_index": None,
                                                            "neighbor_ENWS": None,
                                                            "entering_lane_ENWS": None,
                                                            }

            top_k = self.dic_traffic_env_conf["TOP_K_ADJACENCY"]
            total_inter_num = len(traffic_light_node_dict.keys())
            inter_id_to_index = {}

            edge_id_dict = {}
            for road in net['roads']:
                if road['id'] not in edge_id_dict.keys():
                    edge_id_dict[road['id']] = {}
                edge_id_dict[road['id']]['from'] = road['startIntersection']
                edge_id_dict[road['id']]['to'] = road['endIntersection']
                edge_id_dict[road['id']]['num_of_lane'] = len(road['lanes'])
                edge_id_dict[road['id']]['length'] = np.sqrt(np.square(pd.DataFrame(road['points'])).sum(axis=1)).sum()

            index = 0

            for i in traffic_light_node_dict.keys():
                inter_id_to_index[i] = index
                index += 1

            for i in traffic_light_node_dict.keys():

                traffic_light_node_dict[i]['inter_id_to_index'] = inter_id_to_index
                traffic_light_node_dict[i]['neighbor_ENWS'] = []
                traffic_light_node_dict[i]['entering_lane_ENWS'] = {"lane_ids": [], "lane_length": []}
                for j in range(4):
                    road_id = i.replace("intersection", "road")+"_"+str(j)
                    neighboring_node = edge_id_dict[road_id]['to']
                    # calculate the neighboring intersections
                    if neighboring_node not in traffic_light_node_dict.keys(): # virtual node
                        traffic_light_node_dict[i]['neighbor_ENWS'].append(None)
                    else:
                        traffic_light_node_dict[i]['neighbor_ENWS'].append(neighboring_node)
                    # calculate the entering lanes ENWS
                    for key, value in edge_id_dict.items():
                        if value['from'] == neighboring_node and value['to'] == i:
                            neighboring_road = key

                            neighboring_lanes = []
                            for k in range(value['num_of_lane']):
                                neighboring_lanes.append(neighboring_road+"_{0}".format(k))

                            traffic_light_node_dict[i]['entering_lane_ENWS']['lane_ids'].append(neighboring_lanes)
                            traffic_light_node_dict[i]['entering_lane_ENWS']['lane_length'].append(value['length'])


            for i in traffic_light_node_dict.keys():
                location_1 = traffic_light_node_dict[i]['location']

                # TODO return with Top K results
                if not self.dic_traffic_env_conf['ADJACENCY_BY_CONNECTION_OR_GEO']: # use geo-distance
                    row = np.array([0]*total_inter_num)
                    # row = np.zeros((self.dic_traffic_env_conf["NUM_ROW"],self.dic_traffic_env_conf["NUM_col"]))
                    for j in traffic_light_node_dict.keys():
                        location_2 = traffic_light_node_dict[j]['location']
                        dist = AnonEnv._cal_distance(location_1,location_2)
                        row[inter_id_to_index[j]] = dist
                    if len(row) == top_k:
                        adjacency_row_unsorted = np.argpartition(row, -1)[:top_k].tolist()
                    elif len(row) > top_k:
                        adjacency_row_unsorted = np.argpartition(row, top_k)[:top_k].tolist()
                    else:
                        adjacency_row_unsorted = [k for k in range(total_inter_num)]
                    adjacency_row_unsorted.remove(inter_id_to_index[i])
                    traffic_light_node_dict[i]['adjacency_row'] = [inter_id_to_index[i]] + adjacency_row_unsorted
                else: # use connection infomation
                    traffic_light_node_dict[i]['adjacency_row'] = [inter_id_to_index[i]]
                    for j in traffic_light_node_dict[i]['neighbor_ENWS']: ## TODO
                        if j is not None:
                            traffic_light_node_dict[i]['adjacency_row'].append(inter_id_to_index[j])
                        else:
                            traffic_light_node_dict[i]['adjacency_row'].append(-1)


                traffic_light_node_dict[i]['total_inter_num'] = total_inter_num

        return traffic_light_node_dict

    def _adjacency_extraction_lane(self):
        traffic_light_node_dict = {}
        file = os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf["ROADNET_FILE"])

        roadnet = RoadNet('{0}'.format(file))
        with open('{0}'.format(file)) as json_data:
            net = json.load(json_data)
            # print(net)
            for inter in net['intersections']:
                if not inter['virtual']:
                    traffic_light_node_dict[inter['id']] = {'location': {'x': float(inter['point']['x']),
                                                                         'y': float(inter['point']['y'])},
                                                            "total_inter_num": None, 'adjacency_row': None,
                                                            "inter_id_to_index": None,
                                                            "neighbor_ENWS": None,
                                                            "entering_lane_ENWS": None,
                                                            "total_lane_num": None, 'adjacency_matrix_lane': None,
                                                            "lane_id_to_index": None,
                                                            "lane_ids_in_intersction": []
                                                            }

            top_k = self.dic_traffic_env_conf["TOP_K_ADJACENCY"]
            top_k_lane = self.dic_traffic_env_conf["TOP_K_ADJACENCY_LANE"]
            total_inter_num = len(traffic_light_node_dict.keys())

            edge_id_dict = {}
            for road in net['roads']:
                if road['id'] not in edge_id_dict.keys():
                    edge_id_dict[road['id']] = {}
                edge_id_dict[road['id']]['from'] = road['startIntersection']
                edge_id_dict[road['id']]['to'] = road['endIntersection']
                edge_id_dict[road['id']]['num_of_lane'] = len(road['lanes'])
                edge_id_dict[road['id']]['length'] = np.sqrt(np.square(pd.DataFrame(road['points'])).sum(axis=1)).sum()


            # set inter id to index dict
            inter_id_to_index = {}
            index = 0
            for i in traffic_light_node_dict.keys():
                inter_id_to_index[i] = index
                index += 1

            # set the neighbor_ENWS nodes and entring_lane_ENWS for intersections
            for i in traffic_light_node_dict.keys():
                traffic_light_node_dict[i]['inter_id_to_index'] = inter_id_to_index
                traffic_light_node_dict[i]['neighbor_ENWS'] = []
                traffic_light_node_dict[i]['entering_lane_ENWS'] = {"lane_ids": [], "lane_length": []}
                for j in range(4):
                    road_id = i.replace("intersection", "road")+"_"+str(j)
                    neighboring_node = edge_id_dict[road_id]['to']
                    # calculate the neighboring intersections
                    if neighboring_node not in traffic_light_node_dict.keys(): # virtual node
                        traffic_light_node_dict[i]['neighbor_ENWS'].append(None)
                    else:
                        traffic_light_node_dict[i]['neighbor_ENWS'].append(neighboring_node)
                    # calculate the entering lanes ENWS
                    for key, value in edge_id_dict.items():
                        if value['from'] == neighboring_node and value['to'] == i:
                            neighboring_road = key

                            neighboring_lanes = []
                            for k in range(value['num_of_lane']):
                                neighboring_lanes.append(neighboring_road+"_{0}".format(k))

                            traffic_light_node_dict[i]['entering_lane_ENWS']['lane_ids'].append(neighboring_lanes)
                            traffic_light_node_dict[i]['entering_lane_ENWS']['lane_length'].append(value['length'])


            lane_id_dict = roadnet.net_lane_dict
            total_lane_num = len(lane_id_dict.keys())
            # output an adjacentcy matrix for all the intersections
            # each row stands for a lane id,
            # each column is a list with two elements: first is the lane's entering_lane_LSR, second is the lane's leaving_lane_LSR
            def _get_top_k_lane(lane_id_list, top_k_input):
                top_k_lane_indexes = []
                for i in range(top_k_input):
                    lane_id = lane_id_list[i] if i < len(lane_id_list) else None
                    top_k_lane_indexes.append(lane_id)
                return top_k_lane_indexes

            adjacency_matrix_lane = {}
            for i in lane_id_dict.keys(): # Todo lane_ids should be in an order
                adjacency_matrix_lane[i] = [_get_top_k_lane(lane_id_dict[i]['input_lanes'], top_k_lane),
                                            _get_top_k_lane(lane_id_dict[i]['output_lanes'], top_k_lane)]



            for i in traffic_light_node_dict.keys():
                location_1 = traffic_light_node_dict[i]['location']

                # TODO return with Top K results
                if not self.dic_traffic_env_conf['ADJACENCY_BY_CONNECTION_OR_GEO']: # use geo-distance
                    row = np.array([0]*total_inter_num)
                    # row = np.zeros((self.dic_traffic_env_conf["NUM_ROW"],self.dic_traffic_env_conf["NUM_col"]))
                    for j in traffic_light_node_dict.keys():
                        location_2 = traffic_light_node_dict[j]['location']
                        dist = AnonEnv._cal_distance(location_1,location_2)
                        row[inter_id_to_index[j]] = dist
                    if len(row) == top_k:
                        adjacency_row_unsorted = np.argpartition(row, -1)[:top_k].tolist()
                    elif len(row) > top_k:
                        adjacency_row_unsorted = np.argpartition(row, top_k)[:top_k].tolist()
                    else:
                        adjacency_row_unsorted = [k for k in range(total_inter_num)]
                    adjacency_row_unsorted.remove(inter_id_to_index[i])
                    traffic_light_node_dict[i]['adjacency_row'] = [inter_id_to_index[i]] + adjacency_row_unsorted
                else: # use connection infomation
                    traffic_light_node_dict[i]['adjacency_row'] = [inter_id_to_index[i]]
                    for j in traffic_light_node_dict[i]['neighbor_ENWS']: ## TODO
                        if j is not None:
                            traffic_light_node_dict[i]['adjacency_row'].append(inter_id_to_index[j])
                        else:
                            traffic_light_node_dict[i]['adjacency_row'].append(-1)


                traffic_light_node_dict[i]['total_inter_num'] = total_inter_num
                traffic_light_node_dict[i]['total_lane_num'] = total_lane_num
                traffic_light_node_dict[i]['adjacency_matrix_lane'] = adjacency_matrix_lane



        return traffic_light_node_dict



    @staticmethod
    def _cal_distance(loc_dict1, loc_dict2):
        a = np.array((loc_dict1['x'], loc_dict1['y']))
        b = np.array((loc_dict2['x'], loc_dict2['y']))
        return np.sqrt(np.sum((a-b)**2))

    def end_sumo(self):
        print("anon process end")
        pass


