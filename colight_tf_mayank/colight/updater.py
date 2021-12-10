"""
Creation Date : 7th Dec 2021
Last Updated : 7th Dec 2021
Author/s : Mayank Sharan

File to update the Q network
"""

# Base python imports

import os
import time
import pickle
import random
import traceback

import numpy as np
import pandas as pd

# Import from modules

from config import DIC_AGENTS, DIC_ENVS


class Updater:

    def __init__(self, cnt_round, dic_agent_conf, dic_exp_conf, dic_traffic_env_conf, dic_path, best_round=None,
                 bar_round=None):

        self.cnt_round = cnt_round
        self.dic_path = dic_path
        self.dic_exp_conf = dic_exp_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_agent_conf = dic_agent_conf
        self.agents = []
        self.sample_set_list = []
        self.sample_indexes = None

        # temporary path_to_log
        self.path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round", "round_0",
                                        "generator_0")

        env_tmp = DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](
            path_to_log=self.path_to_log,
            path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_conf)
        env_tmp.reset()

        for i in range(dic_traffic_env_conf['NUM_AGENTS']):
            agent_name = self.dic_exp_conf["MODEL_NAME"]
            agent = DIC_AGENTS[agent_name](
                self.dic_agent_conf, self.dic_traffic_env_conf,
                self.dic_path, self.cnt_round, intersection_id=str(i))
            self.agents.append(agent)

    def load_sample(self, i):

        sample_set = []

        try:
            sample_file = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                            "total_samples_inter_{0}".format(i) + ".pkl"), "rb")
            try:
                while True:
                    sample_set += pickle.load(sample_file)
            except EOFError:
                sample_file.close()
                pass

        except Exception as e:

            error_dir = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"]).replace("records", "errors")
            if not os.path.exists(error_dir):
                os.makedirs(error_dir)
            f = open(os.path.join(error_dir, "error_info_inter_{0}.txt".format(i)), "a")
            f.write("Fail to load samples for inter {0}\n".format(i))
            f.write('traceback.format_exc():\n%s\n' % traceback.format_exc())
            f.close()
            print('Updater: traceback.format_exc():\n%s' % traceback.format_exc())
            pass

        if i % 100 == 0:
            print("load_sample for inter {0}".format(i))

        return sample_set

    def load_sample_for_agents(self):

        start_time = time.time()
        print("Updater: Start load samples at", start_time)

        if self.dic_exp_conf['MODEL_NAME'] not in ["GCN", "CoLight"]:

            for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
                sample_set = self.load_sample(i)
                self.agents[i].prepare_Xs_Y(sample_set, self.dic_exp_conf)

        else:

            samples_gcn_df = []
            print("start get samples")
            get_samples_start_time = time.time()

            # for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
            #     sample_set = self.load_sample(i)
            #
            #     samples_set_df = pd.DataFrame.from_records(sample_set, columns=['state', 'action', 'next_state',
            #                                                                     'inst_reward', 'reward', 'time',
            #                                                                     'generator'])
            #
            #     samples_set_df['input'] = samples_set_df[['state', 'action', 'next_state', 'inst_reward',
            #                                               'reward']].values.tolist()
            #
            #     samples_set_df.drop(['state', 'action', 'next_state', 'inst_reward', 'reward', 'time', 'generator'],
            #                         axis=1, inplace=True)
            #
            #     # samples_set_df['inter_id'] = i
            #
            #     samples_gcn_df.append(samples_set_df['input'])
            #
            #     if i % 100 == 0:
            #         print("Updater: inter {0} samples_set_df.shape: ".format(i), samples_set_df.shape)
            #
            # samples_gcn_df = pd.concat(samples_gcn_df, axis=1)

            choice_idx_set = False
            sample_slice = []

            for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
                sample_set = self.load_sample(i)

                if not choice_idx_set:
                    ind_end = len(sample_set)
                    print("Updater: memory size before forget: {0}".format(ind_end))

                    sample_set_idx = np.arange(0, ind_end)
                    ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
                    memory_after_forget = sample_set_idx[ind_sta: ind_end]
                    print("Updater: memory size after forget:", len(memory_after_forget))

                    # sample the memory
                    sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(memory_after_forget))
                    sample_slice = random.sample(memory_after_forget.tolist(), sample_size)
                    print("Updater: memory samples number:", sample_size)

                    choice_idx_set = True

                sel_sample_set = np.array(sample_set)
                sel_sample_set = sel_sample_set[sample_slice]

                samples_set_df = pd.DataFrame.from_records(sel_sample_set,
                                                           columns=['state', 'action', 'next_state', 'inst_reward',
                                                                    'reward', 'time', 'generator'])

                samples_set_df['input'] = samples_set_df[['state', 'action', 'next_state', 'inst_reward',
                                                          'reward']].values.tolist()

                samples_set_df.drop(['state', 'action', 'next_state', 'inst_reward', 'reward', 'time', 'generator'],
                                    axis=1, inplace=True)

                samples_gcn_df.append(samples_set_df['input'])

            samples_gcn_df = pd.concat(samples_gcn_df, axis=1)

        print("Updater: samples_gcn_df.shape :", samples_gcn_df.shape)
        print("Updater: Getting samples time: ", time.time()-get_samples_start_time)

        for i in range(self.dic_traffic_env_conf['NUM_AGENTS']):
            sample_set_list = samples_gcn_df.values.tolist()
            self.agents[i].prepare_Xs_Y(sample_set_list, self.dic_exp_conf)

        print("Updater: ------------------Load samples time: ", time.time()-start_time)

    def update_network(self, i):

        print('update agent %d' % i)

        self.agents[i].train_network(self.dic_exp_conf)

        if self.dic_traffic_env_conf["ONE_MODEL"]:
            self.agents[i].save_network("round_{0}".format(self.cnt_round))
        else:
            self.agents[i].save_network("round_{0}_inter_{1}".format(self.cnt_round, self.agents[i].intersection_id))

    def update_network_for_agents(self):

        if self.dic_traffic_env_conf["ONE_MODEL"]:
            self.update_network(0)
        else:
            print("update_network_for_agents", self.dic_traffic_env_conf['NUM_AGENTS'])
            for i in range(self.dic_traffic_env_conf['NUM_AGENTS']):
                self.update_network(i)
