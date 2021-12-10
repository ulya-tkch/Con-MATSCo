"""
Creation Date : 7th Dec 2021
Last Updated : 7th Dec 2021
Author/s : Mayank Sharan

Pipeline file to run the training for CoLight
"""

# Base python imports

import os
import json
import time
import random
import shutil
import xml.etree.ElementTree as Elt

from multiprocessing import Process

# Import from modules

import colight.model_test as model_test

from colight.updater import Updater
from colight.generator import Generator
from colight.construct_sample import ConstructSample


class Pipeline:

    def __init__(self, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):

        # load configurations
        self.dic_exp_conf = dic_exp_conf
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path

        # do file operations
        self._path_check()
        self._copy_conf_file()

        if self.dic_traffic_env_conf["SIMULATOR_TYPE"] == 'anon':
            self._copy_anon_file()
        else:
            print("Error", self.dic_traffic_env_conf["SIMULATOR_TYPE"], "Not handled")

        # test_duration

        self.test_duration = []

        sample_num = min(self.dic_traffic_env_conf["NUM_INTERSECTIONS"], 10)

        print("sample_num for early stopping:", sample_num)
        self.sample_inter_id = random.sample(range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]), sample_num)

    @staticmethod
    def _set_traffic_file(sumo_config_file_tmp_name, sumo_config_file_output_name, list_traffic_file_name):

        # update sumocfg
        sumo_cfg = Elt.parse(sumo_config_file_tmp_name)
        config_node = sumo_cfg.getroot()
        input_node = config_node.find("input")
        for route_files in input_node.findall("route-files"):
            input_node.remove(route_files)
        input_node.append(
            Elt.Element("route-files", attrib={"value": ",".join(list_traffic_file_name)}))
        sumo_cfg.write(sumo_config_file_output_name)

    def _path_check(self):
        # check path
        if os.path.exists(self.dic_path["PATH_TO_WORK_DIRECTORY"]):
            if self.dic_path["PATH_TO_WORK_DIRECTORY"] != "records/default":
                raise FileExistsError
            else:
                pass
        else:
            os.makedirs(self.dic_path["PATH_TO_WORK_DIRECTORY"])

        if os.path.exists(self.dic_path["PATH_TO_MODEL"]):
            if self.dic_path["PATH_TO_MODEL"] != "model/default":
                raise FileExistsError
            else:
                pass
        else:
            os.makedirs(self.dic_path["PATH_TO_MODEL"])

        if os.path.exists(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"]):
            pass
        else:
            os.makedirs(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"])

        if os.path.exists(self.dic_path["PATH_TO_PRETRAIN_MODEL"]):
            pass
        else:
            os.makedirs(self.dic_path["PATH_TO_PRETRAIN_MODEL"])

    def _copy_conf_file(self, path=None):
        # write conf files

        if path is None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        json.dump(self.dic_exp_conf, open(os.path.join(path, "exp.conf"), "w"),
                  indent=4)
        json.dump(self.dic_agent_conf, open(os.path.join(path, "agent.conf"), "w"),
                  indent=4)
        json.dump(self.dic_traffic_env_conf,
                  open(os.path.join(path, "traffic_env.conf"), "w"), indent=4)

    def _copy_anon_file(self, path=None):
        # hard code !!!
        if path is None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]

        # copy sumo files

        shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], self.dic_exp_conf["TRAFFIC_FILE"][0]),
                    os.path.join(path, self.dic_exp_conf["TRAFFIC_FILE"][0]))
        shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], self.dic_exp_conf["ROADNET_FILE"]),
                    os.path.join(path, self.dic_exp_conf["ROADNET_FILE"]))

    def generator_wrapper(self, cnt_round, cnt_gen, dic_path, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                          best_round=None):

        generator = Generator(cnt_round=cnt_round,
                              cnt_gen=cnt_gen,
                              dic_path=dic_path,
                              dic_exp_conf=dic_exp_conf,
                              dic_agent_conf=dic_agent_conf,
                              dic_traffic_env_conf=dic_traffic_env_conf,
                              best_round=best_round
                              )

        print("make generator")
        generator.generate()
        print("generator_wrapper end")

        return

    def updater_wrapper(self, cnt_round, dic_agent_conf, dic_exp_conf, dic_traffic_env_conf, dic_path, best_round=None,
                        bar_round=None):

        updater = Updater(
            cnt_round=cnt_round,
            dic_agent_conf=dic_agent_conf,
            dic_exp_conf=dic_exp_conf,
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=dic_path,
            best_round=best_round,
            bar_round=bar_round
        )

        updater.load_sample_for_agents()
        updater.update_network_for_agents()
        print("updater_wrapper end")
        return

    def run(self, multi_process=False):

        best_round, bar_round = None, None

        print("Printing for pipeline the min action time", self.dic_traffic_env_conf)

        f_time = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "running_time.csv"), "w")
        f_time.write("generator_time\tmaking_samples_time\tupdate_network_time\ttest_evaluation_times\tall_times\n")
        f_time.close()

        # trainf

        for cnt_round in range(self.dic_exp_conf["NUM_ROUNDS"]):
            print("-----------------------------Round %d starts-----------------------------" % cnt_round)
            round_start_time = time.time()

            process_list = []

            print("==============  generator =============")
            generator_start_time = time.time()
            if multi_process:

                for cnt_gen in range(self.dic_exp_conf["NUM_GENERATORS"]):
                    p = Process(target=self.generator_wrapper,
                                args=(cnt_round, cnt_gen, self.dic_path, self.dic_exp_conf,
                                      self.dic_agent_conf, self.dic_traffic_env_conf, best_round))
                    p.start()
                    process_list.append(p)
                for i in range(len(process_list)):
                    p = process_list[i]
                    p.join()
                print("Generator join complete")

            else:

                for cnt_gen in range(self.dic_exp_conf["NUM_GENERATORS"]):
                    self.generator_wrapper(cnt_round=cnt_round,
                                           cnt_gen=cnt_gen,
                                           dic_path=self.dic_path,
                                           dic_exp_conf=self.dic_exp_conf,
                                           dic_agent_conf=self.dic_agent_conf,
                                           dic_traffic_env_conf=self.dic_traffic_env_conf,
                                           best_round=best_round)

            generator_end_time = time.time()
            generator_total_time = generator_end_time - generator_start_time

            print("==============  make samples =============")

            # make samples and determine which samples are good
            making_samples_start_time = time.time()

            train_round = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round")
            if not os.path.exists(train_round):
                os.makedirs(train_round)

            cs = ConstructSample(path_to_samples=train_round, cnt_round=cnt_round,
                                 dic_traffic_env_conf=self.dic_traffic_env_conf)
            cs.make_reward_for_system()

            making_samples_end_time = time.time()
            making_samples_total_time = making_samples_end_time - making_samples_start_time

            print("==============  update network =============")
            update_network_start_time = time.time()

            if self.dic_exp_conf["MODEL_NAME"] in self.dic_exp_conf["LIST_MODEL_NEED_TO_UPDATE"]:

                if multi_process:
                    p = Process(target=self.updater_wrapper,
                                args=(cnt_round,
                                      self.dic_agent_conf,
                                      self.dic_exp_conf,
                                      self.dic_traffic_env_conf,
                                      self.dic_path,
                                      best_round,
                                      bar_round))
                    p.start()
                    p.join()
                    print("Updater join complete")
                else:
                    self.updater_wrapper(cnt_round=cnt_round,
                                         dic_agent_conf=self.dic_agent_conf,
                                         dic_exp_conf=self.dic_exp_conf,
                                         dic_traffic_env_conf=self.dic_traffic_env_conf,
                                         dic_path=self.dic_path,
                                         best_round=best_round,
                                         bar_round=bar_round)

            update_network_end_time = time.time()
            update_network_total_time = update_network_end_time - update_network_start_time

            print("==============  test evaluation =============")

            test_evaluation_start_time = time.time()

            if multi_process:

                p = Process(target=model_test.test,
                            args=(self.dic_path["PATH_TO_MODEL"], cnt_round,
                                  self.dic_exp_conf["RUN_COUNTS"], self.dic_traffic_env_conf, False))
                p.start()
                p.join()

            else:
                model_test.test(self.dic_path["PATH_TO_MODEL"], cnt_round, self.dic_exp_conf["RUN_COUNTS"],
                                self.dic_traffic_env_conf, if_gui=False)

            test_evaluation_end_time = time.time()
            test_evaluation_total_time = test_evaluation_end_time - test_evaluation_start_time

            print("best_round: ", best_round)

            print("Generator time: ", generator_total_time)
            print("Making samples time:", making_samples_total_time)
            print("update_network time:", update_network_total_time)
            print("test_evaluation time:", test_evaluation_total_time)

            print("round {0} ends, total_time: {1}".format(cnt_round, time.time()-round_start_time))
            f_time = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "running_time.csv"), "a")
            f_time.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(generator_total_time, making_samples_total_time,
                                                            update_network_total_time, test_evaluation_total_time,
                                                            time.time()-round_start_time))
            f_time.close()
