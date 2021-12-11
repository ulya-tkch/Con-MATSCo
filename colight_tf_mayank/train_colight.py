"""
Creation Date : 5th Dec 2021
Last Updated : 5th Dec 2021
Author/s : Mayank Sharan

Main file to train the CoLight agent
"""

# Base python imports

import os
import copy
import time
import config
import argparse

from multiprocessing import Process

# Import from modules

from colight.pipeline import Pipeline

# Initialize global variables

NUM_ROUNDS = 50
TOP_K_ADJACENCY = -1
TOP_K_ADJACENCY_LANE = -1

NEIGHBOR = False
PRETRAIN = False
SAVEREPLAY = True
EARLY_STOP = False
multi_process = True
hangzhou_archive = True
ADJACENCY_BY_CONNECTION_OR_GEO = False

ANON_PHASE_REPRE = []


def parse_args_exp():

    parser = argparse.ArgumentParser()

    # The file folder to create/log in
    parser.add_argument("--memo", type=str, default='1210_switch_constraint_ny_Colight_6_6_bi')  # 1_3,2_2,3_3,4_4
    parser.add_argument("--env", type=int, default=1)  # env=1 means you will run CityFlow
    parser.add_argument("--gui", type=bool, default=False)
    # parser.add_argument("--road_net", type=str, default='6_6')  # '1_2') # which road net you are going to run
    # parser.add_argument("--volume", type=str, default='300')  # '300'
    # parser.add_argument("--suffix", type=str, default="0.3_bi")  # 0.3

    parser.add_argument("--road_net", type=str, default='16_3')  # '1_2') # which road net you are going to run
    parser.add_argument("--volume", type=str, default='newyork')  # '300'
    parser.add_argument("--suffix", type=str, default="real")  # 0.3


    global hangzhou_archive
    hangzhou_archive = False

    global TOP_K_ADJACENCY
    TOP_K_ADJACENCY = 5

    global TOP_K_ADJACENCY_LANE
    TOP_K_ADJACENCY_LANE = 5

    global NUM_ROUNDS
    NUM_ROUNDS = 50

    global EARLY_STOP
    EARLY_STOP = False

    global NEIGHBOR
    # TAKE CARE
    NEIGHBOR = False

    global SAVEREPLAY  # if you want to relay your simulation, set it to be True
    SAVEREPLAY = True

    global ADJACENCY_BY_CONNECTION_OR_GEO
    # TAKE CARE
    # Consider setting this to True
    ADJACENCY_BY_CONNECTION_OR_GEO = False

    # modify:TOP_K_ADJACENCY in line 154

    global PRETRAIN
    PRETRAIN = False

    parser.add_argument("--mod", type=str, default='CoLight')  # SimpleDQN,SimpleDQNOne,GCN,CoLight,Lit
    parser.add_argument("--cnt", type=int, default=3600)  # 3600
    parser.add_argument("--gen", type=int, default=1)  # 4

    parser.add_argument("-all", action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--onemodel", type=bool, default=False)

    parser.add_argument("--visible_gpu", type=str, default="-1")

    global ANON_PHASE_REPRE

    tt = parser.parse_args()

    if 'CoLight_Signal' in tt.mod:
        # 12dim
        ANON_PHASE_REPRE = {
            1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],  # 'WSES',
            2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],  # 'NSSS',
            3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],  # 'WLEL',
            4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1],  # 'NLSL',
        }
    else:
        # 8 dim
        ANON_PHASE_REPRE = {
            1: [0, 1, 0, 1, 0, 0, 0, 0],
            2: [0, 0, 0, 0, 0, 1, 0, 1],
            3: [1, 0, 1, 0, 0, 0, 0, 0],
            4: [0, 0, 0, 0, 1, 0, 1, 0],
        }

    print('agent_name:%s', tt.mod)
    print('ANON_PHASE_REPRE:', ANON_PHASE_REPRE)

    print("-----------------------------Argument Parse Complete-----------------------------")

    return parser.parse_args()


def merge(dic_tmp, dic_to_change):

    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)

    return dic_result


def pipeline_wrapper(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):

    ppl = Pipeline(dic_exp_conf=dic_exp_conf,  # experiment config
                   dic_agent_conf=dic_agent_conf,  # RL agent config
                   dic_traffic_env_conf=dic_traffic_env_conf,  # the simulation configuration
                   dic_path=dic_path  # where should I save the logs?
                   )

    global multi_process
    ppl.run(multi_process=multi_process)

    print("pipeline_wrapper end")
    return


def main_exp(memo, env, road_net, gui, volume, suffix, mod, cnt, gen, r_all, workers, onemodel):

    # Get number of intersections in the network and the environment

    NUM_COL = int(road_net.split('_')[0])
    NUM_ROW = int(road_net.split('_')[1])
    num_intersections = NUM_ROW * NUM_COL
    print('num_intersections:', num_intersections)

    ENVIRONMENT = ["sumo", "anon"][env]

    traffic_file_list = ["{0}_{1}_{2}_{3}.json".format(ENVIRONMENT, road_net, volume, suffix)]

    process_list = []
    n_workers = workers

    global PRETRAIN
    global NUM_ROUNDS
    global EARLY_STOP
    global multi_process

    global NEIGHBOR
    global SAVEREPLAY
    global TOP_K_ADJACENCY
    global ANON_PHASE_REPRE
    global TOP_K_ADJACENCY_LANE
    global ADJACENCY_BY_CONNECTION_OR_GEO

    for traffic_file in traffic_file_list:

        dic_exp_conf_extra = {

            "RUN_COUNTS": cnt,
            "MODEL_NAME": mod,
            "TRAFFIC_FILE": [traffic_file],  # here: change to multi_traffic

            "ROADNET_FILE": "roadnet_{0}.json".format(road_net),

            "NUM_ROUNDS": NUM_ROUNDS,
            "NUM_GENERATORS": gen,

            "MODEL_POOL": False,
            "NUM_BEST_MODEL": 3,

            "PRETRAIN": PRETRAIN,
            "PRETRAIN_MODEL_NAME": mod,
            "PRETRAIN_NUM_ROUNDS": 0,
            "PRETRAIN_NUM_GENERATORS": 15,

            "AGGREGATE": False,
            "DEBUG": False,
            "EARLY_STOP": EARLY_STOP,
        }

        dic_agent_conf_extra = {
            "EPOCHS": 100,
            "SAMPLE_SIZE": 1000,
            "MAX_MEMORY_LEN": 10000,
            "UPDATE_Q_BAR_EVERY_C_ROUND": False,
            "UPDATE_Q_BAR_FREQ": 5,

            # network

            "N_LAYER": 2,
            "TRAFFIC_FILE": traffic_file,
        }

        dic_traffic_env_conf_extra = {
            "USE_LANE_ADJACENCY": True,
            "ONE_MODEL": onemodel,
            "NUM_AGENTS": num_intersections,
            "NUM_INTERSECTIONS": num_intersections,
            "ACTION_PATTERN": "set",
            "MEASURE_TIME": 1,
            "IF_GUI": gui,
            "DEBUG": False,
            "TOP_K_ADJACENCY": TOP_K_ADJACENCY,
            "ADJACENCY_BY_CONNECTION_OR_GEO": ADJACENCY_BY_CONNECTION_OR_GEO,
            "TOP_K_ADJACENCY_LANE": TOP_K_ADJACENCY_LANE,
            "SIMULATOR_TYPE": ENVIRONMENT,
            "BINARY_PHASE_EXPANSION": True,
            "FAST_COMPUTE": True,

            "NEIGHBOR": NEIGHBOR,
            "MODEL_NAME": mod,



            "SAVEREPLAY": SAVEREPLAY,
            "NUM_ROW": NUM_ROW,
            "NUM_COL": NUM_COL,

            "TRAFFIC_FILE": traffic_file,
            "VOLUME": volume,
            "ROADNET_FILE": "roadnet_{0}.json".format(road_net),

            "phase_expansion": {
                1: [0, 1, 0, 1, 0, 0, 0, 0],
                2: [0, 0, 0, 0, 0, 1, 0, 1],
                3: [1, 0, 1, 0, 0, 0, 0, 0],
                4: [0, 0, 0, 0, 1, 0, 1, 0],
                5: [1, 1, 0, 0, 0, 0, 0, 0],
                6: [0, 0, 1, 1, 0, 0, 0, 0],
                7: [0, 0, 0, 0, 0, 0, 1, 1],
                8: [0, 0, 0, 0, 1, 1, 0, 0]
            },

            "phase_expansion_4_lane": {
                1: [1, 1, 0, 0],
                2: [0, 0, 1, 1],
            },

            "LIST_STATE_FEATURE": ["cur_phase", "lane_num_vehicle"],
            "DIC_FEATURE_DIM": dict(
                D_LANE_QUEUE_LENGTH=(4,),
                D_LANE_NUM_VEHICLE=(4,),

                D_COMING_VEHICLE=(12,),
                D_LEAVING_VEHICLE=(12,),

                D_LANE_NUM_VEHICLE_BEEN_STOPPED_THRES1=(4,),
                D_CUR_PHASE=(1,),
                D_NEXT_PHASE=(1,),
                D_TIME_THIS_PHASE=(1,),
                D_TERMINAL=(1,),
                D_LANE_SUM_WAITING_TIME=(4,),
                D_VEHICLE_POSITION_IMG=(4, 60,),
                D_VEHICLE_SPEED_IMG=(4, 60,),
                D_VEHICLE_WAITING_TIME_IMG=(4, 60,),

                D_PRESSURE=(1,),

                D_ADJACENCY_MATRIX=(2,),

                D_ADJACENCY_MATRIX_LANE=(6,),

            ),
            "DIC_REWARD_INFO": {
                "flickering": 0,  # -5,#
                "sum_lane_queue_length": 0,
                "sum_lane_wait_time": 0,
                "sum_lane_num_vehicle_left": 0,  # -1,#
                "sum_duration_vehicle_left": 0,
                "sum_num_vehicle_been_stopped_thres01": 0,
                "sum_num_vehicle_been_stopped_thres1": -0.25,
                "pressure": 0  # -0.25
            },
            "LANE_NUM": {
                "LEFT": 1,
                "RIGHT": 1,
                "STRAIGHT": 1
            },
            "PHASE": {
                "anon": ANON_PHASE_REPRE,
            }
        }

        if volume == 'newyork':
            template = 'NewYork'
        elif dic_traffic_env_conf_extra["LANE_NUM"] == config._LSR:
            template = "template_lsr"
        else:
            template = ""

        if mod in ['CoLight']:

            dic_traffic_env_conf_extra["NUM_AGENTS"] = 1
            dic_traffic_env_conf_extra['ONE_MODEL'] = False

            if "adjacency_matrix" not in dic_traffic_env_conf_extra['LIST_STATE_FEATURE'] and \
                    "adjacency_matrix_lane" not in dic_traffic_env_conf_extra['LIST_STATE_FEATURE'] and \
                    mod not in ['SimpleDQNOne']:

                dic_traffic_env_conf_extra['LIST_STATE_FEATURE'].append("adjacency_matrix")
                dic_traffic_env_conf_extra['LIST_STATE_FEATURE'].append("adjacency_matrix_lane")

                if dic_traffic_env_conf_extra['ADJACENCY_BY_CONNECTION_OR_GEO']:
                    TOP_K_ADJACENCY = 5
                    dic_traffic_env_conf_extra['LIST_STATE_FEATURE'].append("connectivity")
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CONNECTIVITY'] = (5,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_ADJACENCY_MATRIX'] = (5,)
                else:
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_ADJACENCY_MATRIX'] = \
                        (dic_traffic_env_conf_extra['TOP_K_ADJACENCY'],)

                if dic_traffic_env_conf_extra['USE_LANE_ADJACENCY']:
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_ADJACENCY_MATRIX_LANE'] = \
                        (dic_traffic_env_conf_extra['TOP_K_ADJACENCY_LANE'],)
        else:
            dic_traffic_env_conf_extra["NUM_AGENTS"] = dic_traffic_env_conf_extra["NUM_INTERSECTIONS"]

        if dic_traffic_env_conf_extra['BINARY_PHASE_EXPANSION']:
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE'] = (8,)

            if dic_traffic_env_conf_extra['NEIGHBOR']:
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_0'] = (8,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_0'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_1'] = (8,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_1'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_2'] = (8,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_2'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_3'] = (8,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_3'] = (4,)

            else:

                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_0'] = (1,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_0'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_1'] = (1,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_1'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_2'] = (1,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_2'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_3'] = (1,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_3'] = (4,)

        print("Processing traffic file:", traffic_file)

        prefix_intersections = str(road_net)

        dic_path_extra = {
            "PATH_TO_MODEL": os.path.join("model", memo, traffic_file + "_" +
                                          time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
            "PATH_TO_WORK_DIRECTORY": os.path.join("records", memo, traffic_file + "_" +
                                                   time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
            "PATH_TO_DATA": os.path.join("data", template, prefix_intersections),
            "PATH_TO_PRETRAIN_MODEL": os.path.join("model", "initial", traffic_file),
            "PATH_TO_PRETRAIN_WORK_DIRECTORY": os.path.join("records", "initial", traffic_file),
            "PATH_TO_ERROR": os.path.join("errors", memo)
        }

        deploy_dic_exp_conf = merge(config.DIC_EXP_CONF, dic_exp_conf_extra)
        deploy_dic_agent_conf = merge(getattr(config, "DIC_{0}_AGENT_CONF".format(mod.upper())),
                                      dic_agent_conf_extra)
        deploy_dic_traffic_env_conf = merge(config.dic_traffic_env_conf, dic_traffic_env_conf_extra)
        deploy_dic_path = merge(config.DIC_PATH, dic_path_extra)

        if multi_process:
            ppl = Process(target=pipeline_wrapper,
                          args=(deploy_dic_exp_conf,
                                deploy_dic_agent_conf,
                                deploy_dic_traffic_env_conf,
                                deploy_dic_path))
            process_list.append(ppl)
        else:
            pipeline_wrapper(dic_exp_conf=deploy_dic_exp_conf,
                             dic_agent_conf=deploy_dic_agent_conf,
                             dic_traffic_env_conf=deploy_dic_traffic_env_conf,
                             dic_path=deploy_dic_path)

    if multi_process:
        for i in range(0, len(process_list), n_workers):
            i_max = min(len(process_list), i + n_workers)
            for j in range(i, i_max):
                print(j)
                print("start_traffic")
                process_list[j].start()
                print("after_traffic")
            for k in range(i, i_max):
                print("traffic to join", k)
                process_list[k].join()
                print("traffic finish join", k)


if __name__ == "__main__":

    args = parse_args_exp()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu

    main_exp(args.memo, args.env, args.road_net, args.gui, args.volume, args.suffix, args.mod, args.cnt,
             args.gen, args.all, args.workers, args.onemodel)
