import os
import copy

from datetime import datetime

from runexp import main_exp
from runexp import parse_args_exp

from run_baseline import main_base
from run_baseline import parse_args_base

from summary_multi_anon import parse_args_sum
from summary_multi_anon import summary_detail_test
from summary_multi_anon import summary_detail_baseline

if __name__ == '__main__':

    # Populate the params used to uniquely store the results, logs etc.

    date = '01'  # Expected to be a length 2 string which is the date of run
    month = '12'  # Expected to be a length 2 string which is the month of run
    hour = '13'  # Expected to be a length 2 string which is the hour of run
    ident = 'test_rewardshaping_2'  # Expected to be a string of your choosing that can ensure uniqueness

    # Initialize control variables related to model training

    # 'CoLight' (run using runexp.py)
    # 'Fixedtime', 'MaxPressure' (run using run_baseline.py)

    model_name = 'MaxPressure'

    # These are also dimensions of the intersections grid
    # Synthetic Data : '3_3', '6_6', '10_10'
    # NewYork : '16_3', '28_7'

    road_net = '6_6'

    # Synthetic Data : '300'
    # NewYork : 'newyork'

    volume = '300'

    # Synthetic Data : '0.3_uni' (unidirectional traffic), '0.3_bi' (bidirectional traffic)
    # NewYork : 'real_triple' / 'real_double' (corresponding to '28_7'), 'real' (corr to '16_3')

    ratio = '0.3_bi'

    # Initialize flags to direct run flow

    run_train = True
    run_eval = True

    # Prepare a few variables

    if '.' in ratio:
        rand_suff = ratio.split('_')[-1]
    else:
        rand_suff = ratio

    memo = month + date + '_' + hour + '_' + model_name + '_' + road_net + '_' + rand_suff + '_' + ident

    t_before_train = datetime.now()

    if run_train:

        # Run Training

        if model_name in ['CoLight']:

            # Running using runexp.py

            run_args = parse_args_exp()
            os.environ["CUDA_VISIBLE_DEVICES"] = run_args.visible_gpu

            # Set variables

            run_args.memo = memo
            run_args.road_net = road_net
            run_args.volume = volume
            run_args.suffix = ratio
            run_args.mod = model_name

            main_exp(run_args.memo, run_args.env, run_args.road_net, run_args.gui, run_args.volume,
                     run_args.suffix, run_args.mod, run_args.cnt, run_args.gen, run_args.all, run_args.workers,
                     run_args.onemodel)

        elif model_name in ['Fixedtime', 'MaxPressure']:

            # Running using run_baseline.py

            run_args = parse_args_base()
            print(run_args)

            # Set variables

            run_args.memo = memo
            run_args.road_net = road_net
            run_args.volume = volume
            run_args.ratio = ratio
            run_args.model = model_name

            print(run_args)

            main_base(run_args.memo, run_args.all, run_args.road_net, run_args.env, run_args.gui,
                      run_args.volume, run_args.ratio, run_args.model, run_args.count, run_args.lane)

        print("Training took", datetime.now() - t_before_train)

    else:
        print("Training skipped")

    # Run evaluation

    t_before_eval = datetime.now()

    if run_eval:

        if not os.path.exists('records/' + memo):
            print("Trained model not present. No evaluation possible")
        else:

            total_summary = {
                "traffic": [],
                "inter_num": [],
                "traffic_volume": [],
                "ratio": [],
                "min_queue_length": [],
                "min_queue_length_round": [],
                "min_duration": [],
                "min_duration_round": [],
                "final_duration": [],
                "final_duration_std": [],
                "convergence_1.2": [],
                "convergence_1.1": [],
                "nan_count": [],
                "min_duration2": []
            }

            if model_name in ['Fixedtime', 'MaxPressure']:
                summary_detail_baseline(memo)
            else:
                summary_detail_test(memo, copy.deepcopy(total_summary))

        print("Evaluation took", datetime.now() - t_before_eval)

    else:
        print("Evaluation skipped")
