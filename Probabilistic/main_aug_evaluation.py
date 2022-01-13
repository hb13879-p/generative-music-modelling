from data_loaders.evaluation_data_loader import HmmMelodyDataLoader
from models.evaluation_model import HmmMelodyModel
from evaluators.simple_hmm_evaluator import SimpleHmmEvaluator
from output_interpreter.readable_output import ReadableOutput
from utils.config import process_config
from utils.dirs import create_dirs
from utils.args import get_args
import pickle
import os
import numpy as np


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    print("Create the data generator.")
    data_loader = HmmMelodyDataLoader(config)
    print("Create the model.")
    # model = HmmMelodyModel(data_loader,config)
    # model.train()
    # pickle.dump( model, open( os.path.abspath(r"evaluation/final_models/hmm_aug_training_set_model.p"), "wb" ) )
    # pickle.dump( data_loader, open( os.path.abspath(r"evaluation/final_data_loaders/hmm_aug_training_set_data_loader.p"), "wb" ) )
    model = pickle.load(
        open(
            os.path.abspath(
                r"C:\Users\User\Documents\Project\Models/Probabilistic/evaluation/final_models/hmm_aug_training_set_model.p"
            ),
            "rb",
        )
    )
    data_loader = pickle.load(
        open(
            os.path.abspath(
                r"C:\Users\User\Documents\Project\Models/Probabilistic/evaluation/final_data_loaders/hmm_aug_training_set_data_loader.p"
            ),
            "rb",
        )
    )
    obs_test = data_loader.get_obs_test_seq()
    state_seq_test = data_loader.get_state_test_seq()
    count = 0
    total = 0
    for obs_list, test_data in zip(
        list(
            zip(
                obs_test[0],
                obs_test[1],
                obs_test[2],
                obs_test[3],
                obs_test[4],
                obs_test[5],
            )
        )[0:-1],
        list(
            zip(
                state_seq_test[0],
                state_seq_test[1],
                state_seq_test[2],
                state_seq_test[3],
                state_seq_test[4],
            )
        )[0:-1],
    ):
        evaluator = SimpleHmmEvaluator(model, obs_list, config)
        paths = evaluator.get_paths()
        for prediction, label in zip(paths, test_data):
            count += np.sum(prediction == label)
            total += len(prediction)
    print("accuracy = {}".format(str(count / total)))
    """
    #model.print_params()
    model.calculate_fwd_bckwd_probs()
    #model.print_PAs()
    evaluator = SimpleHmmEvaluator(model,[[3,8,1,6,0,5,10],[4,1,3,3,2,1,5],[-1,-1,-1,-1,-1,-1,-1],[-1,-1,3,5,-1,11,-1],[0,0,0,0,4,4,5],[2,3,-1,-1,-1,2,9]],config)
    evaluator.factorial_sample(0)
    output = ReadableOutput(evaluator,data_loader)

"""


if __name__ == "__main__":
    main()
