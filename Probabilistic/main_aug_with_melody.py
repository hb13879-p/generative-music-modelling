from data_loaders.hmm_aug_with_melody_data_loader import HmmMelodyDataLoader
from models.hmm_with_melody_model import HmmMelodyModel
from evaluators.simple_hmm_evaluator import SimpleHmmEvaluator
from output_interpreter.readable_output import ReadableOutput
from utils.config import process_config
from utils.dirs import create_dirs
from utils.args import get_args
import pickle
import os


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
    model = HmmMelodyModel(data_loader, config)
    model.train()
    # model.print_params()
    # what is this thing:[[10,3,8,8,5,10,3,3],[2,1,5,5,2,1,3,3],[10,1,11,11,10,6,7,7],[1,3,-1,-1,0,-1,-1,-1],[-1,-1,-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1,-1,-1]]
    model.calculate_fwd_bckwd_probs(
        [
            [10, 3, 8, 8, 5, 10, 3, 3],
            [2, 1, 5, 5, 2, 1, 3, 3],
            [10, 1, 11, 11, 10, 6, 7, 7],
            [1, 3, -1, -1, 0, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
        ]
    )
    # model.print_PAs()
    evaluator = SimpleHmmEvaluator(
        model,
        [
            [10, 3, 8, 8, 5, 10, 3, 3],
            [2, 1, 5, 5, 2, 1, 3, 3],
            [10, 1, 11, 11, 10, 6, 7, 7],
            [1, 3, -1, -1, 0, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
        ],
        config,
    )
    evaluator.factorial_sample(0)
    evaluator.factorial_sample(1)
    evaluator.factorial_sample(2)
    evaluator.factorial_sample(3)
    evaluator.factorial_sample(4)
    for le in data_loader.get_label_encoders():
        print(le.classes_)
    output = ReadableOutput(evaluator, data_loader)

    pickle.dump(
        model,
        open(
            os.path.abspath(
                r"evaluation/final_models/hmm_aug_cons_transp_melody_model.p"
            ),
            "wb",
        ),
    )
    pickle.dump(
        data_loader,
        open(
            os.path.abspath(
                r"evaluation/final_data_loaders/hmm_aug_cons_tranp_melody_data_loader.p"
            ),
            "wb",
        ),
    )


if __name__ == "__main__":
    main()
