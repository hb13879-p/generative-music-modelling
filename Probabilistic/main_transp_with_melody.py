from data_loaders.hmm_transp_with_melody_data_loader import HmmMelodyDataLoader
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
    model.calculate_fwd_bckwd_probs()
    # model.print_PAs()
    evaluator = SimpleHmmEvaluator(
        model,
        [
            [3, 8, 1, 6, 0, 5, 10],
            [4, 1, 3, 3, 2, 1, 5],
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, 3, 5, -1, 11, -1],
            [0, 0, 0, 0, 4, 4, 5],
            [2, 3, -1, -1, -1, 2, 9],
        ],
        config,
    )
    evaluator.factorial_sample(0)
    evaluator.factorial_sample(1)
    evaluator.factorial_sample(2)
    evaluator.factorial_sample(3)
    evaluator.factorial_sample(4)
    les = data_loader.get_label_encoders()
    for le in les:
        print(le.classes_)
    output = ReadableOutput(evaluator, data_loader)

    pickle.dump(
        model,
        open(
            os.path.abspath(r"evaluation/final_models/hmm_transp_melody_model.p"), "wb"
        ),
    )
    pickle.dump(
        data_loader,
        open(
            os.path.abspath(
                r"evaluation/final_data_loaders/hmm_tranp_melody_data_loader.p"
            ),
            "wb",
        ),
    )


if __name__ == "__main__":
    main()
