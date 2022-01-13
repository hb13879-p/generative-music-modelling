from data_loaders.simple_hmm_data_loader import SimpleHmmDataLoader
from models.simple_hmm_model import SimpleHmmModel
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
    data_loader = SimpleHmmDataLoader(config)
    print("Create the model.")
    model = SimpleHmmModel(data_loader, config)
    model.train()

    evaluator = SimpleHmmEvaluator(
        model, [[3, 8, 1, 6, 0, 5, 10], [4, 1, 3, 3, 2, 1, 5]], config
    )
    output = ReadableOutput(evaluator, data_loader)

    pickle.dump(
        model,
        open(os.path.abspath(r"evaluation/final_models/simple_hmm_model.p"), "wb"),
    )
    pickle.dump(
        data_loader,
        open(
            os.path.abspath(r"evaluation/final_data_loaders/simple_hmm_data_loader.p"),
            "wb",
        ),
    )


if __name__ == "__main__":
    main()
