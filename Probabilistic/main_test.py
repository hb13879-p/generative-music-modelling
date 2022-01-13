from data_loaders.test_data_loader import HmmMelodyDataLoader
from models.test_model import HmmMelodyModel
from evaluators.hmm_evaluator_sample import SimpleHmmEvaluator
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
    model.calculate_fwd_bckwd_probs([[0, 2, 1, 2], [0, 1, 1, 1]])
    # model.print_params()
    model.print_PAs()
    evaluator = SimpleHmmEvaluator(model, [[0, 2, 1, 2], [0, 1, 1, 1]], config)
    evaluator.factorial_sample(0)
    # output = ReadableOutput(evaluator,data_loader)

    # pickle.dump( model, open( os.path.abspath(r"evaluation/final_models/hmm_melody_model.p"), "wb" ) )
    # pickle.dump( data_loader, open( os.path.abspath(r"evaluation/final_data_loaders/hmm_melody_data_loader.p"), "wb" ) )


if __name__ == "__main__":
    main()
