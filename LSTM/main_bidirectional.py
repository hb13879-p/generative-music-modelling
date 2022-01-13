from data_loader.simple_LSTM_data_loader import SimpleLSTMDataLoader
from models.bidirectional_simple_LSTM_model import SimpleLSTMModel
from trainers.simple_LSTM_trainer import SimpleLSTMModelTrainer
from evaluators.simple_LSTM_evaluator import SimpleLSTMEvaluator
from utils.config import process_config
from utils.dirs import create_dirs
from utils.args import get_args
from utils.read_activations import get_activations
from output_interpreter.readable_output import ReadableOutput
from output_interpreter.music_xml_writer import MusicXmlWriter
from output_interpreter.writable_output import WritableOutput

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

    print('Create the data generator.')
    data_loader = SimpleLSTMDataLoader(config)


    print('Create the model.')
    model = SimpleLSTMModel(config)
    print('Create the trainer')
    trainer = SimpleLSTMModelTrainer(model.model, data_loader.get_train_data(), config)

    print('Start training the model.')
    trainer.train()

    print('Evaluating on test set')
    evaluator = SimpleLSTMEvaluator(model.model,data_loader.get_test_data())
    print("test set loss and accuaracy: {} {}".format(evaluator.loss,evaluator.acc))

    activations = get_activations(model.model, 0,data_loader.get_test_data()[0][[0],:,:])

    print('Convert output to readable list')
    readable_output = ReadableOutput(evaluator.sample_result,data_loader.get_test_data()[0],config.evaluator.test_sample_no)

    music_xml_writer = MusicXmlWriter(readable_output.voicings,readable_output.labels)
    output_score_filename = "output_scores/" + config.exp.name + "_output_score.xml"
    music_xml_writer.write(output_score_filename)

    writable_output = WritableOutput(config,trainer,evaluator)
    writable_output.write("model_selection.csv")

    trainer.model.save("evaluation/final_models/bidirectional_model.h5")

if __name__ == '__main__':
    main()
