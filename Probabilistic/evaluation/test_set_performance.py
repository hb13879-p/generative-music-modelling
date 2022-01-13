import pickle
import os
import sys
import os.path
pathname = r"C:\Users\User\Documents\Project\Dataset"
sys.path.insert(0, os.path.normpath(pathname))
pathname2 = r"C:\Users\User\Documents\Project\Models\Probabilistic"
sys.path.insert(0, os.path.normpath(pathname2))



def main():

    model = pickle.load(open(os.path.abspath(r"C:\Users\User\Documents\Project\Models/Probabilistic/evaluation/final_models/hmm_melody_training_set_model.p"),"rb"))
    data_loader = pickle.load(open(os.path.abspath(r"C:\Users\User\Documents\Project\Models/Probabilistic/evaluation/final_data_loaders/hmm_melody_training_set_data_loader.p"),"rb"))
    obs_test = data_loader.get_obs_test_seq()
    state_seq_test = data_loader.get_state_test_seq()
    print(obs_test[0])
    #for seq in obs_test:
    #    evaluator = SimpleHmmEvaluator(model,,config)

if(__name__ == '__main__'):
    main()


#seq_length = config.model_data.seq_length
#inc_melody = config.model_data.inc_melody
#output_score_filename = config.model_data.output_score_filename
#model_filename = config.model_data.model_filename
#data_loader_filename = config.model_data.data_loader_filename
#test_filename = r"C:\Users\User\Documents\Project\Test_Scores/" + config.model_data.test_filename




#model = pickle.load(open(model_filename, "rb" ))
#data_loader = pickle.load(open(data_loader_filename, "rb" ))
#evaluator = SimpleHmmEvaluator(model,X,config)
#paths = evaluator.get_paths()
#readable_output = ReadableOutput(evaluator,data_loader)
