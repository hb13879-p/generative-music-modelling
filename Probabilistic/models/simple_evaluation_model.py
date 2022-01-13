from base.base_model import BaseModel
from hmm_lib.multinomial_hmm_mult_obs import MultinomialHMM
import numpy as np

class HmmMelodyModel(BaseModel):
    def __init__(self,data_loader,config):
        obs_space = [[0,1,2,3,4,5,6,7,8,9,10,11],[0,1,2,3,4,5,6,7]]
        self.obs_seq = data_loader.get_obs_train_seq()
        self.state_seq = data_loader.get_state_train_seq() #list of state sequences
        self.seq_length = config.data.seq_length
        self.bottom_note_model = MultinomialHMM(data_loader.get_state_spaces()[0],obs_space,n_obs_seq=config.data.n_obs_seq)
        self.note_one_model = MultinomialHMM(data_loader.get_state_spaces()[1],obs_space,n_obs_seq=config.data.n_obs_seq)
        self.note_two_model = MultinomialHMM(data_loader.get_state_spaces()[2],obs_space,n_obs_seq=config.data.n_obs_seq)
        self.note_three_model = MultinomialHMM(data_loader.get_state_spaces()[3],obs_space,n_obs_seq=config.data.n_obs_seq)
        self.top_note_model = MultinomialHMM(data_loader.get_state_spaces()[4],obs_space,n_obs_seq=config.data.n_obs_seq)
        self.model_list = [self.bottom_note_model,self.note_one_model,self.note_two_model,self.note_three_model,self.top_note_model]

    def train(self):
        self.bottom_note_model.fit(self.state_seq[0],self.obs_seq)
        self.note_one_model.fit(self.state_seq[1],self.obs_seq)
        self.note_two_model.fit(self.state_seq[2],self.obs_seq)
        self.note_three_model.fit(self.state_seq[3],self.obs_seq)
        self.top_note_model.fit(self.state_seq[4],self.obs_seq)

    def calculate_fwd_bckwd_probs(self):
        self.bottom_note_model.approximate_joint_dist(self.seq_length)
        self.note_one_model.approximate_joint_dist(self.seq_length)
        self.note_two_model.approximate_joint_dist(self.seq_length)
        self.note_three_model.approximate_joint_dist(self.seq_length)
        self.top_note_model.approximate_joint_dist(self.seq_length)

    def print_params(self):
        #print(self.bottom_note_model.alpha)
        #print(self.bottom_note_model.beta)
        print(self.top_note_model.alpha)
        print(self.top_note_model.beta)
        #print(self.bottom_note_model.psi)

    def print_PAs(self):
        #print(self.bottom_note_model.PA)
        print(self.top_note_model.PA)
