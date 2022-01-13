from base.base_model import BaseModel
from hmm_lib.multinomial_hmm_mult_obs import MultinomialHMM
import numpy as np

class HmmMelodyModel(BaseModel):
    def __init__(self,data_loader,config):
        obs_space = [list(data_loader.root_state_space),list(data_loader.qual_state_space),data_loader.melody_state_space,data_loader.melody_state_space,data_loader.melody_state_space,data_loader.melody_state_space]
        self.obs_seq = data_loader.get_obs_seq()
        self.state_seq = data_loader.get_state_seq() #list of state sequences
        self.seq_length = config.data.seq_length
        self.bottom_note_model = MultinomialHMM(data_loader.bn_state_space,obs_space,n_obs_seq=config.data.n_obs_seq)
        self.note_one_model = MultinomialHMM(data_loader.n1_state_space,obs_space,n_obs_seq=config.data.n_obs_seq)
        self.note_two_model = MultinomialHMM(data_loader.n2_state_space,obs_space,n_obs_seq=config.data.n_obs_seq)
        self.note_three_model = MultinomialHMM(data_loader.n3_state_space,obs_space,n_obs_seq=config.data.n_obs_seq)
        self.top_note_model = MultinomialHMM(data_loader.tn_state_space,obs_space,n_obs_seq=config.data.n_obs_seq)
        self.model_list = [self.bottom_note_model,self.note_one_model,self.note_two_model,self.note_three_model,self.top_note_model]

    def train(self):
        self.bottom_note_model.fit(self.state_seq[0],self.obs_seq)
        self.note_one_model.fit(self.state_seq[1],self.obs_seq)
        self.note_two_model.fit(self.state_seq[2],self.obs_seq)
        self.note_three_model.fit(self.state_seq[3],self.obs_seq)
        self.top_note_model.fit(self.state_seq[4],self.obs_seq)

    def calculate_fwd_bckwd_probs(self,y):
        self.bottom_note_model.approximate_joint_dist(y)
        self.note_one_model.approximate_joint_dist(y)
        self.note_two_model.approximate_joint_dist(y)
        self.note_three_model.approximate_joint_dist(y)
        self.top_note_model.approximate_joint_dist(y)

    def print_params(self):
        #print(self.bottom_note_model.alpha)
        #print(self.bottom_note_model.beta)
        print(self.top_note_model.alpha)
        print(self.top_note_model.beta)
        #print(self.bottom_note_model.psi)

    def print_PAs(self):
        #print(self.bottom_note_model.PA)
        print(self.top_note_model.PA)
