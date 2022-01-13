from base.base_model import BaseModel
from hmm_lib.multinomial_hmm_mult_obs import MultinomialHMM

class SimpleHmmModel(BaseModel):
    def __init__(self,data_loader,config):
        obs_space = [list(data_loader.root_state_space),list(data_loader.qual_state_space)]
        self.obs_seq = data_loader.get_obs_seq()
        #list of state sequences
        self.state_seq = data_loader.get_state_seq()
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
