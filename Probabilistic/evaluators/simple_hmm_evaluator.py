from hmm_lib.multinomial_hmm_mult_obs import MultinomialHMM
import numpy as np
from functools import reduce


class SimpleHmmEvaluator(object):
    def __init__(self,model,obs_seq,config):
        self.model = model
        self.model_list = model.model_list
        self.config = config
        self.__viterbi(obs_seq)
        self.final_dist = []

    def __viterbi(self,obs_seq):
        print(obs_seq)
        self.bn_path, self.bn_prob = self.model.bottom_note_model.viterbi(obs_seq)
        self.n1_path, self.n1_prob = self.model.note_one_model.viterbi(obs_seq)
        self.n2_path, self.n2_prob = self.model.note_two_model.viterbi(obs_seq)
        self.n3_path, self.n3_prob = self.model.note_three_model.viterbi(obs_seq)
        self.tn_path, self.tn_prob = self.model.top_note_model.viterbi(obs_seq)

    def get_paths(self):
        return [self.bn_path,self.n1_path,self.n2_path,self.n3_path,self.tn_path]

    def factorial_sample(self,focus_index): #eq2
        self.__get_probabilities()
        sum_terms = [P*psi for P,psi in zip(self.probs,self.psis)]
        #extract focus index
        sum_lhs = sum_terms[focus_index]
        sum_rhs_terms = [x for i,x in enumerate(sum_terms) if i != focus_index]
        sum_rhs = [np.sum(x,axis=1,keepdims=True) for x in sum_rhs_terms]
        multiplied_list = reduce(lambda x,y:x*y,sum_rhs)
        result = sum_lhs * multiplied_list
        #normalise
        row_sums = result.sum(axis=1)
        normalised_result = result / row_sums[:, np.newaxis]
        print(normalised_result)
        np.savetxt("output_dist_mod{}.csv".format(focus_index), normalised_result, delimiter=",")
        '''
        sum_terms = [] #dimension (models-1,T,1)
        psia_times_PA = [] #dimension(T,state_space)
        #go through the list of 5 PAs and 5 psis

        #self.probs = P(A) for all 5 models, likewise for psis
        sumlist = [[P,psi] for P,psi in zip(self.probs,self.psis)]
        #print(sumlist)
        sumlist = [np.sum(P[0]*P[1],axis=1,keepdims=True) for P in sumlist] #model 1 P * model 1 psi, model2 P * model 2 psi
        #want to exclude focus_index
        sumlist_rhs = [x for i,x in enumerate(sumlist) if i!=focus_index] #dimension (models-1,T,state_space)
        sumlist_lhs = [P*psi for P,psi in zip(self.ordered_probs[n],self.ordered_psis[n])][focus_index] #dimension (1,T,state_space) == P(A) * psi_a for obs seq n
        sum_terms.append(sumlist_rhs)
        psia_times_PA_for_obs_seq_n.append(sumlist_lhs)
        #multiply together summation terms for all models bar focus:
        multiplied_sums = [] #will be dimension (obs_seq,T,1)
        result_list = [] #will be dimension (obs_seq,T,state_space_of_focus_model)
        for n in range(self.config.data.n_obs_seq):
            #get one list for each obs_seq, each list of dimension [T,1]
            list_to_mul = sum_terms[n]
            multiplied_list = reduce(lambda x,y:x*y,list_to_mul)
            multiplied_sums.append(multiplied_list)
            #multiply first term (PA * psi of dim (obs_seq,T,state_space)) with second (multiplied_sums dim (obs_seq,T,1))
            result_n = psia_times_PA_for_obs_seq_n[n] * multiplied_list
            result_list.append(result_n)
        #normalize
        normalized_result_list = []
        for obs_seq_dist in result_list:
            normalized_result_list.append(np.divide(obs_seq_dist, obs_seq_dist.sum(axis=1)[:,np.newaxis], out=np.zeros_like(obs_seq_dist), where=obs_seq_dist.sum(axis=1)[:,np.newaxis]!=0))
        #write_output
        for i,elem in enumerate(normalized_result_list):
            np.savetxt("output_dist_mod{}_obs{}.csv".format(focus_index,i), elem, delimiter=",")
        #multiply all together
        final_dist = reduce(lambda x, y: x*y, normalized_result_list)
        #normalise
        row_sums = final_dist.sum(axis=1)
        final_dist = final_dist / row_sums[:, np.newaxis]
        self.final_dist.append(final_dist)

        np.savetxt("output_dist_mod{}.csv".format(focus_index), final_dist, delimiter=",")
        '''
    #get values for PA, PB ... and psis
    def __get_probabilities(self):
        self.probs = []
        self.psis = []
        for mod in self.model_list:
            self.probs.append(mod.PA) # dim = (models,obs_seq,T,state_space)
            self.psis.append(mod.psi) #dim = (models,obs_seq,T,state_space)
        #each model has a psi and a PA
        #so here we now have a list of 5 PAs and a list of 5 psis.
        #dimension of ordered_probs = (obs_seq,models,T,state_space)

    def get_final_dist(self):
        return self.final_dist
