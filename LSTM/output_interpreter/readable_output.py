from utils.utils import demidify
import numpy as np

class ReadableOutput(object):
    def __init__(self,result,Xtest,sample_no):
        self.result = result
        self.predictions = self.result_to_predictions(result)
        self.voicings, self.midi_voicings = self.one_hot_to_voicing_list(self.predictions[sample_no])
        self.labels = self.one_hot_input_to_readable_label(Xtest[sample_no])

    def result_to_predictions(self, result):
        return (result > 0.5).astype(int)

    def one_hot_to_voicing_list(self, result):
        voicings = []
        midi_voicings = []
        for row in result:
            tmp = []
            midi_tmp = []
            x = np.where(row)
            x = x[0] + 51
            for elem in x:
                midi_tmp.append(elem)
                tmp.append(demidify(elem))
            voicings.append(tmp)
            midi_voicings.append(midi_tmp)
        return voicings, midi_voicings

    def one_hot_input_to_readable_label(self, label):
        labels = []
        root_demap = {0 : 'A', 1 : 'Bb', 2:'B',3:'C',4:'Db',5:'D',6:'Eb',7:'E',8:'F',9:'F#',10:'G',11:'Ab'}
        quality_map = {0:'diminished',1:'dominant',2:'halfdim',3:'major',4:'minor',5:'majmin',6:'augmented',7:'domshp11'}
        for row in label:
            x = np.where(row)
            x = x[0]
            root = x[0]
            root = root_demap[root]
            quality = x[1] - 12
            quality = quality_map[quality]
            labels.append(root + quality)
        return labels
