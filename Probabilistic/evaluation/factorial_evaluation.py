import sys
import os.path
pathname = r"C:\Users\User\Documents\Project\Dataset"
sys.path.insert(0, os.path.normpath(pathname))
pathname2 = r"C:\Users\User\Documents\Project\Models\Probabilistic"
sys.path.insert(0, os.path.normpath(pathname2))

from midiparse2 import MidiParser
from labelparse2 import MusicXMLParser
import xml.etree.cElementTree as ET
import numpy as np
import pickle
from output_interpreter.music_xml_writer import MusicXmlWriter
from output_interpreter.readable_output import ReadableOutput
from evaluators.simple_hmm_evaluator import SimpleHmmEvaluator
from utils.utils import demidify
from utils.config import process_config
from utils.args import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    PPQ = config.musical_data.PPQ
    min_res = config.musical_data.min_res #minimum resolution of chord and melody changes in terms of notes per beat - eg 8th note = 2
    low = config.musical_data.low #Eb
    high = config.musical_data.high #G
    chord_rhythm = config.musical_data.chord_rhythm

    seq_length = config.model_data.seq_length
    inc_melody = config.model_data.inc_melody
    output_score_filename = config.model_data.output_score_filename
    model_filename = config.model_data.model_filename
    data_loader_filename = config.model_data.data_loader_filename
    test_filename = r"C:\Users\User\Documents\Project\Test_Scores/" + config.model_data.test_filename

    midi_data = MidiParser(PPQ,min_res,high,low)
    midi_data.parseMidi(test_filename + ".mid")
    melody = midi_data.melody
    musicxml_data = MusicXMLParser()
    tree = ET.ElementTree(file=test_filename+".xml")
    musicxml_data.parse_xml_tree(tree)

    melody_xml = get_melody_xml(tree)
    format_labels(musicxml_data)
    encoded_melody = encode_melody(melody,chord_rhythm)
    encoded_qualities, n_qualities = encode_qualities(musicxml_data.qualities)
    X = [list(musicxml_data.rootnums),encoded_qualities]
    X = list(X)

    model = pickle.load(open(model_filename, "rb" ))
    data_loader = pickle.load(open(data_loader_filename, "rb" ))
    label_encoders = data_loader.get_label_encoders()
    evaluator = SimpleHmmEvaluator(model,[[3,8,1,6,0,5,10],[4,1,3,3,2,1,5],[-1,-1,-1,-1,-1,-1,-1],[-1,-1,3,5,-1,11,-1],[0,0,0,0,4,4,5],[2,3,-1,-1,-1,2,9]],config)
    evaluator.factorial_sample(0)
    evaluator.factorial_sample(1)
    evaluator.factorial_sample(2)
    evaluator.factorial_sample(3)
    evaluator.factorial_sample(4)
    final_dists = evaluator.get_final_dist()
    print(len(final_dists))
    results = []
    for mod,le in zip(final_dists,label_encoders):
        mod_res = []
        for t in mod:
            mod_res.append(np.random.choice(le.classes_,p=t))
        results.append(mod_res)
    print(results)
    voicing_list = []
    for chord in results:
        cho = []
        for note in chord:
            if note != 0:
                cho.append(demidify(note))
        voicing_list.append(cho)
    root_demap = {0 : 'A', 1 : 'Bb', 2:'B',3:'C',4:'Db',5:'D',6:'Eb',7:'E',8:'F',9:'F#',10:'G',11:'Ab'}
    quality_map = {0:'diminished',1:'dominant',2:'halfdim',3:'major',4:'minor',5:'majmin',6:'augmented',7:'domshp11'}
    new_qualities = []
    new_roots = []
    for q,r in zip(encoded_qualities,musicxml_data.rootnums):
        new_qualities.append(quality_map[q])
        new_roots.append(root_demap[r])
    labels = [r+q for r,q in zip(new_roots,new_qualities)]
    music_xml_writer = MusicXmlWriter(voicing_list,labels)
    music_xml_writer.insert_melody(melody_xml)
    music_xml_writer.write(output_score_filename)


def get_melody_xml(tree):
    root = tree.getroot()
    return root.find(".//part[@id='" + "P1" + "']")


def format_labels(musicxml_data):
    musicxml_data.alter_roots()
    musicxml_data.alter_degrees()
    musicxml_data.rootnums = np.array(musicxml_data.rootnums)
    musicxml_data.qualitynums = np.array(musicxml_data.qualitynums)

def convert_to_one_hot(arr,categories):
    b = np.zeros((arr.size, categories))
    b[np.arange(arr.size), arr] = 1
    return b

def encode_melody(melody,chord_rhythm):
    i = 0
    encoded_melody = []
    for chord in chord_rhythm:
        mel = melody[i:chord]
        encoded_melody.append(create_12d_vector_and_extract_melody(mel))
        i += len(mel)
    return encoded_melody

def create_12d_vector_and_extract_melody(melody):
    if len(melody) == 0:
        return np.zeros(12)
    res = np.zeros(12)
    melody = [(x + 3) % 12 for x in melody]
    for note in melody:
        res[note] += 1
    res /= len(melody)
    assert np.sum(res) == 1
    return res

def encode_qualities(qualities):
    encoded_quals = []
    qual_encode = {'diminished-seventh' : 0, 'diminished' : 0, 'dominant' : 1, "dominantsharp9" : 1, 'dominant-ninth' : 1, 'dominantflat9' : 1,
    'dominant-13th' : 1, "dominant-ninthsus4" : 1, "dominant-11th" : 1, "dominantalt" : 1, 'dominant-ninthflat5' : 1, "augmented7#9" : 1,
    'dominant-13thflat9' : 1, "dominantsus4" : 1, 'dominantflat5' : 1,'half-diminished' : 2, 'major' : 3,\
    'major-seventh' : 3, 'major-sixth' : 3, "major-seventhsharp5" : 3, "major-sixth9" : 3, 'major-ninth' : 3, 'major-seventhsharp11' : 3, 'minor-seventh' : 4, 'minor-11th' : 4, 'minor' : 4, 'minor-ninth' : 4, \
    'minor-sixth' : 5, "minor-sixth9" : 5, 'major-minor' : 5, 'augmented-seventh' : 6, 'augmented' : 6, 'dominantsharp5' : 6, "dominantsharp11" : 7,
    "dominant-13thsharp11" : 7, "dominant-ninthsharp11" : 7 }
    n_qualities = len(set(elem for elem in qual_encode.values()))
    for c in qualities:
        encoded_quals.append(qual_encode[c])
    return encoded_quals, n_qualities

def note_generator(self):
    for chord in self.voicings:
        #get bottom note
        yield chord[0]
        chord = chord[1:]
        #get top note
        yield chord[-1]
        chord = chord[0:-1]
        #check if there are other notes and return them bottom to top
        for i in range(3):
            if not chord:
                yield 0
            else:
                yield chord[0]
                chord = chord[1:]



if(__name__ == '__main__'):
    main()
