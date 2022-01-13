import sys
import os.path
pathname = r"C:\Users\User\Documents\Project\Dataset"
sys.path.insert(0, os.path.normpath(pathname))
pathname2 = r"C:\Users\User\Documents\Project\Models\LSTM"
sys.path.insert(0, os.path.normpath(pathname2))

from utils.losses import musical_closeness_loss3
from utils.metrics import chord_accuracy
from midiparse2 import MidiParser
from labelparse2 import MusicXMLParser
import xml.etree.cElementTree as ET
import numpy as np
from output_interpreter.readable_output import ReadableOutput
from output_interpreter.music_xml_writer import MusicXmlWriter
from utils.config import process_config
from utils.args import get_args
from keras.models import load_model

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
    custom_loss = config.model_data.custom_loss
    output_score_filename = config.model_data.output_score_filename
    model_filename = config.model_data.model_filename
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
    X_no_melody = np.hstack((musicxml_data.rootnums,musicxml_data.qualitynums))
    if not inc_melody:
        m = int(np.shape(X_no_melody)[0] / seq_length)
        X = X_no_melody.reshape(m,seq_length,np.shape(X_no_melody)[1])
    else:
        X_inc_melody = np.concatenate((X_no_melody,encoded_melody),axis=1)
        mmel = int(np.shape(X_inc_melody)[0] / seq_length)
        X = X_inc_melody.reshape(mmel,seq_length,np.shape(X_inc_melody)[1])
    if custom_loss:
        model = load_model(model_filename,custom_objects={'musical_closeness_loss3': musical_closeness_loss3,'chord_accuracy':chord_accuracy})
    else:
        model = load_model(model_filename)
    result = model.predict(X)
    readable_output = ReadableOutput(result,X,0)
    music_xml_writer = MusicXmlWriter(readable_output.voicings,readable_output.labels)
    music_xml_writer.insert_melody(melody_xml)
    music_xml_writer.write(output_score_filename)

def get_melody_xml(tree):
    root = tree.getroot()
    return root.find(".//part[@id='" + "P1" + "']")


def format_labels(musicxml_data):
    musicxml_data.alter_roots()
    musicxml_data.alter_degrees()
    musicxml_data.encode_qualities()
    musicxml_data.rootnums = np.array(musicxml_data.rootnums)
    musicxml_data.rootnums = convert_to_one_hot(musicxml_data.rootnums,12)
    musicxml_data.qualitynums = np.array(musicxml_data.qualitynums)
    musicxml_data.qualitynums = convert_to_one_hot(musicxml_data.qualitynums,8)

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





if(__name__ == '__main__'):
    main()
