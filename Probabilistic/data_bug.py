from data_loaders.hmm_aug_with_melody_data_loader import HmmMelodyDataLoader
import pickle
import os
from utils.config import process_config
from utils.dirs import create_dirs
from utils.args import get_args
import csv


def main():

    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)
    data_loader = HmmMelodyDataLoader(config)
    labels = pickle.load(open(r"data_bug/labels.p", "rb"))
    voicings = pickle.load(open(r"data_bug/voicings.p", "rb"))
    print("labels0")
    print(labels[0][0])
    print(labels[1][0])
    print(labels[2][0])
    print(labels[3][0])
    print(labels[4][0])
    print(labels[5][0])
    print("voicing0")
    print(voicings[0][0])
    print(voicings[1][0])
    print(voicings[2][0])
    print(voicings[3][0])
    print(voicings[4][0])
    print("labels410")
    print(labels[0][408815])
    print(labels[1][408815])
    print(labels[2][408815])
    print(labels[3][408815])
    print(labels[4][408815])
    print(labels[5][408815])
    print("voicing410")
    print(voicings[0][408815])
    print(voicings[1][408815])
    print(voicings[2][408815])
    print(voicings[3][408815])
    print(voicings[4][408815])


if __name__ == "__main__":
    main()
