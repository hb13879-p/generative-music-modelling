class ReadableOutput(object):
    def __init__(self,evaluator,data_loader):
        self.paths = evaluator.get_paths()
        self.label_encoders = data_loader.get_label_encoders()
        self.midi_paths = []
        self.__decode_paths()

    def __decode_paths(self):
        for i,enc in enumerate(self.paths):
            self.midi_paths.append(self.label_encoders[i].inverse_transform(list(map(int,enc))))
