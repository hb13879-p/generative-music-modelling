import csv

class WritableOutput(object):
    def __init__(self, config,trainer,evaluator):
        self.name = config.exp.name
        self.seq_length = str(config.data.seq_length)
        self.learning_rate = str(config.model.learning_rate)
        self.hidden_units = str(config.model.hidden_units)
        self.epochs = str(config.trainer.num_epochs)
        self.batch_size = str(config.trainer.batch_size)
        self.train_chord_acc = str(round(trainer.chord_acc[-1],2))
        self.train_loss = str(round(trainer.loss[-1],2))
        self.train_acc = str(round(trainer.acc[-1],2))
        self.val_chord_acc = str(round(trainer.val_chord_acc[-1],2))
        self.val_loss = str(round(trainer.val_loss[-1]))
        self.val_acc = str(round(trainer.val_acc[-1],2))
        self.test_chord_acc = str(round(evaluator.chord_accuracy,2))
        self.test_loss = str(round(evaluator.loss,2))
        self.test_acc = str(round(evaluator.acc,2))
        self.__create_list()

    def __create_list(self):
        self.args = []
        self.args.extend((self.name, self.seq_length,self.learning_rate,self.hidden_units,self.epochs,self.batch_size,self.train_loss,self.train_acc,self.val_loss,self.val_acc,self.test_loss,self.test_acc,self.train_chord_acc,self.val_chord_acc,self.test_chord_acc))

    def write(self, filename):
        with open(filename, "a") as myfile:
            writer = csv.writer(myfile)
            writer.writerow(self.args)
