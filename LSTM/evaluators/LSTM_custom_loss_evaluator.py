from keras.models import Sequential

class SimpleLSTMEvaluator(object):
    def __init__(self,model,test_data):
        self.model = model
        self.test_data = test_data
        self.evaluate()

    def evaluate(self):
        self.loss, self.acc, self.chord_accuracy = self.model.evaluate(self.test_data[0],self.test_data[1])
        self.sample_result = self.model.predict(self.test_data[0])
