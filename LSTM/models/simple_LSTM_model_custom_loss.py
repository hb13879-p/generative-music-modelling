from base.base_model import BaseModel
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras import losses
from utils.losses import musical_closeness_loss3
from utils.metrics import chord_accuracy


class SimpleLSTMModel(BaseModel):
    def __init__(self, config):
        super(SimpleLSTMModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(
            LSTM(
                self.config.model.hidden_units,
                input_shape=(
                    self.config.data.seq_length,
                    self.config.data.no_input_features,
                ),
                return_sequences=True,
            )
        )
        self.model.add(Dense(self.config.data.no_output_units, activation="sigmoid"))

        self.model.compile(
            loss=musical_closeness_loss3,
            optimizer=self.config.model.optimizer,
            metrics=["acc", chord_accuracy],
        )
