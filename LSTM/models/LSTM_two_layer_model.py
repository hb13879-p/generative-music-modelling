from base.base_model import BaseModel
from keras.models import Sequential
from keras.layers import LSTM, Dense


class LSTMTwoLayerModel(BaseModel):
    def __init__(self, config):
        super(LSTMTwoLayerModel, self).__init__(config)
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
        self.model.add(LSTM(self.config.model.hidden_units, return_sequences=True))
        self.model.add(Dense(self.config.data.no_output_units, activation="sigmoid"))

        self.model.compile(
            loss="binary_crossentropy",
            optimizer=self.config.model.optimizer,
            metrics=["acc"],
        )
