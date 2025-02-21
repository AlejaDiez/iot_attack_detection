import flwr as fl
import tensorflow as tf

from data import load_datasets
from model import keras_model

model = keras_model()
train, test = load_datasets(400, limit=0.8)

train_len = sum(1 for _ in train.unbatch())
test_len = sum(1 for _ in test.unbatch())

class FlowerClient(fl.client.NumPyClient):
    def get_weights(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(train, epochs=100)
        return model.get_weights(), train_len, {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(test)
        return loss, test_len, {"accuracy": accuracy}


# Start Flower client
fl.client.start_client(server_address="[::]:8080", client=FlowerClient())

