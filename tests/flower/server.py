from typing import Optional
import flwr as fl
from matplotlib import pyplot as plt
import numpy as np
from data import fn
from model import keras_model


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results,
        failures,
    ) -> Optional[fl.common.Parameters]:
        weights = super().aggregate_fit(rnd, results, failures)
        if weights is not None:
            self.global_weights = weights[0]
        return weights

    def get_global_weights(self):
        return self.global_weights


model = keras_model()
config = fl.server.ServerConfig(num_rounds=4)
strategy = SaveModelStrategy(
    initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    fraction_fit=0.8,  # clientes que participan en cada ronda
    fraction_evaluate=0.5,  # clientes que se evalúan en cada ronda
    min_fit_clients=2,  # minimo de clientes para el entrenamiento
    min_evaluate_clients=2,  # minimo de clientes para la evaluación
    min_available_clients=2,  # minimos clientes disponibles
)
fl.server.start_server(
    server_address="localhost:8080", config=config, strategy=strategy
)

# Hacer predicciones
model.set_weights(fl.common.parameters_to_ndarrays(strategy.get_global_weights()))
X = np.random.uniform(-10, 10, 200)
y = fn(X)
p = model.predict(X)

# Mostrar las predicciones
plt.figure(figsize=(10, 5))
plt.title("Predicciones")
plt.xlabel("X")
plt.ylabel("y")
plt.grid()
for i in range(len(X)):
    plt.scatter(X[i], y[i], color="black")
    plt.scatter(X[i], p[i], color="red")
plt.show()
