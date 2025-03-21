import json
from copy import deepcopy
from pathlib import Path
from keras import Model
from flwr.common import (
    FitRes,
    Parameters,
    Metrics,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server import start_server, ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from utils.path import get_abs_path


class TensorFlowServer(FedAvg):
    """
    Servidor personalizado para el uso de modelos de TensorFlow. Este servidor extiende la clase FedAvg de Flower.
    """

    def __init__(
        self,
        model: Model,
        test_data: tuple[list, list],
        output: str,
        num_rounds: int = 1,
        batch_size: int = 32,
        fraction_fit: float = 1,
        fraction_evaluate: float = 1,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
    ):
        """
        Inicializa un nuevo servidor personalizado para el uso de modelos de TensorFlow.

        Args:
            model (Model): Modelo de TensorFlow.
            test_data (tuple[list, list]): Tupla con los datos de prueba (x_test, y_test).
            num_rounds (int, optional): Número de rondas de entrenamiento. Defaults to 1.
            batch_size (int, optional): Tamaño del lote. Defaults to 32.
            fraction_fit (float, optional): Fracción de clientes utilizados para el ajuste. Defaults to 1.
            fraction_evaluate (float, optional): Fracción de clientes utilizados para la evaluación. Defaults to 1.
            min_fit_clients (int, optional): Número mínimo de clientes para el ajuste. Defaults to 2.
            min_evaluate_clients (int, optional): Número mínimo de clientes para la evaluación. Defaults to 2.
            min_available_clients (int, optional): Número mínimo de clientes disponibles. Defaults to 2.
        """

        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            initial_parameters=ndarrays_to_parameters(model.get_weights()),
            evaluate_metrics_aggregation_fn=self.weighted_average,
            evaluate_fn=self.evaluate_fn(),
        )
        self.model = deepcopy(model)
        self.test_data = test_data
        self.num_rounds = num_rounds
        self.batch_size = batch_size
        self.results = {}
        self.output_dir = get_abs_path(output)

        # Crear directorio de salida si no existe
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def get_model(self) -> Model:
        """
        Devuelve una copia profunda del modelo.

        Returns:
            Model: Copia profunda del modelo.
        """
        return deepcopy(self.model)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:
        """
        Agrega los resultados de la ronda de ajuste. Actualiza los pesos del modelo con los pesos agregados mediante la media.

        Args:
            server_round (int): Ronda del servidor.
            results (list[tuple[ClientProxy, FitRes]]): Resultados de la ronda.
            failures (list[tuple[ClientProxy, FitRes] | BaseException]): Fallos de la ronda.

        Returns:
            tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]: Parámetros y métricas.
        """
        # Llamada al comportamiento predeterminado de FedAvg
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )
        # Actualizar los pesos del modelo con los pesos agregados
        ndarrays = parameters_to_ndarrays(parameters_aggregated)
        self.model.set_weights(ndarrays)
        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[float, dict]:
        """
        Agrega los resultados de la ronda de evaluación.

        Args:
            server_round (int): Ronda del servidor.
            results (list[tuple[ClientProxy, FitRes]]): Resultados de la ronda.
            failures (list[tuple[ClientProxy, FitRes] | BaseException]): Fallos de la ronda.

        Returns:
            tuple[float, dict]: Pérdida y métricas.
        """
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        return loss, metrics

    def weighted_average(self, metrics: list[tuple[int, Metrics]]) -> Metrics:
        """ "
        Calcula la precisión promedio ponderada.

        Args:
            metrics (list[tuple[int, Metrics]]): Lista de tuplas con el número de ejemplos y las métricas.
        """
        # Calcular la precisión promedio ponderada
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        total_examples = sum(num_examples for num_examples, _ in metrics)

        # Retornar la precisión promedio ponderada
        return {"accuracy": sum(accuracies) / total_examples}

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        """
        Evalúa el modelo con los parámetros dados.

        Args:
            server_round (int): Ronda del servidor.
            parameters (Parameters): Parámetros del modelo.

        Returns:
            tuple[float, dict[str, bool | bytes | float | int | str]] | None: Pérdida y métricas.
        """
        # Llamada al comportamiento predeterminado de FedAvg
        loss, metrics = super().evaluate(server_round, parameters)
        # Guardar resultados en un diccionario local
        self.results[server_round] = {"loss": loss, **metrics}

        # Guardar resultados en un archivo JSON
        with open(f"{self.output_dir}/results.json", "w") as file:
            json.dump(self.results, file, indent=4)
        # Retornar resultados para que sean agregados
        return loss, metrics

    def evaluate_fn(self):
        """
        Devuelve una función de evaluación que evalúa el modelo en los datos de prueba.

        Returns:
            function: Función de evaluación.
        """

        def evaluate(server_round, parameters_ndarrays, config):
            """
            Evalúa el modelo en los datos de prueba.

            Args:
                server_round (int): Ronda del servidor.
                parameters_ndarrays (list): Parámetros del modelo.
                config (Config): Configuración del servidor.

            Returns:
                tuple: Pérdida y métricas.
            """
            model = self.get_model()
            model.set_weights(parameters_ndarrays)
            loss, accuracy = model.evaluate(
                self.test_data[0], self.test_data[1], batch_size=self.batch_size
            )

            return loss, {"accuracy": accuracy}

        return evaluate

    def start(
        self, server_address: str, certificates: tuple[bytes, bytes, bytes] = None
    ):
        """
        Inicia el servidor.

        Args:
            server_address (str): Dirección del servidor.
            certificates (tuple[bytes, bytes, bytes], optional): Certificados para la conexión segura. Defaults to None
        """
        start_server(
            server_address=server_address,
            config=ServerConfig(num_rounds=self.num_rounds),
            strategy=self,
            certificates=certificates,
        )
