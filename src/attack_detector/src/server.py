import json
from pathlib import Path
from typing import Optional, Union
from grpc import RpcError
from keras import Model
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import start_server, ServerConfig
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import (
    aggregate,
    aggregate_inplace,
    weighted_loss_avg,
)
from utils.model import clone_model, evaluate_model
from utils.path import get_abs_path


class TensorFlowServer(FedAvg):
    """
    Servidor personalizado para el uso de modelos de TensorFlow. Este servidor extiende de la clase FedAvg de Flower y permite la agregación de resultados de múltiples rondas de entrenamiento y evaluación.

    Args:
        model (Model): Modelo de TensorFlow.
        test_data (tuple[list, list]): Tupla con los datos de prueba (x_test, y_test).
        output (str): Directorio de salida.
        batch_size (int, optional): Tamaño del lote. Por defecto es 32.
        num_rounds (int, optional): Número de rondas de entrenamiento. Por defecto es 1.
        fraction_fit (float, optional): Fracción de clientes utilizados para el ajuste. Por defecto es 1.
        fraction_evaluate (float, optional): Fracción de clientes utilizados para la evaluación. Por defecto es 1.
        min_fit_clients (int, optional): Número mínimo de clientes para el ajuste. Por defecto es 2.
        min_evaluate_clients (int, optional): Número mínimo de clientes para la evaluación. Por defecto es 2.
        min_available_clients (int, optional): Número mínimo de clientes disponibles. Por defecto es 2.
    """

    def __init__(
        self,
        model: Model,
        test_data: tuple[list, list],
        output: str,
        batch_size: int = 32,
        num_rounds: int = 1,
        fraction_fit: float = 1,
        fraction_evaluate: float = 1,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
    ):
        self.model = clone_model(model)
        self.x_test, self.y_test = test_data
        self.batch_size = batch_size
        self.num_rounds = num_rounds
        # Inicializar el servidor
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            initial_parameters=ndarrays_to_parameters(self.model.get_weights()),
        )
        # Crear directorio de salida si no existe
        self.output_dir = get_abs_path(output)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        # Metricas
        self.metrics = {}

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """
        Configura la siguiente ronda de entrenamiento.

        Args:
            server_round (int): Ronda del servidor.
            parameters (Parameters): Parámetros del modelo.
            client_manager (ClientManager): Gestor de clientes.

        Returns:
            list[tuple[ClientProxy, FitIns]]: Lista de tuplas (cliente, configuración de ajuste).
        """
        # Instrucciones de ajuste para los clientes
        fit_ins = FitIns(parameters, {"server_round": server_round})

        # Seleccionar clientes
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        print(
            f"\033[1m\033[96mTraining\033[0m the model on {len(clients)} clients in round \033[92m{server_round}\033[0m"
        )

        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """
        Junta los resultados de la ronda de ajuste. Actualiza los pesos del modelo con los pesos agregados mediante la media.

        Args:
            server_round (int): Ronda del servidor.
            results (list[tuple[ClientProxy, FitRes]]): Resultados de la ronda.
            failures (list[Union[tuple[ClientProxy, FitRes], BaseException]]): Fallos de la ronda.

        Returns:
            tuple[Optional[Parameters], dict[str, Scalar]]: Parámetros y métricas.
        """
        # No juntar si no hay resultados o si hay fallos y no se aceptan
        if not results or (not self.accept_failures and failures):
            return None, {}

        # Agregar resultados mediante una media ponderada
        if self.inplace:
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)
        self.model.set_weights(aggregated_ndarrays)

        # Metricas de la unión de los resultados de ajuste
        metrics_aggregated = {
            "num_round": server_round,
            "num_clients": len(results),
            "num_failures": len(failures),
        }

        # Guardar métricas
        if server_round not in self.metrics:
            self.metrics[server_round] = {
                "training": {},
                "evaluation": {},
            }
        self.metrics[server_round]["training"] = {
            "clients": {
                client.cid: {
                    **{key: json.loads(value) for key, value in res.metrics.items()},
                    "num_examples": res.num_examples,
                }
                for client, res in results
            },
            "failures": len(failures),
        }
        return ndarrays_to_parameters(aggregated_ndarrays), metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, Scalar]]:
        """
        Evalúa el modelo global con los parámetros dados.

        Args:
            server_round (int): Ronda del servidor.
            parameters (Parameters): Parámetros del modelo.

        Returns:
            tuple[float, dict[str, Scalar]]: Pérdida y métricas.
        """
        print(
            f"\033[1m\033[95mEvaluating\033[0m the global model in round \033[92m{server_round}\033[0m"
        )

        # Generar una copia del modelo con los pesos actuales
        model = clone_model(self.model)
        model.set_weights(parameters_to_ndarrays(parameters))

        # Evaluar el modelo
        loss, metrics = evaluate_model(model, self.x_test, self.y_test, self.batch_size)

        # Guardar metricas
        if server_round not in self.metrics:
            self.metrics[server_round] = {
                "training": {},
                "evaluation": {},
            }
        self.metrics[server_round]["evaluation"] = {
            "loss": loss,
            **{
                key: json.loads(value) if isinstance(value, str) else value
                for key, value in metrics.items()
            },
            "num_examples": len(self.x_test),
        }

        return loss, metrics

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """
        Configura la siguiente ronda de evaluación.

        Args:
            server_round (int): Ronda del servidor.
            parameters (Parameters): Parámetros del modelo.
            client_manager (ClientManager): Gestor de clientes.

        Returns:
            list[tuple[ClientProxy, EvaluateIns]]: Lista de tuplas (cliente, configuración de evaluación).
        """
        # Compronar si hay clientes disponibles para evaluar
        if self.fraction_evaluate == 0.0:
            return []

        # Instrucciones de evaluación para los clientes
        evaluate_ins = EvaluateIns(parameters, {"server_round": server_round})

        # Seleccionar clientes
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        print(
            f"\033[1m\033[93mEvaluating\033[0m the global model on {len(clients)} clients in round \033[92m{server_round}\033[0m"
        )

        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """
        Junta los resultados de la ronda de evaluación.

        Args:
            server_round (int): Ronda del servidor.
            results (list[tuple[ClientProxy, EvaluateRes]]): Resultados de la ronda.
            failures (list[Union[tuple[ClientProxy, EvaluateRes], BaseException]]): Fallos de la ronda.

        Returns:
            tuple[Optional[float], dict[str, Scalar]]: Pérdida y métricas.
        """
        # No juntar si no hay resultados o si hay fallos y no se aceptan
        if not results or (not self.accept_failures and failures):
            return None, {}

        # Juntar la perdida mediante una media ponderada
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Metricas de la unión de los resultados de evaluación
        metrics_aggregated = {
            "num_round": server_round,
            "num_clients": len(results),
            "num_failures": len(failures),
        }

        # Guardar métricas
        self.metrics[server_round]["evaluation"] = {
            **self.metrics[server_round]["evaluation"],
            "clients": {
                client.cid: {
                    "loss": res.loss,
                    **{
                        key: json.loads(value) if isinstance(value, str) else value
                        for key, value in res.metrics.items()
                    },
                    "num_examples": res.num_examples,
                }
                for client, res in results
            },
            "failures": len(failures),
        }

        return loss_aggregated, metrics_aggregated

    def save_model(self, model_name: str = "model.keras"):
        """
        Guarda el modelo en el directorio de salida.

        Args:
            model_name (str, optional): Nombre del modelo, por defecto es "model.keras".
        """
        self.model.save(f"{self.output_dir}/{model_name}")

    def save_metrics(self, metrics_name: str = "metrics.json"):
        """
        Guarda las métricas en un archivo JSON.

        Args:
            metrics_name (str, optional): Nombre del archivo de métricas, por defecto es "metrics.json".
        """
        with open(f"{self.output_dir}/{metrics_name}", "w", encoding="utf-8") as file:
            json.dump(self.metrics, file, indent=4)

    def start(
        self, server_address: str, certificates: tuple[bytes, bytes, bytes] = None
    ):
        """
        Inicia el servidor.

        Args:
            server_address (str): Dirección del servidor.
            certificates (tuple[bytes, bytes, bytes], optional): Certificados para la conexión segura. Defaults to None
        """
        try:
            print(f"\033[1mTensorFlow Server at {server_address}\033[0m")
            start_server(
                server_address=server_address,
                config=ServerConfig(num_rounds=self.num_rounds),
                strategy=self,
                certificates=certificates,
            )
        except KeyboardInterrupt:
            exit(0)
        except RpcError as e:
            print(
                f"\n\033[91mError: an unexpected error occurred ({e.code().name})\033[0m"
            )
            exit(1)
