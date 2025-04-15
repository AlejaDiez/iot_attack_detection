from grpc import RpcError
from keras import Model
from flwr.client import NumPyClient, start_client
from flwr.common import NDArrays, Scalar
from utils.model import clone_model, evaluate_model


class TensorFlowClient(NumPyClient):
    """
    Cliente de TensorFlow para Flower.

    Args:
        model (Model): Modelo de Keras.
        train_data (tuple[list, list]): Datos de entrenamiento.
        test_data (tuple[list, list]): Datos de prueba.
        batch_size (int, optional): Tamaño del lote. Por defecto es 32.
        epochs (int, optional): Número de épocas. Por defecto es 1.
    """

    def __init__(
        self,
        model: Model,
        train_data: tuple[list, list],
        test_data: tuple[list, list],
        batch_size: int = 32,
        epochs: int = 1,
    ):
        self.model = clone_model(model)
        self.x_train, self.y_train = train_data
        self.x_test, self.y_test = test_data
        self.batch_size = batch_size
        self.epochs = epochs

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        """
        Obtener los pesos del modelo.

        Args:
            config (dict[str, Scalar]): Parámetros de configuración solicitados por el servidor.

        Returns:
            NDArrays: Parámetros del modelo local.
        """
        return self.model.get_weights()

    def fit(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """
        Ajustar el modelo.

        Args:
            parameters (NDArrays): Parámetros del modelo.
            config (dict[str, Scalar]): Parámetros de configuración solicitados por el servidor.

        Returns:
            tuple[NDArrays, int, dict[str, Scalar]]: Parámetros actualizados, número de ejemplos y métricas.
        """
        print(
            f"\033[1m\033[94mTraining\033[0m the local model in round \033[92m{config.get('server_round', '?')}\033[0m"
        )

        # Actualizar el modelo local con los pesos recibidos del servidor
        self.model.set_weights(parameters)

        # Entrenar el modelo local
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1,
        )

        # Obtener las metricas del entrenamiento
        metrics = {
            "loss": str(history.history["loss"]),
            "accuracy": str(history.history["accuracy"]),
        }

        return (
            self.get_parameters(config),
            len(self.x_train),
            metrics,
        )

    def evaluate(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, int, dict[str, Scalar]]:
        """
        Evaluar el modelo.

        Args:
            parameters (NDArrays): Parámetros del modelo.
            config (dict[str, Scalar]): Parámetros de configuración solicitados por el servidor.

        Returns:
            tuple[float, int, dict[str, Scalar]]: Pérdida, número de ejemplos y métricas.
        """
        print(
            f"\033[1m\033[95mEvaluating\033[0m the local model in round \033[92m{config.get('server_round', '?')}\033[0m"
        )

        # Generar un nuevo modelo local y cargar los pesos para la evaluación
        model = clone_model(self.model)
        model.set_weights(parameters)

        # Evaluar el modelo local
        loss, metrics = evaluate_model(
            model, self.x_test, self.y_test, batch_size=self.batch_size
        )

        return (loss, len(self.x_test), metrics)

    def start(self, server_address: str):
        """
        Iniciar el cliente.

        Args:
            server_address (str): Dirección del servidor.
        """
        try:
            print("\033[1mTensorFlow Client\033[0m")
            start_client(server_address=server_address, client=self.to_client())
        except KeyboardInterrupt:
            exit(0)
        except RpcError as e:
            print(
                f"\n\033[91mError: an unexpected error occurred ({e.code().name})\033[0m"
            )
            exit(1)
