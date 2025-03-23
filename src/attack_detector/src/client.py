from copy import deepcopy
from keras import Model
from flwr.client import NumPyClient, start_client
from flwr.common import NDArrays, Scalar


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
        self.model = deepcopy(model)
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
        # Actualizar el modelo local con los pesos recibidos del servidor
        self.model.set_weights(parameters)
        # Entrenar el modelo local
        self.model.fit(
            self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs
        )
        return self.get_parameters(config), len(self.x_train), {}

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
        # Actualizar el modelo local con los pesos recibidos del servidor
        self.model.set_weights(parameters)
        # Evaluar el modelo local
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": accuracy}

    def start(self, server_address: str):
        """
        Iniciar el cliente.

        Args:
            server_address (str): Dirección del servidor.
        """
        start_client(server_address=server_address, client=self.to_client())
