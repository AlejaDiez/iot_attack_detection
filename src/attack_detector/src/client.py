from copy import deepcopy
from keras import Model
from keras.api.metrics import Accuracy, Precision, Recall, AUC
from flwr.client import NumPyClient, start_client
from flwr.common import NDArrays, Scalar
import numpy as np
import tensorflow as tf


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
            self.x_train,
            self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1,
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

        # Evaluar el modelo local y obtener las clases predichas
        y_pred_probs = self.model.predict(self.x_test, batch_size=self.batch_size)
        if y_pred_probs.shape[1] == 1:
            y_pred = (y_pred_probs > 0.5).astype("int32").flatten()
            y_true = self.y_test.flatten()
        else:
            y_pred = np.argmax(y_pred_probs, axis=1)
            y_true = np.argmax(self.y_test, axis=1)

        # Calcular las métricas
        # Accuracy
        accuracy = Accuracy()
        accuracy.update_state(y_true, y_pred)
        accuracy_val = float(accuracy.result())
        # Precision
        precision = Precision()
        precision.update_state(y_true, y_pred)
        precision_val = float(precision.result())
        # Recall
        recall = Recall()
        recall.update_state(y_true, y_pred)
        recall_val = float(recall.result())
        # F1 Score
        f1_score = (
            2 * (precision_val * recall_val) / (precision_val + recall_val + 1e-8)
        )
        # AUC
        auc = AUC()
        if y_pred_probs.shape[1] == 1:
            auc.update_state(y_true, y_pred_probs)
        else:
            y_true_oh = tf.one_hot(y_true, depth=y_pred_probs.shape[1])
            auc.update_state(y_true_oh, y_pred_probs)
        auc_val = float(auc.result())
        # Confusion Matrix
        confusion_matrix = tf.math.confusion_matrix(y_true, y_pred).numpy().tolist()
        # Loss
        loss, _ = self.model.evaluate(
            self.x_test, self.y_test, batch_size=self.batch_size, verbose=0
        )
        return (
            loss,
            len(self.x_test),
            {
                "accuracy": accuracy_val,
                "precision": precision_val,
                "recall": recall_val,
                "f1_score": f1_score,
                "auc": auc_val,
                "confusion_matrix": str(confusion_matrix),
            },
        )

    def start(self, server_address: str):
        """
        Iniciar el cliente.

        Args:
            server_address (str): Dirección del servidor.
        """
        start_client(server_address=server_address, client=self.to_client())
