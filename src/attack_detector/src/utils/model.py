from keras import Model, models, optimizers, metrics
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from utils.path import get_abs_path


def load_model(path: str) -> Model:
    """
    Carga un modelo de un archivo JSON.

    Args:
        path (str): Ruta del modelo a cargar.

    Returns:
        Model: Modelo keras cargado.
    """
    model_path = get_abs_path(path)
    if not model_path.endswith(".json"):
        return models.load_model(model_path)
    with open(model_path, "r", encoding="utf-8") as file:
        json_model = file.read()
    return models.model_from_json(json_model)


def clone_model(model: Model) -> Model:
    """
    Clona un modelo de Keras.
    Args:
        model (Model): Modelo de Keras a clonar.

    Returns:
        Model: Modelo de Keras clonado y compilado.
    """
    cloned = models.clone_model(model)
    cloned.set_weights(model.get_weights())
    cloned.compile_from_config(model.get_compile_config())
    return cloned


def evaluate_model(
    model: Model, x_test: list, y_test: list, batch_size: int = 32
) -> tuple[float, dict]:
    """
    Evalua un modelo de Keras.

    Args:
        model (Model): Modelo de Keras a evaluar.
        x_test (list): Datos de entrada para la evaluación.
        y_test (list): Etiquetas para la evaluación.
        batch_size (int): Tamaño del lote para la evaluación.

    Returns:
        dict: Resultados de la evaluación.
    """
    # Calcular la perdida
    loss, _ = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)

    # Calcular las metricas
    metrics = {}
    y_pred_probs = model.predict(x_test, batch_size=batch_size)
    if y_pred_probs.shape[1] == 1:
        y_pred = (y_pred_probs > 0.5).astype("int32").flatten()
        y_true = y_test.flatten()
    else:
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average="weighted")
    metrics["recall"] = recall_score(y_true, y_pred, average="weighted")
    metrics["f1_score"] = f1_score(y_true, y_pred, average="weighted")
    metrics["auc"] = (
        roc_auc_score(y_true, y_pred_probs)
        if y_pred_probs.shape[1] == 1
        else roc_auc_score(y_true, y_pred_probs, multi_class="ovr")
    )
    metrics["confusion_matrix"] = str(confusion_matrix(y_true, y_pred).tolist())

    return (loss, metrics)
