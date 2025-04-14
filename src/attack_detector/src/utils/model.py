from keras import Model, models
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
    with open(model_path, "r") as file:
        json_model = file.read()
    return models.model_from_json(json_model)
