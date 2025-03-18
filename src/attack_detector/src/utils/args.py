from argparse import ArgumentParser

from utils.path import get_path, get_rel_path


def parse_args():
    """
    Parsear los argumentos de la línea de comandos
    """

    default_model_path = get_rel_path(get_path("../data/model.json"))

    # Crear el parser de argumentos
    parser = ArgumentParser(
        prog="Attack Detector",
        description="Entrenamiento de la red neuronal de detección de ataques en la red mediante aprendizaje federado",
    )

    # Argumento para la ruta del modelo
    parser.add_argument(
        "-m",
        "--model",
        default=default_model_path,
        type=str,
        help=f"Ruta al archivo que contiene el modelo de la red neuronal (default: {default_model_path})",
        metavar="MODEL",
    )

    return parser.parse_args()
