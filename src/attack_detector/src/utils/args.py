from argparse import ArgumentParser
from utils.path import get_path, get_rel_path


def parse_args():
    """
    Parsear los argumentos de la línea de comandos
    """

    default_model_path = get_rel_path(get_path("../data/model.json"))
    default_train_path = get_rel_path(
        *get_path("../data/x_train.csv", "../data/y_train.csv")
    )
    default_test_path = get_rel_path(
        *get_path("../data/x_test.csv", "../data/y_test.csv")
    )

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

    # Argumento para la ruta del modelo
    parser.add_argument(
        "-t",
        "--train",
        nargs=2,
        default=default_train_path,
        type=str,
        help=f"Ruta al archivo que contiene el conjunto de entrenamiento (default: {default_train_path[0]} {default_train_path[1]})",
        metavar=("X_TRAIN", "Y_TRAIN"),
    )
    parser.add_argument(
        "-T",
        "--test",
        nargs=2,
        default=default_test_path,
        type=str,
        help=f"Ruta al archivo que contiene el conjunto de prueba (default: {default_test_path[0]} {default_test_path[1]})",
        metavar=("X_TEST", "Y_TEST"),
    )

    return parser.parse_args()
