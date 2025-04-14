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
    default_output_path = get_rel_path(get_path("../data/output"))

    # Crear el parser de argumentos
    parser = ArgumentParser(
        prog="Attack Detector",
        description="Entrenamiento de la red neuronal de detección de ataques en la red mediante aprendizaje federado",
    )

    # Opciones de inicialización del programa
    config = parser.add_mutually_exclusive_group(required=True)

    # Argumento para inicializar el servidor central
    config.add_argument(
        "-s",
        "--server",
        type=str,
        const="127.0.0.1:8080",
        nargs="?",
        help="Ejecutar el servidor de entrenamiento con la opción HOST:PORT (default: 127.0.0.1:8080)",
        metavar="HOST:PORT",
    )

    # Argumento para inicializar el cliente
    config.add_argument(
        "-c",
        "--client",
        type=str,
        const="127.0.0.1:8080",
        nargs="?",
        help="Ejecutar el cliente de entrenamiento con la opción HOST:PORT (default: 127.0.0.1:8080)",
        metavar="HOST:PORT",
    )

    # Argumento para dividir el conjunto de entrenamiento
    config.add_argument(
        "-d",
        "--divide",
        type=int,
        const=5,
        nargs="?",
        help="Dividir el conjunto de entrenamiento en NUM_FILES partes (default: 5)",
        metavar="NUM_FILES",
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

    # Argumento para la ruta del dataset
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

    # Argumento para la ruta de salida
    parser.add_argument(
        "-o",
        "--output",
        default=default_output_path,
        type=str,
        help=f"Ruta al directorio de salida (default: {default_output_path})",
        metavar="OUTPUT",
    )

    # Argumento para el tamaño del lote
    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        help="Tamaño del lote de entrenamiento (default: 32)",
        metavar="BATCH_SIZE",
    )

    # Argumento para el número de rondas
    parser.add_argument(
        "--rounds",
        default=1,
        type=int,
        help="Número de rondas de entrenamiento (default: 1)",
        metavar="ROUNDS",
    )

    # Argumento para el número de épocas
    parser.add_argument(
        "--epochs",
        default=1,
        type=int,
        help="Número de épocas de entrenamiento (default: 1)",
        metavar="EPOCHS",
    )

    return parser.parse_args()
