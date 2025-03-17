from argparse import ArgumentParser


def parse_args():
    # Crear el parser de argumentos
    parser = ArgumentParser(
        prog="Attack Detector",
        description="Entrenamiento de la red neuronal de detección de ataques en la red mediante aprendizaje federado",
    )

    # Agregar los argumentos

    return parser.parse_args()
