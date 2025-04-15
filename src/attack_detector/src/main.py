"""
Attack Detector

Descripción: Este script se encarga de cargar el modelo y los datos, de crear el servidor y los clientes para el aprendizaje federado y de ejecutar el entrenamiento y de la división del conjunto de entrenamiento.
Autor: Alejandro Diez Bermejo
Fecha: 2025-03-01
Versión: 1.0
"""

import logging
from utils.args import parse_args


def main(args):
    from utils.dataset import load_dataset, split_dataset

    # Dividir el conjunto de entrenamiento
    if args.divide:
        split_dataset(*args.train, output=args.output, num_clients=args.divide)
        return

    from utils.model import load_model

    # Cargar el modelo
    model = load_model(args.model)
    model.summary()

    # Crear el servidor
    if args.server:
        from server import TensorFlowServer

        # Cargar los datos
        x_test, y_test = load_dataset(*args.test)
        print("\n", "\033[1mDataset\033[0m", sep="")
        print(" \033[1mTest:\033[0m", x_test.shape, y_test.shape, end="\n\n")

        # Inicializar el servidor
        server = TensorFlowServer(
            model,
            (x_test, y_test),
            args.output,
            args.batch_size,
            args.rounds,
        )
        server.start(args.server)

        # Guardar el modelo
        server.save_model(args.model.split("/")[-1].split(".")[0] + ".h5")
        # Guardar las métricas
        server.save_metrics(args.model.split("/")[-1].split(".")[0] + "_metrics.json")

    # Crear el cliente
    if args.client:
        from client import TensorFlowClient

        # Cargar los datos
        x_train, y_train = load_dataset(*args.train)
        x_test, y_test = load_dataset(*args.test)
        print("\n", "\033[1mDataset\033[0m", sep="")
        print(" \033[1mTrain:\033[0m", x_train.shape, y_train.shape)
        print(" \033[1mTest:\033[0m", x_test.shape, y_test.shape, end="\n\n")

        # Inicializar el cliente
        client = TensorFlowClient(
            model, (x_train, y_train), (x_test, y_test), args.batch_size, args.epochs
        )
        client.start(args.client)


if __name__ == "__main__":
    logging.disable()
    main(parse_args())
