"""
Attack Detector

Descripción: Este script se encarga de cargar el modelo y los datos, de crear el servidor y los clientes para el aprendizaje federado y de ejecutar el entrenamiento y de la división del conjunto de entrenamiento.
Autor: Alejandro Diez Bermejo
Fecha: 2025-03-01
Versión: 1.0
"""

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
    # Cargar los datos
    x_train, y_train = load_dataset(*args.train)
    x_test, y_test = load_dataset(*args.test)

    # Imprimir la información del modelo y los datos
    model.summary()
    print()
    print(
        "\033[1mDataset\033[0m",
    )
    print(" \033[1mTrain:\033[0m", x_train.shape, y_train.shape)
    print(" \033[1mTest:\033[0m", x_test.shape, y_test.shape)
    print()

    # Crear el servidor
    if args.server:
        from server import TensorFlowServer

        server = TensorFlowServer(
            model,
            (x_test, y_test),
            args.output,
            args.batch_size,
            args.rounds,
        )
        server.start(args.server)

    # Crear el cliente
    if args.client:
        from client import TensorFlowClient

        server = TensorFlowClient(
            model, (x_train, y_train), (x_test, y_test), args.batch_size, args.epochs
        )
        server.start(args.client)


if __name__ == "__main__":
    main(parse_args())
