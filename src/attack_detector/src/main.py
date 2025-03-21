from utils.model import load_model
from utils.dataset import load_dataset, split_dataset
from utils.args import parse_args

# Obtener los argumentos
args = parse_args()

# Dividir el conjunto de entrenamiento
if args.divide:
    split_dataset(*args.train, output=args.output, num_clients=args.divide)
else:
    # Cargar el modelo
    model = load_model(args.model)
    # Cargar los datos
    x_train, y_train = load_dataset(*args.train)
    x_test, y_test = load_dataset(*args.test)

    # Imprimir la información del modelo y los datos
    model.summary()
    print()
    print("\033[1mDataset\033[0m")
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
            args.rounds,
            args.batch_size,
        )
        server.start(args.server)
