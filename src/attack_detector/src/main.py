from utils.model import load_model
from utils.path import get_abs_path
from utils.dataset import load_dataset, split_dataset
from utils.args import parse_args

# Obtener los argumentos
args = parse_args()

# Dividir el conjunto de entrenamiento
if args.divide:
    split_dataset(
        *get_abs_path(*args.train),
        output=get_abs_path(args.output),
        num_clients=args.divide
    )
else:
    # Cargar el modelo
    model = load_model(args.model)
    print(model.summary())

    # Cargar los datos
    x_train, y_train = load_dataset(*get_abs_path(*args.train))
    x_test, y_test = load_dataset(*get_abs_path(*args.test))
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
