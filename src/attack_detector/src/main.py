from utils.dataset import load_dataset
from utils.model import load_model
from utils.args import parse_args

# Obtener los argumentos
args = parse_args()

# Cargar el modelo
model = load_model(args.model)
print(model.summary())

# Cargar los datos
x_train, y_train = load_dataset(*args.train)
x_test, y_test = load_dataset(*args.test)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
