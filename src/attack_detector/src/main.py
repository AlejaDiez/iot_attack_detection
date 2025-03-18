from utils.model import load_model
from utils.args import parse_args


# Obtener los argumentos
args = parse_args()

# Cargar el modelo
model = load_model(args.model)
print(model.summary())
