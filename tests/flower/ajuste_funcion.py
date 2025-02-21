import matplotlib.pyplot as plt
import numpy as np
from model import keras_model
from data import load_datasets


##########################################
# Entrenamiento sin aprendizaje federado #
##########################################
# Cargar los datasets
train, test = load_datasets(1000)

# Entrenar el modelo
model = keras_model()
model.fit(train, epochs=200)

# Hacer predicciones
X = []
y = []
for X_, y_ in test:
    X = np.concatenate([X, X_.numpy()])
    y = np.concatenate([y, y_.numpy()])

predictions = model.predict(X)

# Mostrar las predicciones
plt.figure(figsize=(10, 5))
plt.title("Predicciones")
plt.xlabel("X")
plt.ylabel("y")
plt.grid()
for i in range(len(X)):
    plt.scatter(X[i], y[i], color="black")
    plt.scatter(X[i], predictions[i], color="red")
plt.show()
