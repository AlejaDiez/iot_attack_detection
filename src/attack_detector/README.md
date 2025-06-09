# Attack Detector

Este directorio contiene el script que se encarga de entrenar un modelo de detección de ataques mediante aprendizaje federado.

## Requisitos

Para ejecutar el script, es necesario tener Python 3.8 o superior y las siguientes bibliotecas instaladas:

- `argparse`
- `tensorflow`
- `numpy`
- `flwr`
- `scikit-learn`
- `matplotlib`
- `seaborn`

Puedes instalar las dependencias necesarias ejecutando el siguiente comando:

```bash
pip install -r requirements.txt
```

## Ejecución

Para ejecutar el script, utiliza el siguiente comando:

```bash
python src/main.py ...
```

Reemplaza `...` con los argumentos necesarios para tu caso de uso.

```bash
-s [HOST:PORT], --server [HOST:PORT] Ejecutar el servidor de entrenamiento con la opción HOST:PORT (default: 127.0.0.1:8080)

-c [HOST:PORT], --client [HOST:PORT] Ejecutar el cliente de entrenamiento con la opción HOST:PORT (default: 127.0.0.1:8080)

-d [NUM_FILES], --divide [NUM_FILES] Dividir el conjunto de entrenamiento en NUM_FILES partes (default: 5)

-m [METRICS], --metrics [METRICS] Convierte las metricas en archivos csv y genera gráficas de estas (default: data/output/model_metrics.json)

--model MODEL Ruta al archivo que contiene el modelo de la red neuronal (default: data/model.json)

--train X_TRAIN Y_TRAIN Ruta al archivo que contiene el conjunto de entrenamiento (default: data/x_train.csv data/y_train.csv)

--test X_TEST Y_TEST Ruta al archivo que contiene el conjunto de prueba (default: data/x_test.csv data/y_test.csv)

--output OUTPUT Ruta al directorio de salida (default: data/output)

--batch-size BATCH_SIZE Tamaño del lote de entrenamiento (default: 32)

--rounds ROUNDS Número de rondas de entrenamiento (default: 1)

--epochs EPOCHS Número de épocas de entrenamiento (default: 1)
```
