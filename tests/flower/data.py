from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

NUM_CLIENTS = 10
BATCH_SIZE = 32


def fn(x):
    return (-x * (x - 6) * x) + (30 * np.sin(3 * x))


def load_datasets(points: int, limit: float = 0.8):
    limit = int(points * limit)

    # Generar los puntos
    X = np.random.uniform(-10.0, 10.0, points)
    np.random.shuffle(X)

    # Generar las etiquetas
    y = fn(X)

    # Crear los datasets
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
    train = (
        tf.data.Dataset.from_tensor_slices((X_tensor[:limit], y_tensor[:limit]))
        .shuffle(len(X_tensor[:limit]), 0)
        .batch(BATCH_SIZE)
    )
    test = tf.data.Dataset.from_tensor_slices(
        (X_tensor[limit:], y_tensor[limit:])
    ).batch(BATCH_SIZE)
    return train, test


def load_clients_datasets(points: int, limit: float = 0.8):
    limit = int(points * limit)

    # Generar los puntos
    X = np.linspace(-10.0, 10.0, points * NUM_CLIENTS)
    np.random.shuffle(X)

    # Generar las etiquetas
    y = fn(X)

    # Crear las particiones de los datos en diferentes clientes
    partitions_X = np.array_split(X, NUM_CLIENTS)
    partitions_y = np.array_split(y, NUM_CLIENTS)

    # Crear los datasets para cada cliente
    train = []
    test = []
    for i in range(NUM_CLIENTS):
        X_tensor = tf.convert_to_tensor(partitions_X[i], dtype=tf.float32)
        y_tensor = tf.convert_to_tensor(partitions_y[i], dtype=tf.float32)
        trainDataset = (
            tf.data.Dataset.from_tensor_slices((X_tensor[:limit], y_tensor[:limit]))
            .shuffle(points, 0)
            .batch(BATCH_SIZE)
        )
        train.append(trainDataset)
        testDataset = tf.data.Dataset.from_tensor_slices(
            (X_tensor[limit:], y_tensor[limit:])
        ).batch(BATCH_SIZE)
        test.append(testDataset)
    return train, test


def show_clients_data(data):
    colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "pink",
        "brown",
        "gray",
        "cyan",
        "lime",
    ]

    plt.figure(figsize=(10, 5))
    plt.title("Data distribution")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.grid()
    for i, client in enumerate(data):
        for j, (X, y) in enumerate(client):
            plt.scatter(
                X,
                y,
                color=colors[i % len(colors)],
                label=f"Client {i + 1} batch {j + 1}",
            )
    plt.legend()
    plt.show()
