import csv
from pathlib import Path
import numpy as np
from utils.path import get_abs_path


def load_dataset(*path: str) -> list | tuple[list]:
    """
    Carga un dataset de un archivo CSV

    Args:
        path (str): Ruta de los archivos CSV, si es un solo archivo, se pasa como un solo argumento, si son varios archivos, se pasan como argumentos separados

    Returns:
        list: Dataset cargado, si es un solo archivo se retorna una lista, si son varios archivos se retorna una tupla de listas
    """
    datasets = []
    dataset_path = get_abs_path(*path) if len(path) > 1 else tuple(get_abs_path(path))

    # Leer los archivos
    for p in dataset_path:
        with open(p, "r") as file:
            reader = csv.reader(file)
            dataset = []
            # Salta la cabecera
            next(reader)
            # Lee el resto de filas
            for row in reader:
                dataset.append(
                    np.array([np.float64(value) for value in row])
                )  # Convierte cada fila a un array de numpy de tipo float64
            datasets.append(np.array(dataset))
    return tuple(datasets) if len(datasets) > 1 else datasets[0]


def split_dataset(*paths: str, output: str, num_clients: int):
    """
    Divide un dataset en varios subconjuntos y los guarda en archivos CSV.

    Args:
        paths (str): Rutas de los archivos CSV, si son dos archivos, se pasan como dos argumentos, si es un solo archivo, se pasa como un solo argumento
        output (str): Ruta del directorio donde se guardarán los subconjuntos
        num_clients (int): Número de subconjuntos en los que se dividirá el dataset

    Raises:
        ValueError: Si el número de paths es mayor a 2
    """
    # Comprobar que el número de argumentos es correcto
    if len(paths) > 2:
        raise ValueError("El número de argumentos de la función debe ser 2 o inferior")

    # Crear directorio de salida si no existe
    output_dir = get_abs_path(output)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    datasets, headers = [], []
    for path in paths:
        with open(path, "r") as file:
            reader = csv.reader(file)
            headers.append(next(reader))
            datasets.append(list(reader))

    if len(paths) > 1:
        dataset = list(zip(*datasets))  # Agrupa los datos por pares
    else:
        dataset = datasets[0]

    # Mezclar los datos
    np.random.shuffle(dataset)
    # Dividir los datos
    num_clients = min(num_clients, len(dataset))
    chunk_sizes = [
        len(dataset) // num_clients + (i < len(dataset) % num_clients)
        for i in range(num_clients)
    ]

    # Generar los subconjuntos
    start = 0
    for i, size in enumerate(chunk_sizes):
        for j, (filename, header) in enumerate(zip(paths, headers)):
            # Escribir los datos en un archivo CSV
            with open(f"{output_dir}/{Path(filename).stem}_{i}.csv", "w") as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerows(
                    row[j] if len(paths) == 2 else row
                    for row in dataset[start : start + size]
                )
        start += size
