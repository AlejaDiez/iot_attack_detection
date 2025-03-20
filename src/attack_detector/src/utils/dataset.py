import csv
import numpy as np
from utils.path import get_path


def load_dataset(*path: str) -> list | tuple[list]:
    dataset_path = get_path(*path)
    if len(path) == 1:
        with open(dataset_path, "r") as csvfile:
            reader = csv.reader(csvfile)
            dataset = []
            # Salta la cabecera
            next(reader)
            # Lee el resto de filas
            for row in reader:
                dataset.append(
                    np.array([np.float64(value) for value in row])
                )  # Convierte cada fila a un array de numpy de tipo float64
            return np.array(dataset)
    else:
        datasets = []
        for p in dataset_path:
            with open(p, "r") as csvfile:
                reader = csv.reader(csvfile)
                dataset = []
                # Salta la cabecera
                next(reader)
                # Lee el resto de filas
                for row in reader:
                    dataset.append(
                        np.array([np.float64(value) for value in row])
                    )  # Convierte cada fila a un array de numpy de tipo float64
                datasets.append(np.array(dataset))
        return tuple(datasets)
