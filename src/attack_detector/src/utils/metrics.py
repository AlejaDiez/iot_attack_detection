import csv
import json
import os
from keras import Model
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, roc_curve


def metrics_to_csv(path: str, output: str):
    """
    Convertir un archivo JSON con métricas de entrenamiento y evaluación a dos archivos CSV.

    Args:
        path (str): Ruta al archivo JSON de métricas de entrada.
        output (str): Ruta al directorio de salida.
    """

    def write_csv(path: str, headers: list[str], rows: list[list[any]]):
        """
        Escribe una lista de filas en un archivo CSV.

        Args:
            path (str): Ruta al archivo CSV de salida.
            headers (list[str]): Lista de encabezados de columna.
            rows (list[list[any]]): Lista de filas a escribir.
        """

        with open(path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(rows)

    def extract_training_data(metrics: dict) -> list[list[any]]:
        """
        Extrae los datos de entrenamiento de las métricas.

        Args:
            metrics (dict): Diccionario de métricas.
        Returns:
            list[list[any]]: Lista de filas con los datos de entrenamiento.
        """

        rows = []
        for round_id, data in metrics.items():
            training = data.get("training", {})
            clients = training.get("clients", {})

            for client_id, client_data in clients.items():
                accs = client_data.get("accuracy", [])
                losses = client_data.get("loss", [])

                for epoch, (acc, loss) in enumerate(zip(accs, losses)):
                    rows.append([round_id, client_id, epoch, acc, loss])
        return rows

    def extract_evaluation_data(metrics: dict) -> list[list[any]]:
        """
        Extrae los datos de evaluación de las métricas.

        Args:
            metrics (dict): Diccionario de métricas.
        Returns:
            list[list[any]]: Lista de filas con los datos de evaluación.
        """

        rows = []
        for round_id, data in metrics.items():
            evaluation = data.get("evaluation", {})

            def get_row(source: dict, device_id: str) -> list[any]:
                cm = source.get("confusion_matrix", [[0, 0], [0, 0]])
                return [
                    round_id,
                    device_id,
                    source.get("num_examples", 0),
                    source.get("loss", 0),
                    source.get("accuracy", 0),
                    source.get("precision", 0),
                    source.get("recall", 0),
                    source.get("f1_score", 0),
                    source.get("auc", 0),
                    cm[0][0],
                    cm[0][1],
                    cm[1][0],
                    cm[1][1],
                ]

            # Server
            rows.append(get_row(evaluation, "server"))

            # Clients
            for client_id, client_data in evaluation.get("clients", {}).items():
                rows.append(get_row(client_data, client_id))

        return rows

    # Obtener el nombre del archivo sin la extensión y el prefijo
    file_name = os.path.splitext(os.path.basename(path))[0].removesuffix("_metrics")
    # Leer el archivo JSON
    with open(path, "r") as file:
        metrics = json.load(file)
    # Crea el directorio de salida si no existe
    os.makedirs(output, exist_ok=True)

    # Metricas de entrenamiento
    path = os.path.join(output, f"{file_name}_training_metrics.csv")
    headers = ["round", "device", "epoch", "accuracy", "loss"]
    rows = extract_training_data(metrics)
    write_csv(path, headers, rows)

    # Metricas de evaluación
    path = os.path.join(output, f"{file_name}_evaluation_metrics.csv")
    headers = [
        "round",
        "device",
        "num_examples",
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auc",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ]
    rows = extract_evaluation_data(metrics)
    write_csv(path, headers, rows)


def metrics_to_graph(
    path: str, output: str, model: Model, test_data: tuple[list, list]
):
    def extract_metrics(metrics: dict) -> tuple[dict, dict]:
        """
        Extrae las métricas de entrenamiento y evaluación del archivo JSON.

        Args:
            path (dict): Diccionario con las métricas de entrenamiento y evaluación.

        Returns:
            tuple[dict, dict]: Diccionarios con las métricas de entrenamiento y evaluación.
        """
        training = {}
        evaluation = {}

        for round_metrics in metrics.values():
            # Entrenamiento
            clients = round_metrics.get("training", {}).get("clients", {})
            for id, metrics in clients.items():
                training.setdefault(id, {"accuracy": [], "loss": []})
                training[id]["accuracy"].extend(metrics.get("accuracy", []))
                training[id]["loss"].extend(metrics.get("loss", []))

            # Evaluación
            server = round_metrics.get("evaluation", {})
            evaluation.setdefault(
                "server",
                {
                    "accuracy": [],
                    "loss": [],
                    "precision": [],
                    "recall": [],
                    "f1_score": [],
                    "auc": [],
                    "confusion_matrix": [],
                },
            )
            for key in evaluation["server"]:
                evaluation["server"][key].append(server.get(key))

            clients = round_metrics.get("evaluation", {}).get("clients", {})
            for id, metrics in clients.items():
                evaluation.setdefault(
                    id,
                    {
                        "accuracy": [np.nan],
                        "loss": [np.nan],
                        "precision": [np.nan],
                        "recall": [np.nan],
                        "f1_score": [np.nan],
                        "auc": [np.nan],
                        "confusion_matrix": [np.nan],
                        "num_examples": [np.nan],
                    },
                )
                for metric, value in metrics.items():
                    evaluation[id][metric].append(value)

        return training, evaluation

    def plot_metrics(
        x: list[float],
        y: dict[str, list[float]],
        title: str,
        xlabel: str,
        ylabel: str,
        path: str,
    ):
        """
        Genera un gráfico de líneas a partir de los datos proporcionados.

        Args:
            x (list[float]): Lista de valores en el eje x.
            y (dict[str, list[float]]): Diccionario con etiquetas como claves y listas de valores como valores.
            title (str): Título del gráfico.
            xlabel (str): Etiqueta del eje x.
            ylabel (str): Etiqueta del eje y.
            path (str): Ruta donde se guardará la imagen del gráfico.
        """
        plt.figure(figsize=(10, 6))
        for label, values in y.items():
            plt.plot(x, values, label=label)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def confusion_matrix_metrics(
        values: list[list[float]],
        title: str,
        xlabel: list[str],
        ylabel: list[str],
        path: str,
    ):
        """
        Genera una matriz de confusión a partir de los datos proporcionados.

        Args:
            values (list[list[float]]): Lista de listas con los valores de la matriz de confusión.
            title (str): Título del gráfico.
            xlabel (str): Etiqueta del eje x.
            ylabel (str): Etiqueta del eje y.
            path (str): Ruta donde se guardará la imagen del gráfico.
        """
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            values,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=xlabel,
            yticklabels=ylabel,
        )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def device_name(id: str) -> str:
        """
        Devuelve el nombre del dispositivo a partir de su ID.

        Args:
            id (str): ID del dispositivo.

        Returns:
            str: Nombre del dispositivo.
        """
        return f"Client {id}" if id != "server" else "Server"

    def roc_curve_metrics(
        model: Model, test_data: tuple[list, list], title: str, path: str
    ):
        """
        Genera la curva ROC a partir de los datos proporcionados.

        Args:
            model (Model): Modelo de Keras utilizado para la predicción.
            test_data (tuple[list, list]): Datos de prueba (características y etiquetas).
            title (str): Título del gráfico.
            path (str): Ruta donde se guardará la imagen del gráfico.
        """
        y_true = test_data[1]
        y_scores = model.predict(test_data[0])
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="red", linestyle="--")
        plt.title(title)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    # Obtener el nombre del archivo sin la extensión y el prefijo
    file_name = os.path.splitext(os.path.basename(path))[0].removesuffix("_metrics")

    # Leer el archivo JSON
    with open(path, "r") as file:
        metrics = json.load(file)

    # Crea el directorio de salida si no existe
    os.makedirs(os.path.join(output, "graphs", "training"), exist_ok=True)
    os.makedirs(os.path.join(output, "graphs", "evaluation"), exist_ok=True)
    os.makedirs(os.path.join(output, "graphs", "training_evaluation"), exist_ok=True)

    # Extraer gráficas
    training, evaluation = extract_metrics(metrics)

    # Gráficas de entrenamiento
    for id, metrics in training.items():
        for metric in ["accuracy", "loss"]:
            plot_metrics(
                list(range(len(metrics[metric]))),
                {metric.capitalize(): metrics[metric]},
                f"{metric.capitalize()} ({device_name(id)})",
                "Epoch",
                "Value",
                os.path.join(
                    output,
                    "graphs",
                    "training",
                    f"{file_name}_graph_{metric}_{id}.png",
                ),
            )
        plot_metrics(
            list(range(len(metrics["accuracy"]))),
            {"Accuracy": metrics["accuracy"], "Loss": metrics["loss"]},
            f"Accuracy and Loss ({device_name(id)})",
            "Epoch",
            "Value",
            os.path.join(
                output,
                "graphs",
                "training",
                f"{file_name}_graph_accuracy_loss_{id}.png",
            ),
        )
    for metric in ["accuracy", "loss"]:
        plot_metrics(
            list(range(len(next(iter(training.values()))[metric]))),
            {
                f"{metric.capitalize()} ({device_name(id)})": metrics[metric]
                for id, metrics in training.items()
            },
            metric.capitalize(),
            "Epoch",
            "Value",
            os.path.join(
                output, "graphs", "training", f"{file_name}_graph_{metric}.png"
            ),
        )

    # Gráficas de evaluación
    for id, metrics in evaluation.items():
        for metric in ["accuracy", "loss", "precision", "recall", "f1_score", "auc"]:
            plot_metrics(
                list(range(len(metrics[metric]))),
                {metric.capitalize(): metrics[metric]},
                f"{metric.capitalize().replace('_', '-')} ({device_name(id)})",
                "Round",
                "Value",
                os.path.join(
                    output,
                    "graphs",
                    "evaluation",
                    f"{file_name}_graph_{metric}_{id}.png",
                ),
            )
        confusion_matrix_metrics(
            metrics["confusion_matrix"][-1],
            f"Confusion Matrix ({device_name(id)})",
            ["Predicted 0", "Predicted 1"],
            ["True 0", "True 1"],
            os.path.join(
                output,
                "graphs",
                "evaluation",
                f"{file_name}_graph_confusion_matrix_{id}.png",
            ),
        )
    for metric in ["accuracy", "loss", "precision", "recall", "f1_score", "auc"]:
        plot_metrics(
            list(range(len(evaluation["server"][metric]))),
            {
                f"{metric.capitalize()} ({device_name(id)})": metrics[metric]
                for id, metrics in evaluation.items()
            },
            metric.capitalize(),
            "Round",
            "Value",
            os.path.join(
                output, "graphs", "evaluation", f"{file_name}_graph_{metric}.png"
            ),
        )
    roc_curve_metrics(
        model,
        test_data,
        "ROC curve",
        os.path.join(
            output, "graphs", "evaluation", f"{file_name}_graph_roc_curve.png"
        ),
    )

    # Graficas de entrenamiento y evaluación
    for metric in ["accuracy", "loss"]:
        for id in training.keys():
            training_data = training[id]
            evaluation_data = evaluation[id]
            rounds = len(evaluation_data[metric]) - 1
            epochs_per_round = len(training_data[metric]) // rounds

            plot_metrics(
                list(range(1, rounds + 1)),
                {
                    f"Training {metric.capitalize()}": [
                        sum(training_data[metric][i : i + epochs_per_round])
                        / epochs_per_round
                        for i in range(0, len(training_data[metric]), epochs_per_round)
                    ],
                    f"Evaluation {metric.capitalize()}": evaluation_data[metric][1:],
                },
                f"Training and Evaluation {metric.capitalize()} ({device_name(id)})",
                "Round",
                "Value",
                os.path.join(
                    output,
                    "graphs",
                    "training_evaluation",
                    f"{file_name}_graph_{metric}_{id}.png",
                ),
            )
    for metric in ["accuracy", "loss"]:
        rounds = len(evaluation["server"][metric]) - 1

        plot_metrics(
            list(range(1, rounds + 1)),
            {
                **{
                    f"Training {metric.capitalize()} ({device_name(id)})": [
                        sum(metrics[metric][i : i + (len(metrics[metric]) // rounds)])
                        / (len(metrics[metric]) // rounds)
                        for i in range(
                            0, len(metrics[metric]), len(metrics[metric]) // rounds
                        )
                    ]
                    for id, metrics in training.items()
                },
                **{
                    f"Evaluation {metric.capitalize()} ({device_name(id)})": metrics[
                        metric
                    ][1:]
                    for id, metrics in evaluation.items()
                },
            },
            f"Training and Evaluation {metric.capitalize()}",
            "Round",
            "Value",
            os.path.join(
                output,
                "graphs",
                "training_evaluation",
                f"{file_name}_graph_{metric}.png",
            ),
        )
