from os import getcwd, path as ospath
from pathlib import Path


def get_path(path: str) -> str:
    """
    Devuelve la ruta absoluta basada en el archivo main del proyecto.

    Args:
        path: Ruta relativa al archivo main.

    Returns:
        str: Ruta absoluta.
    """
    return Path(__file__).resolve().parents[1].joinpath(path).resolve().as_posix()


def get_rel_path(path: str) -> str:
    """"
    Devuelve la ruta relativa basada en la ejecución del proyecto.

    Args:
        path: Ruta absoluta.

    Returns:
        str: Ruta relativa a la ejecución del proyecto.
    """
    return ospath.relpath(path, getcwd())
