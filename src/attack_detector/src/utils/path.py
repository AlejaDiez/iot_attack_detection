from os import getcwd, path as ospath
from pathlib import Path


def get_path(*path: str) -> str | tuple[str]:
    """
    Devuelve la ruta absoluta basada en el archivo main del proyecto.

    Args:
        path: Ruta relativa al archivo main.

    Returns:
        str: Ruta absoluta.
    """
    if len(path) == 1:
        return Path(__file__).resolve().parents[1].joinpath(path[0]).resolve().as_posix()
    return tuple([Path(__file__).resolve().parents[1].joinpath(p).resolve().as_posix() for p in path])


def get_rel_path(*path: str) -> str | tuple[str]:
    """
    Devuelve la ruta relativa basada en la ejecución del proyecto.

    Args:
        path: Ruta absoluta.

    Returns:
        str: Ruta relativa a la ejecución del proyecto.
    """
    if len(path) == 1:
        return ospath.relpath(path[0], getcwd())
    return tuple([ospath.relpath(p, getcwd()) for p in path])

def get_abs_path(*path: str) -> str | tuple[str]:
    """
    Devuelve la ruta absoluta basada en la ejecución del proyecto.

    Args:
        path: Ruta relativa.

    Returns:
        str: Ruta absoluta a la ejecución del proyecto.
    """
    if len(path) == 1:
        return ospath.abspath(path[0])
    return tuple([ospath.abspath(p) for p in path])

