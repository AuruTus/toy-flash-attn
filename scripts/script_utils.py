"""
Utility functions for scripts to locate the project root and set up imports.
"""

import sys
import pathlib


def find_project_root(start_path: pathlib.Path = None, marker: str = "toy_attn") -> pathlib.Path:
    """
    Find the project root directory by ascending from start_path until we find
    a directory containing the marker subdirectory (default: 'toy_attn').

    Args:
        start_path: Starting path for the search. Defaults to the caller's __file__ directory.
        marker: Name of the subdirectory that identifies the project root.

    Returns:
        Path to the project root directory.

    Raises:
        FileNotFoundError: If project root cannot be found.
    """
    if start_path is None:
        # Get the caller's directory
        import inspect
        frame = inspect.currentframe()
        caller_frame = frame.f_back
        caller_file = caller_frame.f_globals.get("__file__")
        if caller_file:
            start_path = pathlib.Path(caller_file).resolve().parent
        else:
            start_path = pathlib.Path.cwd()

    current = start_path.resolve()
    while current != current.parent:
        if (current / marker).is_dir():
            return current
        current = current.parent

    # Check root as well
    if (current / marker).is_dir():
        return current

    raise FileNotFoundError(
        f"Could not find project root (directory containing '{marker}') "
        f"starting from {start_path}"
    )


def setup_project_imports(start_path: pathlib.Path = None, marker: str = "toy_attn") -> pathlib.Path:
    """
    Find the project root and add it to sys.path for imports.

    Args:
        start_path: Starting path for the search.
        marker: Name of the subdirectory that identifies the project root.

    Returns:
        Path to the project root directory.
    """
    project_root = find_project_root(start_path, marker)
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root
