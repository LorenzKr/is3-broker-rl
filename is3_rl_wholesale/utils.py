from pathlib import Path

import is3_rl_wholesale


def get_root_path() -> Path:
    """
    This is the path to the root directory containing the pyproject.toml file.
    :return: Path to the root directory
    """
    return Path(is3_rl_wholesale.__file__).parent.parent
