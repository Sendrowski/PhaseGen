import logging
import sys
from pathlib import Path


def prioritize_installed_packages():
    """
    This function prioritizes installed packages over local packages.
    """
    # Get the current working directory
    cwd = str(Path().resolve())

    # Check if the current working directory is in sys.path
    if cwd in sys.path:
        # Remove the current working directory from sys.path
        sys.path = [p for p in sys.path if p != cwd]
        # Append the current working directory to the end of sys.path
        sys.path.append(cwd)


# run before importing fastdfe
prioritize_installed_packages()

import fastdfe

logger = logging.getLogger('fastdfe')

logger.info(sys.version)
logger.info(f"Running tests for {fastdfe.__file__}")
logger.info(f"fastdfe version: {fastdfe.__version__}")
