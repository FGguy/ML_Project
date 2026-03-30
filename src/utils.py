"""
Shared utility functions used across the pipeline.
"""

import yaml


def load_config(config_path: str) -> dict:
    """
    Load a YAML config file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Parsed config dictionary.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)
