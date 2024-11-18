"""
Helper functions for parsing configurations.
author: Adrián Roselló Pedraza (RosiYo)
"""

import json

from ExperimentConfig import experiment_config_from_dict


def parse_dataset_arguments(config_path: str) -> dict:
    """Parse the arguments for the adaptation script."""
    with open(config_path, 'r', encoding="utf-8") as file:
        config_dict = json.load(file)
        config = experiment_config_from_dict(config_dict)
    return config
