"""
Utility functions for working with PyTorch models.
author: Adrián Roselló Pedraza (RosiYo)
"""

import torch


def get_trainable_params(model: torch.nn.Module) -> int:
    """
    Get the number of trainable parameters in the model.

    Args:
        model: The model to count the parameters of.

    Returns:
        int: The number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
