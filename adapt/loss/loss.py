"""
This module defines the abstract class for the loss functions.
author: Adrián Roselló Pedraza (RosiYo)
"""

from abc import ABC, abstractmethod
import torch


class ILoss(ABC, torch.nn.Module):
    """Abstract class for the loss functions."""

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Define the loss function calculation given any input arguments.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError(
            "Loss function does not implement any logic yet."
        )
