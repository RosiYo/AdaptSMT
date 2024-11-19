"""
This module defines the abstract class for the loss functions.
author: Adrián Roselló Pedraza (RosiYo)
"""

from abc import ABC
import torch


class ILoss(ABC, torch.nn.Module):
    """Abstract class for the loss functions."""
