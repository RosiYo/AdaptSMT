"""
Loss function to perform adaptation of layer normalization statistics.
author: Adrián Roselló Pedraza (RosiYo)
"""

import logging
from dataclasses import dataclass
from typing import List
import torch

from adapt.loss.loss import ILoss

logger = logging.getLogger(__name__)

@dataclass
class LayerNormStatistics:
    """Dataclass to store the layer normalization statistics."""
    mean: torch.Tensor
    variance: torch.Tensor

class LayerNormAdapt(ILoss):
    """
    Loss function to perform adaptation of layer normalization statistics.

    Args:
        model: The model to adapt the layer normalization statistics.
        layer_norm_indices: The indices of the layer normalization modules. Defaults to None.
    """

    __computed_weights: List[LayerNormStatistics] | None
    __lns_weights: List[LayerNormStatistics]
    __lns: List[int]

    def __init__(self, model: torch.nn.Module, layer_norm_indices: List[int] | None = None) -> None:
        super().__init__()
        self.__create_indices(model, layer_norm_indices)
        self.__computed_weights = None

    def __repr__(self) -> str:
        return f"LayerNormAdapt(ln_layers={len(self.ln_indices)})"

    @property
    def ln_indices(self) -> List[int]:
        """[PROPERTY] Get the indices of the layer normalization modules to adapt."""
        return self.__lns

    def clear_weights(self) -> None:
        """Function to clean the stored weights from outside the class."""
        self.__computed_weights.clear()

    def __are_valid_ln_indices(self, model: torch.nn.Module, lns: List[int] | None) -> bool:
        """
        Validate the layer normalization indices to adapt.

        Args:
            lns: The layer normalization indices to adapt.

        Returns:
            bool: Whether the indices are valid or not.
        """
        if lns is not None:
            try:
                for ln in lns:
                    module = model[ln]  # Assumes self.__model is indexable
                    if not module.__class__.__name__.endswith("LayerNorm"):
                        return False
            except (IndexError, KeyError, TypeError):
                return False
        return True

    def __hook_fn(self, module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        """
        Hook function to extract the layer normalization statistics.
        Appends the computed statistics to the list of computed weights.
        At the end of the forward pass, the statistics are cleared for
        the next forward pass.

        Args:
            module: The layer normalization module.
        """
        if self.__computed_weights is None:
            self.__computed_weights = [LayerNormStatistics(
                mean=module.weight.mean().item(),
                variance=module.weight.var().item()
            )]
        else:
            self.__computed_weights.append(LayerNormStatistics(
                mean=module.weight.mean().item(),
                variance=module.weight.var().item()
            ))

    def __create_indices(self, model: torch.nn.Module, lns: List[int] | None) -> None:
        """
        Create the layer normalization indices to adapt. If not provided, it will extract them.
        Model is also modified to allow forward hooks to extract the future statistics.

        Args:
            model: The model to adapt the layer normalization statistics.
            lns: The layer normalization indices to adapt.
        """
        if not self.__are_valid_ln_indices(model, lns):
            raise ValueError("Invalid layer normalization indices.")

        modules = list(model.encoder.modules())

        self.__lns_weights = []
        self.__lns = lns if lns is not None else [
            i for (i, module) in enumerate(modules)
            if module.__class__.__name__.endswith("LayerNorm")
        ]

        for i in self.__lns:
            self.__lns_weights.append(
                LayerNormStatistics(
                    mean=modules[i].weight.mean().item(),
                    variance=modules[i].weight.var().item()
                )
            )
            modules[i].register_forward_hook(self.__hook_fn)

    def dummy_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Dummy loss function to avoid the loss function to be empty."""
        return logits.sum() * 0

    # pylint: disable=arguments-differ
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Define the loss function calculation given any input arguments."""
        if self.__lns is None or self.__lns_weights is None:
            raise ValueError(
                "There has been a problem with the initial \
                    layer normalization statistics calculation."
            )

        # Stack the stored and computed statistics
        stored_means = torch.stack([ln.mean for ln in self.__lns_weights])
        stored_vars = torch.stack([ln.variance for ln in self.__lns_weights])
        computed_means = torch.stack(
            [cw.mean for cw in self.__computed_weights])
        computed_vars = torch.stack(
            [cw.variance for cw in self.__computed_weights])

        # Compute the loss in a batched manner
        log_term = torch.log(torch.sqrt(computed_vars) /
                             torch.sqrt(stored_vars))
        variance_term = (stored_vars + (stored_means -
                         computed_means).pow(2)) / (2 * computed_vars)
        loss = log_term + variance_term - 0.5

        # Return the mean loss across all layers
        return loss.mean() + self.dummy_loss(logits)
