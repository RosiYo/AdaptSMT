"""
Loss function to perform adaptation of layer normalization statistics.
author: Adrián Roselló Pedraza (RosiYo)
"""

import logging
from dataclasses import dataclass
from typing import List
import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm

from adapt.loss.loss import ILoss

logger = logging.getLogger(__name__)


@dataclass
class LayerNormStatistics:
    """Dataclass to store the layer normalization statistics."""
    mean: float
    variance: float


class LayerNormAdapt(ILoss):
    """
    Loss function to perform adaptation of layer normalization statistics.

    Args:
        model: The model to adapt the layer normalization statistics.
        layer_norm_indices: The indices of the layer normalization modules. Defaults to None.
    """

    __computed_weights: List[LayerNormStatistics] | None
    __stored_means: torch.Tensor
    __stored_vars: torch.Tensor
    __lns: List[int]

    def __init__(
        self, 
        model: torch.nn.Module, 
        source_loader: IterableDataset, 
        layer_norm_indices: List[int] | None = None
    ) -> None:
        super().__init__()
        self.__computed_weights = []
        self.__create_indices(model, layer_norm_indices)
        self.__register_hooks(model)
        self.__compute_weights_from_source(model, source_loader)

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

        self.__lns = lns or [
            i for i, module in enumerate(model.encoder.modules())
            if module.__class__.__name__.endswith("ConvNextLayerNorm")
        ]

    def __compute_weights_from_source(self, model: torch.nn.Module, source: IterableDataset) -> None:
        with torch.no_grad():
            lns_weights = [LayerNormStatistics(
                0.0, 0.0) for _ in range(len(self.__lns))]
            for x, _ in tqdm(source):
                # Leverage registered hooks to extract the statistics
                x = x.to(model.device)
                model.encoder(x)
                for i, stats in enumerate(self.__computed_weights):
                    lns_weights[i].mean += stats.mean
                    lns_weights[i].variance += stats.variance
                self.clear_weights()

            print("Hola", lns_weights)
            for stats in lns_weights:
                stats.mean /= len(source)
                stats.variance /= len(source)

            self.__stored_means = torch.Tensor([ln.mean for ln in lns_weights])
            self.__stored_vars = torch.Tensor(
                [ln.variance for ln in lns_weights])

    def __hook_fn(self, module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        """
        Hook function to extract the layer normalization statistics.
        Appends the computed statistics to the list of computed weights.
        At the end of the forward pass, the statistics are cleared for
        the next forward pass.

        Args:
            module: The layer normalization module.
        """
        input = input[0] # Extract the input tensor
        self.__computed_weights.append(LayerNormStatistics(
            mean=input.mean().item(),
            variance=input.var().item()
        ))
        
    def __register_hooks(self, model: torch.nn.Module) -> None:
        """
        Register the forward hooks to extract the layer normalization statistics.
        
        Args:
            model: The model to adapt the layer normalization statistics.
        """
        for i, module in enumerate(model.modules()):
            if i in self.__lns:
                module.register_forward_hook(self.__hook_fn)
                
    def dummy_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Dummy loss function to avoid the loss function to be empty."""
        return logits.sum() * 0

    # pylint: disable=arguments-differ
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Define the loss function calculation given any input arguments."""

        # Compute the means and variances of the computed weights
        computed_means = torch.tensor(
            [cw.mean for cw in self.__computed_weights])
        computed_vars = torch.tensor(
            [cw.variance for cw in self.__computed_weights])

        # Compute the loss in a batched manner
        log_term = torch.log(torch.sqrt(computed_vars) /
                             torch.sqrt(self.__stored_vars))
        variance_term = (self.__stored_vars + (self.__stored_means -
                         computed_means).pow(2)) / (2 * computed_vars)
        loss = log_term + variance_term - 0.5

        # Return the mean loss across all layers
        return loss.mean() + self.dummy_loss(logits)
