"""
This module contains the model for adapting the SMT model.
author: Adrián Roselló Pedraza (RosiYo)
"""

import torch

from smt_model.modeling_smt import SMTModelForCausalLM


class AdaptSMT(SMTModelForCausalLM):
    """Model for adapting the SMT model."""
    
    @property
    def trainable_params(self):
        """[PROPERTY] Get the trainable parameters of the model."""
        return [param for param in self.parameters() if param.requires_grad]

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_encoder(x)
        # TODO: Decoder implies a bigger challenge
        # output = self.forward_decoder(x, y_pred)
        return self.loss(x)
