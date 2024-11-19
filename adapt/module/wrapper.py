"""
Wrapper for the AdaptSMT model in Lightning.
author: Antonio Russo
"""
import random
from dataclasses import dataclass
from typing import Any, Dict
from lightning.pytorch import LightningModule
from torchinfo import summary
from torch.optim import Optimizer
import torch

from adapt.loss.layer_norm_adapt import LayerNormAdapt
from adapt.module.model import AdaptSMT
from adapt.utils import get_trainable_params
from eval_functions import compute_poliphony_metrics


@dataclass
class WrapperAdaptSMTConfig:
    """Dataclass to store the configuration of the SMT model."""
    checkpoint: str


class WrapperAdaptSMT(LightningModule):
    """LightningModule for adapting the SMT model."""

    __model: AdaptSMT
    __is_freezed: bool

    def __init__(self, cfg: WrapperAdaptSMTConfig) -> None:
        super().__init__()
        self.__cfg = cfg
        self.__preds = []
        self.__grtrs = []

    def setup(self, stage: str):
        """Initializes the model configuration"""
        if not hasattr(self, "__model"):
            self.__load_model(self.__cfg.checkpoint)
            self.__configure_model()

    @property
    def model(self) -> AdaptSMT:
        """[PROPERTY] Get the model."""
        return self.__model

    @property
    def is_freezed(self) -> bool:
        """[PROPERTY] Get whether the model is freezed or not."""
        return self.__is_freezed

    def configure_optimizers(self) -> Optimizer:
        return torch.optim.Adam(
            self.__model.trainable_params, lr=1e-4, amsgrad=False
        )

    def __load_model(self, weights_path: str) -> None:
        """
        Load the model from the given path.

        Args:
            weights_path: The path to the weights.
        """
        self.__model = AdaptSMT.from_pretrained(
            f"antoniorv6/{weights_path}"
        ).to(self.device)

    def freeze(self) -> None:
        """
        Freeze the parameters of the model.

        Args:
            model: The model to freeze.
        """
        initial_params = get_trainable_params(self.__model)

        # Freeze all encoder parameters except adapted layer normalization parameters
        for i, module in enumerate(self.model.encoder.modules()):
            if not i in self.__model.loss.ln_indices:
                for param in module.parameters():
                    param.requires_grad = False
            else:
                for param in module.parameters():
                    param.requires_grad = True

        # Freeze decoder parameters
        for param in self.__model.decoder.parameters():
            param.requires_grad = False

        final_params = get_trainable_params(self.__model)

        self.__is_freezed = True
        print(
            f"Reduced trainable parameters from {initial_params:,} to {final_params:,}."
        )

    def unfreeze(self) -> None:
        for param in self.__model.parameters():
            param.requires_grad = True
        self.__is_freezed = False

    def __configure_criteria(self) -> None:
        """
        Configure the loss criteria for the adaptation procedure.
        Currently only accepts LayerNormAdapt, but it will be extended in the future
        to support combinations of different criteria.
        """
        self.__model.loss = LayerNormAdapt(self.__model)

    def __configure_model(self) -> None:
        """Configure the model for adaptation."""
        print("\n", 80*"*", "\n")
        print("\tSETTING UP MODEL FOR ADAPTATION")
        print(80*"-")
        self.__configure_criteria()
        self.freeze()

    # pylint: disable=arguments-differ
    def training_step(self, batch):
        x, _, _, = batch
        self.__model.loss.clear_weights()
        loss = self.__model(x)
        self.log('loss', loss, on_epoch=True, batch_size=1, prog_bar=True)
        return loss

    def validation_step(self, val_batch):
        x, _, y = val_batch
        predicted_sequence, _ = self.model.predict(input=x, convert_to_str=True)

        dec = "".join(predicted_sequence)
        dec = dec.replace("<t>", "\t")
        dec = dec.replace("<b>", "\n")
        dec = dec.replace("<s>", " ")

        gt = "".join([self.model.i2w[str(token.item())]
                     for token in y.squeeze(0)[:-1]])
        gt = gt.replace("<t>", "\t")
        gt = gt.replace("<b>", "\n")
        gt = gt.replace("<s>", " ")

        self.__preds.append(dec)
        self.__grtrs.append(gt)

    def on_validation_epoch_end(self, metric_name="val") -> None:
        cer, ser, ler = compute_poliphony_metrics(self.__preds, self.__grtrs)

        random_index = random.randint(0, len(self.__preds)-1)
        predtoshow = self.__preds[random_index]
        gttoshow = self.__grtrs[random_index]
        print(f"[Prediction] - {predtoshow}")
        print(f"[GT] - {gttoshow}")

        self.log(f'{metric_name}_CER', cer, on_epoch=True, prog_bar=True)
        self.log(f'{metric_name}_SER', ser, on_epoch=True, prog_bar=True)
        self.log(f'{metric_name}_LER', ler, on_epoch=True, prog_bar=True)

        self.__preds.clear()
        self.__grtrs.clear()
        return ser

    def test_step(self, test_batch) -> torch.Tensor | Dict[str, Any] | None:
        return self.validation_step(test_batch)

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end("test")
