"""
This file implements the adaptation logic of the SMTPP model. It focuses
on adaptation of encoder layers through different mechanisms.
author: Adrián Roselló Pedraza (RosiYo)
"""

from types import SimpleNamespace
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from adapt.module import AdaptDataset, WrapperAdaptSMT, WrapperAdaptSMTConfig
from adapt.module.data import GrandStaffIterableDataset
from adapt.utils.cfg import parse_dataset_arguments


def parse_args() -> SimpleNamespace:
    """Parse the arguments for the adaptation script."""
    return SimpleNamespace(
        dataset={
            "config": parse_dataset_arguments("config/Mozarteum/finetuning.json"),
            "fold": 0
        },
        trainer={
            "exp": "Mozaertum_f0"
        },
        model=WrapperAdaptSMTConfig(
            checkpoint="synthetic_mozarteum",
            source_proxy=GrandStaffIterableDataset(
                nsamples=100
            )
        )
    )


def get_trainer(exp: str) -> Trainer:
    """
    Get the trainer for the adaptation process.

    Args:
        exp: The experiment name.
    """

    wandb_logger = WandbLogger(
        entity='grfia',
        project="AdaptSMT",
        name=exp,
        log_model=False
    )

    early_stopping = EarlyStopping(
        monitor="val_SER",
        min_delta=0.01,
        patience=5,
        mode="min",
        verbose=True
    )

    checkpointer = ModelCheckpoint(
        dirpath="weights/finetuning/",
        filename=f"AdaptSMT_{exp}",
        monitor="val_SER",
        mode='min',
        save_top_k=1,
        verbose=True
    )

    return Trainer(
        max_epochs=1000,
        check_val_every_n_epoch=5,
        logger=wandb_logger,
        callbacks=[checkpointer, early_stopping],
        precision='16-mixed'
    )


if __name__ == "__main__":
    args = parse_args()
    dataset = AdaptDataset(**args.dataset)
    model = WrapperAdaptSMT(args.model)
    trainer = get_trainer(**args.trainer)
    trainer.fit(model, datamodule=dataset)
