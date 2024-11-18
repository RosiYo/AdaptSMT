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
from adapt.utils.cfg import parse_dataset_arguments


def parse_args() -> SimpleNamespace:
    """Parse the arguments for the adaptation script."""
    return SimpleNamespace(
        dataset=dict(
            config=parse_dataset_arguments("config/adaptation.json"),
            fold=0
        ),
        model=WrapperAdaptSMTConfig(
            checkpoint="smt-camera-grandstaff",
        )
    )


def get_trainer():
    wandb_logger = WandbLogger(
        project='SMTPP',
        group="Polish_Scores",
        name=f"SMTPP_Polish_Scores_Bekern_f{fold}",
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
        filename=f"SMTPP_Polish_Scores_Bekern_f{fold}",
        monitor="val_SER",
        mode='min',
        save_top_k=1,
        verbose=True
    )

    return Trainer(
        max_epochs=100000,
        check_val_every_n_epoch=3500,
        logger=wandb_logger,
        callbacks=[checkpointer, early_stopping],
        precision='16-mixed'
    )


if __name__ == "__main__":
    args = parse_args()
    dataset = AdaptDataset(**args.dataset)
    model = WrapperAdaptSMT(args.model)
    trainer = get_trainer()
    data.train_dataset.set_trainer_data(trainer)
    trainer.fit(model_wrapper, datamodule=data)
