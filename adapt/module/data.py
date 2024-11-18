"""
Simple module to handle the adaptation dataset.
author: Adrián Roselló Pedraza (RosiYo)
"""

import torch
from lightning import LightningDataModule

from data import RealDataset, batch_preparation_img2seq
from utils.vocab_utils import check_and_retrieveVocabulary


class AdaptDataset(LightningDataModule):
    """Dataset for the adaptation task."""

    def __init__(self, config, fold):
        super().__init__()
        self.data_path = config.data.data_path
        self.vocab_name = config.data.vocab_name
        self.batch_size = config.data.batch_size
        self.num_workers = config.data.num_workers
        self.tokenization_mode = config.data.tokenization_mode
        self.train_dataset = RealDataset(
            data_path=self.data_path,
            split="train",
            augment=True,
            tokenization_mode=self.tokenization_mode,
            reduce_ratio=config.data.reduce_ratio
        )
        self.val_dataset = RealDataset(
            data_path=self.data_path,
            split="val",
            augment=False,
            tokenization_mode=self.tokenization_mode,
            reduce_ratio=config.data.reduce_ratio
        )
        self.test_dataset = RealDataset(
            data_path=self.data_path,
            split="test",
            augment=False,
            tokenization_mode=self.tokenization_mode,
            reduce_ratio=config.data.reduce_ratio
        )
        w2i, i2w = check_and_retrieveVocabulary([self.train_dataset.get_gt(
        ), self.val_dataset.get_gt(), self.test_dataset.get_gt()], "vocab/", f"{self.vocab_name}")

        self.train_dataset.set_dictionaries(w2i, i2w)
        self.val_dataset.set_dictionaries(w2i, i2w)
        self.test_dataset.set_dictionaries(w2i, i2w)

    def train_dataloader(self):
        """Return the training dataloader."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=batch_preparation_img2seq
        )

    def val_dataloader(self):
        """Return the validation dataloader."""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=batch_preparation_img2seq
        )

    def test_dataloader(self):
        """Return the test dataloader."""
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=batch_preparation_img2seq
        )
