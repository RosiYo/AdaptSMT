"""
Simple module to handle the adaptation dataset.
author: Adrián Roselló Pedraza (RosiYo)
"""

from data import FinetuningDataset


class AdaptDataset(FinetuningDataset):
    """Dataset for the adaptation task."""
    def __init__(self, config, fold):
        super().__init__(config=config, fold=fold)
        self.train_dataset.set_trainer_data(self.trainer)
