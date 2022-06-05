import os
import logging

from functools import partial
from torch.utils.data import DataLoader

from .util import dataset_factory, collate_factory


class DataModule:
    def __init__(self, cfg, name):
        self.logger = logging.getLogger('%s-DataModule' % name)
        self.cfg = cfg
        self.name = name
        self._dataset = dataset_factory[name]
        self._collate_fn = collate_factory[name]
        self.epoch_flag = (name in ['alex', 'reg'])
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self.setup()

    def setup(self):
        if self.cfg.train is not None:
            self.logger.info("Constructing Train Data...")
            self.train_dataset = self._dataset(
                self.cfg.train)
        else:
            self.logger.warning('No Valid Train Data.')
        if self.cfg.val is not None:
            self.logger.info("Constructing Validation Data...")
            self.val_dataset = self._dataset(
                self.cfg.val)
        else:
            self.logger.warning('No Valid Val Data.')
        if self.cfg.test is not None:
            self.logger.info("Constructing Test Data...")
            self.test_dataset = self._dataset(
                self.cfg.test)
        else:
            self.logger.warning('No Valid Test Data.')

    def _construct_loader(self, cfg, dataset):
        if cfg is None:
            return None
        if self.epoch_flag:
            return DataLoader(dataset=dataset,
                              batch_size=cfg.batch_size,
                              collate_fn=self._collate_fn,
                              pin_memory=cfg.pin,
                              num_workers=cfg.workers,
                              shuffle=cfg.shuffle
                              )
        else:
            return dataset

    def train_dataloader(self):
        return self._construct_loader(self.cfg.train, self.train_dataset)

    def val_dataloader(self):
        return self._construct_loader(self.cfg.val, self.val_dataset)

    def test_dataloader(self):
        return self._construct_loader(self.cfg.test, self.test_dataset)
