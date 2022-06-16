import os
import logging

from functools import partial
from torch.utils.data import DataLoader

from data import dataset_factory, collate_factory
from .tokenizer import Tokenizer


class WarpDataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self):
        return int(len(self.dataset) / self.batch_size) + 1

    def __iter__(self):
        return self

    def __next__(self):
        return self.dataset.get_batch(batch_size=self.batch_size)


class DataModule:
    def __init__(self, cfg, name, idx: int = -1):
        self.logger = logging.getLogger('%s-DataModule' % name)
        self.cfg = cfg
        self.name = name
        self.tokenizer = cfg.tokenizer
        if self.tokenizer is not None:
            self.tokenizer = Tokenizer(self.tokenizer)
        self._dataset = dataset_factory[name]
        self._collate_fn = collate_factory[name]
        self.epoch_flag = (name in ['alex', 'svm', 'reg'])
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        if idx > 0:
            self.setup(idx)
        else:
            self.setup()

    def setup(self, idx=None):
        if self.cfg.train is not None:
            self.logger.info("Constructing Train Data...")
            self.train_dataset = self._dataset(
                self.cfg.train, tokenizer=self.tokenizer, idx=idx)
        else:
            self.logger.warning('No Valid Train Data.')
        if self.cfg.val is not None:
            self.logger.info("Constructing Validation Data...")
            self.val_dataset = self._dataset(
                self.cfg.val, tokenizer=self.tokenizer, idx=idx)
        else:
            self.logger.warning('No Valid Val Data.')
        if self.cfg.test is not None:
            self.logger.info("Constructing Test Data...")
            self.test_dataset = self._dataset(
                self.cfg.test, tokenizer=self.tokenizer, idx=idx)
        else:
            self.logger.warning('No Valid Test Data.')

    def _construct_loader(self, cfg, dataset, sampler=None, flag=False):
        if cfg is None:
            return None
        if self.epoch_flag or flag:
            if sampler is None:
                return DataLoader(dataset=dataset, batch_size=cfg.batch_size,
                                  collate_fn=self._collate_fn, pin_memory=cfg.pin,
                                  num_workers=cfg.workers, shuffle=cfg.shuffle)
            else:
                return DataLoader(dataset=dataset, batch_size=cfg.batch_size,
                                  sampler=sampler, collate_fn=self._collate_fn,
                                  pin_memory=cfg.pin, num_workers=cfg.workers)
        else:
            return WarpDataLoader(dataset=dataset, batch_size=cfg.batch_size)

    def train_dataloader(self, sampler=None):
        return self._construct_loader(self.cfg.train, self.train_dataset, sampler=sampler)

    def val_dataloader(self, sampler=None):
        return self._construct_loader(self.cfg.val, self.val_dataset, sampler=sampler, flag=True)

    def test_dataloader(self):
        return self._construct_loader(self.cfg.test, self.test_dataset, flag=True)
