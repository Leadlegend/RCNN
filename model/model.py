import os
import sys
import torch
import logging
import numpy as np
import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self, cfg, device='cpu'):
        super(BaseModel, self).__init__()
        self.cfg = cfg
        self.device = device
        self._train_flag = cfg.train
        self.logger = logging.getLogger(cfg.name)

    @property
    def is_train(self):
        return self._train_flag

    def _construct_network(self):
        raise NotImplementedError

    def _init_weights(self, ckpt):
        raise NotImplementedError

    def forward(self, drug_input):
        raise NotImplementedError
