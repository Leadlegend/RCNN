import os
import sys
import torch
import logging
import numpy as np
import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self, cfg):
        super(BaseModel, self).__init__()
        self.cfg = cfg
        self.logger = logging.getLogger(cfg.name)

    def _construct_network(self):
        raise NotImplementedError

    def _init_weights(self, ckpt):
        raise NotImplementedError

    def forward(self, drug_input):
        raise NotImplementedError

    def _save_checkpoint(self, path):
        state_dict = self.state_dict()
        torch.save(state_dict, path)
