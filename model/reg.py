import os
import sys
import torch
import numpy as np
import torch.nn as nn

from .model import BaseModel


class RegModel(BaseModel):
    def __init__(self, cfg):
        super(RegModel, self).__init__(cfg)
        self._construct_network()

    def _construct_network(self):
        self.logger.info('Constructing Bounding-box Regression Network...')
        layers = list()
        fc1 = nn.Linear(4096, 4096)
        fc1.weight.data.normal_(0.0, 0.01)
        layers.append(fc1)
        layers.append(nn.Dropout(0.5))
        layers.append(nn.Tanh())
        fc2 = nn.Linear(4096, self.cfg.output_size)
        fc2.weight.data.normal_(0.0, 0.01)
        layers.append(fc2)
        layers.append(nn.Tanh())
        self.logits = nn.Sequential(*layers)

    def forward(self, x):
        return self.logits(x)
