import os
import sys
import torch
import numpy as np
import torch.nn as nn

from typing import Optional
from torch.utils.model_zoo import load_url as load_state_dict_from_url

from .model import BaseModel


class CNNModel(BaseModel):
    """
        CNN image-classification model for pre-training stage
        currently implemented AlexNet backend, which is default choice of R-CNN
    """

    def __init__(self, cfg):
        super(CNNModel, self).__init__(cfg)
        self._construct_network()
        # self._init_weights()

    def _construct_feature_extractor(self):
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def _construct_network(self):
        self.logger.info('Constructing AlexNet Backend for CNN Part...')
        self._construct_feature_extractor()
        self.pool = nn.AdaptiveAvgPool2d((6, 6))

        self.drop8 = nn.Dropout()
        self.fn8 = nn.Linear(256 * 6 * 6, 4096)
        self.active8 = nn.ReLU(inplace=True)

        self.drop9 = nn.Dropout()
        self.fn9 = nn.Linear(4096, 4096)
        self.active9 = nn.ReLU(inplace=True)

        self.fn10 = nn.Linear(4096, self.cfg.output_size)

    def _init_weights(self, ckpt: Optional[str]):
        if ckpt is None or not ckpt.startswith('http'):
            self.logger.error(
                'Bad Checkpoint URL for Pre-Trained Model.', stack_info=True)
            raise ValueError
        self.logger.info("Downloading Pre-Trained Model Params from %s" % ckpt)
        state_dict = load_state_dict_from_url(
            ckpt, progress=True)
        current_state = self.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            if key.startswith('features'):
                current_state[key] = state_dict[key]
        current_state['fn8.weight'] = state_dict['classifier.1.weight']
        current_state['fn8.bias'] = state_dict['classifier.1.bias']
        current_state['fn9.weight'] = state_dict['classifier.4.weight']
        current_state['fn9.bias'] = state_dict['classifier.4.bias']

    def forward(self, x):
        """
            x: [batch_size, 3, img_size, img_size]
        """
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.active8(self.fn8(self.drop8(x)))

        feature = self.active9(self.fn9(self.drop9(x)))
        final = self.fn10(feature)

        return feature, final


class FtCNNModel(CNNModel):
    def __init__(self, cfg):
        super(FtCNNModel, self).__init__(cfg)

    def _init_weights(self, ckpt: str, flag=False):
        if isinstance(ckpt, str):
            if not os.path.exists(ckpt):
                self.logger.error(
                    'Bad Checkpoint Path for Supervised Pre-Trained Model.', stack_info=True)
                raise ValueError
            self.logger.info(
                "Loading Checkpoint from Supervised Pre-Trained Model at %s" % ckpt)
            ckpt = torch.load(ckpt)
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        # Robust process for various kinds of checkpoint
        current_state = self.state_dict()
        keys = list(ckpt.keys())
        for key in keys:
            if flag or not key.startswith('fn10'):
                current_state[key] = ckpt[key]
