import os
import torch
import torch.nn as nn

from .cnn import FtCNNModel
from .model import BaseModel


class SVMModel(BaseModel):
    def __init__(self, cfg, class_idx: int):
        super(SVMModel, self).__init__(cfg)
        self.class_num = class_idx
        self._construct_network()

    def _construct_network(self):
        self.logger.info('Constructing Linear SVM for class %s' %
                         self.class_num)
        self.classifier = nn.Linear(4096, 2)

    def _init_weights(self, ckpt, class_idx):
        if class_idx != self.class_num:
            self.logger.error('Bad Checkpoint for Linear SVM of class %s (Got %s)' % (
                self.class_num, class_idx))
            raise ValueError
        if isinstance(ckpt, str):
            if not os.path.exists(ckpt):
                self.logger.error(
                    'Bad Checkpoint Path for Linear SVM %s.' % self.class_num, stack_info=True)
                raise ValueError
            self.logger.info(
                "Loading Checkpoint from Linear SVM Model at %s" % ckpt)
            ckpt = torch.load(ckpt)
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        self.load_state_dict(ckpt, strict=True)

    def forward(self, feature):
        """
            feature: [batch_size, 4096]
        """
        outputs = self.classifier(feature)
        return outputs
