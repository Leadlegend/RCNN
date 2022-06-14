import os
import torch
import hydra

from logger import setup_logging
from trainer.base import Trainer
from data.datamodule import DataModule
from model import model_factory, criterion_factory
from config import args_util, init_optimizer


def train(cfg):
    model = model_factory[cfg.name](cfg.model)
    model._init_weights(cfg.trainer.ckpt)
    path = os.path.join(cfg.trainer.save_dir, 'pre-trained.pt')
    model._save_checkpoint(path)


@hydra.main(config_path='./config', config_name='train_step1')
def main(configs):
    setup_logging(save_dir=configs.logger.save_dir,
                  log_config=configs.logger.cfg_path,
                  file_name=configs.name+'.log')
    torch.set_printoptions(precision=5)
    train(cfg=configs)


if __name__ == '__main__':
    args_util()
    main()
